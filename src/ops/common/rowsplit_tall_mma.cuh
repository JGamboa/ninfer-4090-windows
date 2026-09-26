#pragma once

// Pipelined Q4/Q5 RowSplit x BF16 tensor-core GEMM for prefill widths (one CTA per SM).
//
// A CTA of eight warps owns 128 weight rows and `Tokens` (128 or 64) tokens and walks K one
// 64-value quant group per step. Each step's weights are dequantized exactly as the staged-decode
// MMA kernels do it (bf16_rn(float(q) * float(scale))), stored to shared memory and multiplied
// with m16n8k16 BF16 MMAs into FP32 accumulators, one MMA per 16-wide K slice in K order. The
// accumulation of every output is therefore the same instruction sequence on the same operands
// as in those kernels, whatever the CTA and warp tiling, and the outputs are bit-identical.
//
// Pipeline: the dequantized weight tile and the activation tile are both double-buffered, so a
// step needs one barrier. Every thread decodes one half-group (16 code bytes, plus 4 high-bit
// bytes for Q5) of one weight row per step, from registers loaded a step earlier. Warps 0-3
// multiply before they decode the next step and warps 4-7 after, so each SM sub-partition runs
// one warp's MMAs while the other warp decodes. Tiles are launched token-tile fastest, so the
// CTAs of one row block run together and share its code bytes in L2.
//
// A Problem supplies the row sources of a row block, the epilogue's staged values and the global
// write; see the three problems at the end of this file.

#include "core/device.h"
#include "core/weight.h"
#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail::rowsplit_tall {

constexpr int kRows    = 128;
constexpr int kThreads = 256;
constexpr int kStepK   = 64;

template <int Tokens>
struct Config {
    static_assert(Tokens == 128 || Tokens == 64);
    static constexpr int kWarpsN   = Tokens / 32;
    static constexpr int kWarpsM   = 8 / kWarpsN;
    static constexpr int kWarpRows = kRows / kWarpsM;
    static constexpr int MT        = kWarpRows / 16;

    struct Stage {
        __nv_bfloat16 a[kRows * kStepK];   // 128-byte rows, 16-byte chunks XORed with row & 7
        __nv_bfloat16 x[Tokens * kStepK];  // the same swizzle per token
    };
    static constexpr std::size_t kSharedBytes = 2 * sizeof(Stage);
};

// One decode thread's weight row: its code bytes of group 0 (already offset by the half), the
// row's high-bit bytes of group 0 (Q5 only, offset by the half) and its FP16 group scales.
struct RowSource {
    const std::uint8_t* codes;
    const std::uint8_t* high;
    const std::uint16_t* scales;
};

// Epilogue staging row stride in bf16: 8 elements of padding keep the fragment stores
// conflict-free and the 16-byte row reads aligned.
template <int OutRows>
constexpr int kOutLd = OutRows + 8;

template <int Tokens, class Problem>
__global__ void __launch_bounds__(kThreads, 1)
    rowsplit_tall_mma_kernel(const __nv_bfloat16* __restrict__ x, Problem problem, std::int32_t k,
                             std::int32_t tokens, std::int32_t token_tiles) {
    using Cfg   = Config<Tokens>;
    using Stage = typename Cfg::Stage;
    constexpr int MT = Cfg::MT;
    static_assert(Tokens * kOutLd<Problem::kOutRows> * sizeof(__nv_bfloat16) <=
                  Cfg::kSharedBytes);
    extern __shared__ __align__(128) std::uint8_t tall_smem[];
    Stage* stages = reinterpret_cast<Stage*>(tall_smem);

    const int tid       = static_cast<int>(threadIdx.x);
    const int warp      = tid >> 5;
    const int lane      = tid & 31;
    const int wm        = warp % Cfg::kWarpsM;
    const int wn        = warp / Cfg::kWarpsM;
    const bool pong     = warp >= 4;
    const int tile      = static_cast<int>(blockIdx.x);
    const int row_block = tile / token_tiles;
    const int token0    = tile % token_tiles * Tokens;
    const int live      = min(Tokens, tokens - token0);
    const int steps     = k / kStepK;

    // Decode item: weight row tid / 2 of the tile, code bytes 16 (tid % 2) .. + 15.
    const int my_row        = tid >> 1;
    const int my_half       = tid & 1;
    const bool q5           = problem.q5(row_block);
    const RowSource src     = problem.source(row_block, my_row, my_half, steps);
    uint4 raw               = make_uint4(0, 0, 0, 0);
    unsigned raw_high       = 0;
    std::uint16_t raw_scale = 0;
    const auto load = [&](int step) {
        asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(raw.x), "=r"(raw.y), "=r"(raw.z), "=r"(raw.w)
                     : "l"(src.codes + static_cast<std::int64_t>(step) * 32));
        if (q5) {
            asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];\n"
                         : "=r"(raw_high)
                         : "l"(src.high + static_cast<std::int64_t>(step) * 8));
        }
        raw_scale = __ldg(src.scales + step);
    };
    // Q4: q = (n ^ 8) - 8 as 2^23 + (n ^ 8) in FP32 bits minus 2^23 + 8. Q5: q = (v ^ 16) - 16
    // for v = n | h << 4, as 2^23 + (n | (h ^ 1) << 4) minus 2^23 + 16. Both are exact, as is
    // q * scale in FP32; the one rounding is to bf16, as in the staged decode.
    const auto decode_as = [&](Stage& s, auto q5_tag) {
        constexpr bool kQ5     = decltype(q5_tag)::value;
        const float scale      = __half2float(__ushort_as_half(raw_scale));
        const unsigned w[4]    = {raw.x, raw.y, raw.z, raw.w};
        const unsigned high    = ~raw_high;
        std::uint8_t* row_base = reinterpret_cast<std::uint8_t*>(s.a) + my_row * 128;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            unsigned pairs[4];
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                unsigned v0, v1;
                float bias;
                if constexpr (kQ5) {
                    v0   = ((w[j] >> (8 * b)) & 0xfu) | (((high >> (8 * j + 2 * b)) & 1u) << 4);
                    v1   = ((w[j] >> (8 * b + 4)) & 0xfu) |
                         (((high >> (8 * j + 2 * b + 1)) & 1u) << 4);
                    bias = 8388624.0f;
                } else {
                    v0   = ((w[j] ^ 0x88888888u) >> (8 * b)) & 0xfu;
                    v1   = ((w[j] ^ 0x88888888u) >> (8 * b + 4)) & 0xfu;
                    bias = 8388616.0f;
                }
                const float q0         = __int_as_float(0x4B000000 | v0) - bias;
                const float q1         = __int_as_float(0x4B000000 | v1) - bias;
                const __nv_bfloat162 p = __floats2bfloat162_rn(q0 * scale, q1 * scale);
                pairs[b]               = *reinterpret_cast<const unsigned*>(&p);
            }
            const int chunk = (my_half * 4 + j) ^ (my_row & 7);
            *reinterpret_cast<uint4*>(row_base + chunk * 16) =
                make_uint4(pairs[0], pairs[1], pairs[2], pairs[3]);
        }
    };
    const auto decode = [&](Stage& s) {
        if (q5) {
            decode_as(s, std::true_type{});
        } else {
            decode_as(s, std::false_type{});
        }
    };

    // Activation staging: 16-byte chunks; tokens past `live` read token0 with zero fill.
    constexpr int kXChunks = Tokens * 8 / kThreads;
    const __nv_bfloat16* x_src[kXChunks];
    unsigned x_dst[kXChunks];
    int x_bytes[kXChunks];
#pragma unroll
    for (int i = 0; i < kXChunks; ++i) {
        const int item  = tid + i * kThreads;
        const int token = item >> 3;
        const int chunk = item & 7;
        const bool ok   = token < live;
        x_src[i]   = x + static_cast<std::int64_t>(ok ? token0 + token : token0) * k + chunk * 8;
        x_dst[i]   = static_cast<unsigned>(offsetof(Stage, x) + token * 128 +
                                         ((chunk ^ (token & 7)) << 4));
        x_bytes[i] = ok ? 16 : 0;
    }
    const auto issue_x = [&](int step, Stage& s) {
        const unsigned base = smem_addr(&s);
#pragma unroll
        for (int i = 0; i < kXChunks; ++i) {
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                         :
                         : "r"(base + x_dst[i]), "l"(x_src[i] + step * kStepK), "r"(x_bytes[i]));
        }
    };

    // Fragment byte offsets per 16-wide K slice: A x4 per 16-row tile (rows 0-7 / 8-15 of the
    // slice's low chunk, then of its high chunk), B x4 per pair of 8-token tiles. A row or token
    // differs from the lane's line only in multiples of 8, so chunk 2 ks + c of it sits at
    // ((c ^ l7) << 4) ^ (ks << 5).
    unsigned a_off[4];
    unsigned b_off[4];
    {
        const unsigned l7 = static_cast<unsigned>(lane & 7);
        const unsigned a_row =
            static_cast<unsigned>(wm * Cfg::kWarpRows + (lane & 7) + ((lane >> 3) & 1) * 8);
        const unsigned b_tok = static_cast<unsigned>(wn * 32 + (lane >> 4) * 8 + (lane & 7));
        const unsigned ax    = (static_cast<unsigned>(lane >> 4) ^ l7) << 4;
        const unsigned bx    = (static_cast<unsigned>((lane >> 3) & 1) ^ l7) << 4;
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            a_off[ks] = static_cast<unsigned>(offsetof(Stage, a)) + a_row * 128u +
                        (ax ^ static_cast<unsigned>(ks << 5));
            b_off[ks] = static_cast<unsigned>(offsetof(Stage, x)) + b_tok * 128u +
                        (bx ^ static_cast<unsigned>(ks << 5));
        }
    }

    float acc[MT][4][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) { acc[mt][nt][e] = 0.0f; }
        }
    }
    const auto multiply = [&](const Stage& s) {
        const unsigned base = smem_addr(&s);
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            unsigned b[4][2];
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                ldmatrix_x4(b[2 * np][0], b[2 * np][1], b[2 * np + 1][0], b[2 * np + 1][1],
                            base + b_off[ks] + np * 16 * 128);
            }
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                unsigned a0, a1, a2, a3;
                ldmatrix_x4(a0, a1, a2, a3, base + a_off[ks] + mt * 16 * 128);
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    mma_bf16(acc[mt][nt][0], acc[mt][nt][1], acc[mt][nt][2], acc[mt][nt][3], a0,
                             a1, a2, a3, b[nt][0], b[nt][1]);
                }
            }
        }
    };

    load(0);
    decode(stages[0]);
    if (steps > 1) { load(1); }
    issue_x(0, stages[0]);
    cp_commit();
    for (int step = 0; step < steps; ++step) {
        cp_wait<0>();
        // Stage `step` is visible and every warp is done reading step - 1, whose buffers refill.
        __syncthreads();
        if (step + 1 < steps) { issue_x(step + 1, stages[(step + 1) & 1]); }
        cp_commit();
        const Stage& s  = stages[step & 1];
        const auto next = [&] {
            if (step + 1 < steps) {
                decode(stages[(step + 1) & 1]);
                if (step + 2 < steps) { load(step + 2); }
            }
        };
        if (pong) {
            next();
            multiply(s);
        } else {
            multiply(s);
            next();
        }
    }

    // Epilogue: the problem stages bf16 outputs as [token][row] in shared memory, then writes
    // 16-byte row chunks of each live token.
    constexpr int kOutRows = Problem::kOutRows;
    constexpr int kLd      = kOutLd<kOutRows>;
    __syncthreads();
    __nv_bfloat16* staged = reinterpret_cast<__nv_bfloat16*>(tall_smem);
    problem.template stage_outputs<MT>(acc, staged, kLd, wm * Cfg::kWarpRows, wn * 32, lane);
    __syncthreads();
    constexpr int kChunks = kOutRows / 8;
#pragma unroll
    for (int i = 0; i < Tokens * kChunks / kThreads; ++i) {
        const int item  = tid + i * kThreads;
        const int token = item / kChunks;
        const int chunk = item % kChunks;
        if (token < live) {
            problem.write(row_block, token0 + token, chunk * 8,
                          *reinterpret_cast<const uint4*>(&staged[token * kLd + chunk * 8]));
        }
    }
}

// Fragment C of an m16n8 tile: [0], [1] are row gid, tokens 2 lid and 2 lid + 1; [2], [3] are
// row gid + 8.
template <class Value>
__device__ __forceinline__ void stage_fragment(__nv_bfloat16* staged, int ld, int row, int token,
                                               Value value) {
    staged[token * ld + row]           = value(0);
    staged[(token + 1) * ld + row]     = value(1);
    staged[token * ld + row + 8]       = value(2);
    staged[(token + 1) * ld + row + 8] = value(3);
}

// Folded Q4 gate/up with SwiGLU: row block b holds output rows 64 b .. 64 b + 63; its tile rows
// are two folded 64-row blocks (32 gate rows followed by their 32 up rows), and the warp tile of
// 64 rows pairs its first and second halves.
struct SwiGluQ4Problem {
    static constexpr int kOutRows = 64;
    const std::uint8_t* codes;
    const std::uint8_t* scales;
    __nv_bfloat16* out;
    std::int32_t intermediate;

    __device__ bool q5(int) const { return false; }
    __device__ RowSource source(int row_block, int row, int half, int groups) const {
        const std::int64_t grow = static_cast<std::int64_t>(row_block) * 64 + (row >> 6) * 32 +
                                  (row & 31) + ((row >> 5) & 1) * intermediate;
        return {codes + grow * groups * 32 + half * 16, nullptr,
                reinterpret_cast<const std::uint16_t*>(scales) + grow * groups};
    }
    template <int MT>
    __device__ void stage_outputs(const float (&acc)[MT][4][4], __nv_bfloat16* staged, int ld,
                                  int warp_row, int warp_token, int lane) const {
        static_assert(MT == 4, "the folded pairing needs 64-row warp tiles");
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                stage_fragment(staged, ld, warp_row / 2 + mt * 16 + (lane >> 2),
                               warp_token + nt * 8 + 2 * (lane & 3), [&](int e) {
                                   return __float2bfloat16_rn(silu(acc[mt][nt][e]) *
                                                              acc[mt + 2][nt][e]);
                               });
            }
        }
    }
    __device__ void write(int row_block, int token, int row, uint4 values) const {
        *reinterpret_cast<uint4*>(&out[static_cast<std::int64_t>(token) * intermediate +
                                       row_block * 64 + row]) = values;
    }
};

// Grouped Q4/Q5 projections: up to four jobs, each a whole number of 128-row blocks of one
// RowSplit weight view writing bf16 rows at an offset of a token-major output.
struct GroupedJob {
    const std::uint8_t* codes; // the view's first row
    const std::uint8_t* high;  // Q5 only
    const std::uint8_t* scales;
    __nv_bfloat16* out;
    std::int32_t blocks; // 128-row blocks
    std::int32_t out_ld;
    std::int32_t out_row_offset;
    bool q5;
};

struct GroupedProblem {
    static constexpr int kOutRows = kRows;
    GroupedJob jobs[4];

    __device__ int job_of(int& row_block) const {
        int j = 0;
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            if (j == i && row_block >= jobs[i].blocks) {
                row_block -= jobs[i].blocks;
                j = i + 1;
            }
        }
        return j;
    }
    __device__ bool q5(int row_block) const {
        const int j = job_of(row_block);
        return jobs[j].q5;
    }
    __device__ RowSource source(int row_block, int row, int half, int groups) const {
        const int j             = job_of(row_block);
        const std::int64_t grow = static_cast<std::int64_t>(row_block) * kRows + row;
        return {jobs[j].codes + grow * groups * 32 + half * 16,
                jobs[j].q5 ? jobs[j].high + grow * groups * 8 + half * 4 : nullptr,
                reinterpret_cast<const std::uint16_t*>(jobs[j].scales) + grow * groups};
    }
    template <int MT>
    __device__ void stage_outputs(const float (&acc)[MT][4][4], __nv_bfloat16* staged, int ld,
                                  int warp_row, int warp_token, int lane) const {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                stage_fragment(staged, ld, warp_row + mt * 16 + (lane >> 2),
                               warp_token + nt * 8 + 2 * (lane & 3),
                               [&](int e) { return __float2bfloat16_rn(acc[mt][nt][e]); });
            }
        }
    }
    __device__ void write(int row_block, int token, int row, uint4 values) const {
        const int j = job_of(row_block);
        *reinterpret_cast<uint4*>(&jobs[j].out[static_cast<std::int64_t>(token) * jobs[j].out_ld +
                                               jobs[j].out_row_offset + row_block * kRows + row]) =
            values;
    }
};

// Q5 projection added to a bf16 residual in place: out = bf16(out + bf16(W x)).
struct ResidualQ5Problem {
    static constexpr int kOutRows = kRows;
    const std::uint8_t* codes;
    const std::uint8_t* high;
    const std::uint8_t* scales;
    __nv_bfloat16* out;
    std::int32_t rows;

    __device__ bool q5(int) const { return true; }
    __device__ RowSource source(int row_block, int row, int half, int groups) const {
        const std::int64_t grow = static_cast<std::int64_t>(row_block) * kRows + row;
        return {codes + grow * groups * 32 + half * 16, high + grow * groups * 8 + half * 4,
                reinterpret_cast<const std::uint16_t*>(scales) + grow * groups};
    }
    template <int MT>
    __device__ void stage_outputs(const float (&acc)[MT][4][4], __nv_bfloat16* staged, int ld,
                                  int warp_row, int warp_token, int lane) const {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                stage_fragment(staged, ld, warp_row + mt * 16 + (lane >> 2),
                               warp_token + nt * 8 + 2 * (lane & 3),
                               [&](int e) { return __float2bfloat16_rn(acc[mt][nt][e]); });
            }
        }
    }
    __device__ void write(int row_block, int token, int row, uint4 values) const {
        __nv_bfloat16* dst =
            &out[static_cast<std::int64_t>(token) * rows + row_block * kRows + row];
        uint4 residual            = *reinterpret_cast<const uint4*>(dst);
        const unsigned* projected = &values.x;
        unsigned* sum             = &residual.x;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const __nv_bfloat162 r = *reinterpret_cast<const __nv_bfloat162*>(&sum[i]);
            const __nv_bfloat162 p = *reinterpret_cast<const __nv_bfloat162*>(&projected[i]);
            const __nv_bfloat162 s = __floats2bfloat162_rn(__low2float(r) + __low2float(p),
                                                           __high2float(r) + __high2float(p));
            sum[i]                 = *reinterpret_cast<const unsigned*>(&s);
        }
        *reinterpret_cast<uint4*>(dst) = residual;
    }
};

// A grouped job over rows [row_begin, row_begin + rows) of a Q4/Q5 RowSplit weight whose K is
// unpadded, writing rows out_row_offset.. of a token-major bf16 output with row stride out_ld.
inline GroupedJob grouped_job(const Weight& weight, std::int32_t row_begin, std::int32_t rows,
                              __nv_bfloat16* out, std::int32_t out_ld,
                              std::int32_t out_row_offset) {
    const bool q5 = weight.qtype == QType::Q5_G64_FP16;
    if ((!q5 && weight.qtype != QType::Q4_G64_FP16) || weight.padded_shape[1] != weight.k ||
        weight.k % kStepK != 0 || row_begin < 0 || rows <= 0 || rows % kRows != 0 ||
        row_begin + rows > weight.n || out_ld % 8 != 0 || out_row_offset % 8 != 0) {
        throw std::invalid_argument("rowsplit pipelined GEMM: grouped job is unsupported");
    }
    const std::int64_t groups = weight.k / kStepK;
    return {static_cast<const std::uint8_t*>(weight.qdata) + row_begin * groups * 32,
            q5 ? static_cast<const std::uint8_t*>(weight.qhigh) + row_begin * groups * 8 : nullptr,
            static_cast<const std::uint8_t*>(weight.scales) + row_begin * groups * 2,
            out,
            rows / kRows,
            out_ld,
            out_row_offset,
            q5};
}

// Launches `row_blocks` 128-row blocks by ceil(tokens / Tokens) token tiles, token tile fastest.
// The problem's weights must have k == padded_k, a multiple of 64.
template <int Tokens, class Problem>
void launch(const Problem& problem, std::int32_t row_blocks, const __nv_bfloat16* x,
            std::int32_t k, std::int32_t tokens, cudaStream_t stream) {
    constexpr std::size_t kSmem = Config<Tokens>::kSharedBytes;
    static const bool opted_in  = [] {
        CUDA_CHECK(cudaFuncSetAttribute(rowsplit_tall_mma_kernel<Tokens, Problem>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(kSmem)));
        return true;
    }();
    (void)opted_in;
    const std::int32_t token_tiles = (tokens + Tokens - 1) / Tokens;
    const unsigned grid = static_cast<unsigned>(static_cast<std::int64_t>(row_blocks) * token_tiles);
    rowsplit_tall_mma_kernel<Tokens, Problem>
        <<<grid, kThreads, kSmem, stream>>>(x, problem, k, tokens, token_tiles);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail::rowsplit_tall
