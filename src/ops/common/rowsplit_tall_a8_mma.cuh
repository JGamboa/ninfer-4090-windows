#pragma once

// Pipelined Q4/Q5 RowSplit x int8 activation GEMM (A8) for prefill widths (one CTA per SM).
//
// Semantics. The activation arrives quantized by rowsplit_a8_quantize (per token and 64-column
// group g: scale s = amax / 127, q = rint(x / s) in [-127, 127]), aligned with the weight's
// 64-value quant groups. A weight code c (Q4 [-8, 7], Q5 [-16, 15]) is an exact int8 MMA
// operand. For every output, each group's integer dot product d_g = sum c q is exact in int32,
// and the output is sum_g (w_scale_g * s_g) * d_g in FP32 with one FP32 product per group and
// one fused multiply-add per group in K order, before the Problem's epilogue rounding.
//
// Structure (as rowsplit_tall_mma.cuh, which also supplies the Problems): a CTA of eight warps
// owns 128 weight rows and `Tokens` (128 or 64) tokens and walks K two 64-value groups (128
// columns) per step with m16n8k32 s8 MMAs, so a barrier and the pipeline bookkeeping serve two
// groups. The decoded code tile and the activation tile are double-buffered and their row and
// token scales triple-buffered, so a step needs one barrier. Every thread decodes one group (32
// code bytes, plus 8 high-bit bytes for Q5) of one weight row per step into int8 codes, from
// registers loaded a step earlier. Each group's int32 sums start at kFloatMagic, so d_g is one
// exact FADD away from the sum's bits; a warp keeps one set of sums and multiplies and applies
// the step's two groups in K order. Warps 0-3 multiply the first group, decode, apply it, then
// multiply and apply the second; warps 4-7 first apply the previous step's second group (whose
// scales are still in their buffer) and decode, then multiply and apply the first group and
// multiply the second. Each SM sub-partition thus runs one warp's MMAs while the other warp does
// its integer and FP32 work. Tiles are ordered token-tile fastest, so the CTAs of one row block
// run together and share its code bytes in L2.
//
// A Persistent launch holds one CTA per SM, which walks the tiles tile, tile + gridDim.x, ...
// (the order in which one-tile CTAs would start). Before a tile's epilogue, the CTA loads the
// next tile's first code bytes and copies its first activation step into stage 0, and the
// epilogue stages the outputs from stage 1 on, so that latency overlaps the epilogue. The
// folded SwiGLU and residual Q5 problems launch this way; the grouped projections measured
// 2-3 % slower with it and keep one CTA per tile.

#include "ops/common/rowsplit_tall_mma.cuh"

#include <algorithm>

namespace ninfer::ops::detail::rowsplit_tall_a8 {

using rowsplit_tall::kRows;
using rowsplit_tall::kThreads;
using rowsplit_tall::RowSource;

constexpr int kGroupK     = 64; // one weight and activation scale group
constexpr int kStepGroups = 2;  // groups per pipeline step
constexpr int kStepK      = kStepGroups * kGroupK;

// 1.5 * 2^23 as FP32 bits: kFloatMagic + d reinterpreted as a float is 12582912 + d exactly for
// |d| < 2^22 (a group's |d| <= 64 * 16 * 127).
constexpr int kFloatMagic        = 0x4B400000;
constexpr float kFloatMagicValue = 12582912.0f;

template <int Tokens>
struct Config {
    static_assert(Tokens == 128 || Tokens == 64);
    static constexpr int kWarpsN   = Tokens / 32;
    static constexpr int kWarpsM   = 8 / kWarpsN;
    static constexpr int kWarpRows = kRows / kWarpsM;
    static constexpr int MT        = kWarpRows / 16;

    // Code and activation rows are 128 bytes (group 0 in chunks 0-3, group 1 in 4-7); the
    // 16-byte chunk c of line l sits at chunk c ^ (l & 7), so eight consecutive lines of one
    // chunk cover all 32 banks.
    struct Stage {
        std::uint8_t a[kRows * kStepK];
        std::uint8_t x[Tokens * kStepK];
    };
    // A step's row and token scales. They take three buffers, so warps 4-7 can still read the
    // previous step's after the barrier that lets the next step's refill its other buffers.
    struct Scales {
        float row[kStepGroups][kRows];
        float token[kStepGroups][Tokens];
    };
    // A tile's epilogue stages its BF16 outputs in the freed stages. A persistent CTA stages them
    // from stage 1 on, while its next tile fills stage 0 and scale buffer 0, and places the
    // scale buffers after both.
    template <bool Persistent>
    __host__ __device__ static constexpr std::size_t staging_offset() {
        return Persistent ? sizeof(Stage) : 0;
    }
    template <class Problem>
    __host__ __device__ static constexpr std::size_t staging_bytes() {
        return static_cast<std::size_t>(Tokens) * rowsplit_tall::kOutLd<Problem::kOutRows> *
               sizeof(__nv_bfloat16);
    }
    template <class Problem, bool Persistent>
    __host__ __device__ static constexpr std::size_t scales_offset() {
        if constexpr (!Persistent) { return 2 * sizeof(Stage); }
        constexpr std::size_t kEnd = staging_offset<true>() + staging_bytes<Problem>();
        return ((kEnd > 2 * sizeof(Stage) ? kEnd : 2 * sizeof(Stage)) + 127) / 128 * 128;
    }
    template <class Problem, bool Persistent>
    static constexpr std::size_t shared_bytes() {
        return std::max(scales_offset<Problem, Persistent>() + 3 * sizeof(Scales),
                        staging_offset<Persistent>() + staging_bytes<Problem>());
    }
};

__device__ __forceinline__ unsigned swizzled(int line, int chunk) {
    return static_cast<unsigned>(line * kStepK + ((chunk ^ (line & 7)) << 4));
}

// Four int8 lanes of 4-bit two's-complement codes n (0..15 per byte): ((n ^ 8) + 0x78) ^ 0x80
// is n - 16 for n >= 8 and n otherwise, with no carry between bytes.
__device__ __forceinline__ unsigned sign_extend_q4(unsigned bytes) {
    return ((bytes ^ 0x08080808u) + 0x78787878u) ^ 0x80808080u;
}

// Four int8 lanes of Q5 codes: low nibbles n in the bytes, high bits h = bits 0..3 of `high`
// (element order); q = n - 16 h, i.e. n | 0xF0 when h is set.
__device__ __forceinline__ unsigned with_q5_high(unsigned bytes, unsigned high) {
    return bytes | ((((high & 0xfu) * 0x00204081u) & 0x01010101u) * 0xF0u);
}

// A problem whose epilogue quantizes its BF16 outputs (see SwiGluQ4QuantizedProblem).
template <class Problem>
concept QuantizingProblem = requires { Problem::kQuantizesOutput; };

template <int Tokens, bool Persistent, class Problem>
__global__ void __launch_bounds__(kThreads, 1)
    rowsplit_tall_a8_kernel(const std::int8_t* __restrict__ qx,
                            const float* __restrict__ x_scale, Problem problem, std::int32_t k,
                            std::int32_t tokens, std::int32_t token_tiles,
                            std::int32_t total_tiles) {
    using Cfg   = Config<Tokens>;
    using Stage  = typename Cfg::Stage;
    using Scales = typename Cfg::Scales;
    constexpr int MT = Cfg::MT;
    extern __shared__ __align__(128) std::uint8_t tall_a8_smem[];
    Stage* stages  = reinterpret_cast<Stage*>(tall_a8_smem);
    constexpr std::size_t kScalesOffset = Cfg::template scales_offset<Problem, Persistent>();
    constexpr std::size_t kStagingOffset = Cfg::template staging_offset<Persistent>();
    Scales* scales = reinterpret_cast<Scales*>(tall_a8_smem + kScalesOffset);

    const int tid       = static_cast<int>(threadIdx.x);
    const int warp      = tid >> 5;
    const int lane      = tid & 31;
    const int gid       = lane >> 2;
    const int lid       = lane & 3;
    const int wm        = warp % Cfg::kWarpsM;
    const int wn        = warp / Cfg::kWarpsM;
    const bool pong     = warp >= 4;
    const int groups    = k / kGroupK;
    const int steps     = k / kStepK;

    // The CTA's current tile; set_tile() points the decode and staging state at it.
    int tile = static_cast<int>(blockIdx.x);
    int row_block = 0, token0 = 0, live = 0;
    bool q5       = false;
    RowSource src{};

    // Decode item: weight row tid / 2 of the tile, group tid % 2 of each step (its 32 code bytes
    // are the int8 chunks 4 (tid % 2) .. + 3 of the row).
    const int my_row      = tid >> 1;
    const int my_group    = tid & 1;
    uint4 raw[2]          = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
    uint2 raw_high        = make_uint2(0, 0);
    std::uint16_t raw_scale = 0;
    const auto load = [&](int step) {
        const std::int64_t group = 2 * static_cast<std::int64_t>(step) + my_group;
        const std::uint8_t* codes = src.codes + group * 32;
        asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(raw[0].x), "=r"(raw[0].y), "=r"(raw[0].z), "=r"(raw[0].w)
                     : "l"(codes));
        asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(raw[1].x), "=r"(raw[1].y), "=r"(raw[1].z), "=r"(raw[1].w)
                     : "l"(codes + 16));
        if (q5) {
            asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];\n"
                         : "=r"(raw_high.x), "=r"(raw_high.y)
                         : "l"(src.high + group * 8));
        }
        raw_scale = __ldg(src.scales + group);
    };
    // Code byte j of a half-group word holds values 2 j (low nibble) and 2 j + 1 (high nibble);
    // half h of the group (values 32 h .. 32 h + 31) becomes chunks 4 my_group + 2 h, + 1.
    const auto decode_as = [&](Stage& s, Scales& sc, auto q5_tag) {
        constexpr bool kQ5 = decltype(q5_tag)::value;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const unsigned w[4]  = {raw[h].x, raw[h].y, raw[h].z, raw[h].w};
            const unsigned high  = h == 0 ? raw_high.x : raw_high.y;
            unsigned out[8];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const unsigned lo = w[j] & 0x0f0f0f0fu;
                const unsigned hi = (w[j] >> 4) & 0x0f0f0f0fu;
                unsigned e0       = __byte_perm(lo, hi, 0x5140);
                unsigned e1       = __byte_perm(lo, hi, 0x7362);
                if constexpr (kQ5) {
                    e0 = with_q5_high(e0, high >> (8 * j));
                    e1 = with_q5_high(e1, high >> (8 * j + 4));
                } else {
                    e0 = sign_extend_q4(e0);
                    e1 = sign_extend_q4(e1);
                }
                out[2 * j]     = e0;
                out[2 * j + 1] = e1;
            }
            *reinterpret_cast<uint4*>(s.a + swizzled(my_row, 4 * my_group + 2 * h)) =
                make_uint4(out[0], out[1], out[2], out[3]);
            *reinterpret_cast<uint4*>(s.a + swizzled(my_row, 4 * my_group + 2 * h + 1)) =
                make_uint4(out[4], out[5], out[6], out[7]);
        }
        sc.row[my_group][my_row] = __half2float(__ushort_as_half(raw_scale));
    };
    const auto decode = [&](Stage& s, Scales& sc) {
        if (q5) {
            decode_as(s, sc, std::true_type{});
        } else {
            decode_as(s, sc, std::false_type{});
        }
    };

    // Activation staging: chunk tid % 8 of tokens tid / 8 + 32 i, and one token scale per
    // thread < 2 Tokens (group tid / Tokens of the step); tokens past `live` read token0 with
    // zero fill. Lines 32 apart keep the swizzle, so one destination offset serves every i.
    constexpr int kXChunks  = Tokens * (kStepK / 16) / kThreads;
    constexpr int kXStride  = kThreads / (kStepK / 16);
    // Source offsets are unsigned 32-bit (the launcher checks tokens * k < 2^32), which keeps the
    // per-thread staging state to five registers.
    const int x_token = tid >> 3;
    std::uint32_t x_off[kXChunks];
    const int scale_token = tid % Tokens;
    std::uint32_t scale_off = 0;
    const auto set_tile = [&](int t) {
        row_block = t / token_tiles;
        token0    = t % token_tiles * Tokens;
        live      = min(Tokens, tokens - token0);
        q5        = problem.q5(row_block);
        src       = problem.source(row_block, my_row, 0, groups);
#pragma unroll
        for (int i = 0; i < kXChunks; ++i) {
            const int token = x_token + i * kXStride;
            x_off[i]        = static_cast<std::uint32_t>(
                static_cast<std::int64_t>(token < live ? token0 + token : token0) * k +
                (tid & 7) * 16);
        }
        scale_off = static_cast<std::uint32_t>(
            tid / Tokens * tokens + (scale_token < live ? token0 + scale_token : token0));
    };
    const auto issue_x = [&](int step, Stage& s, Scales& sc) {
        const unsigned base = smem_addr(&s) + static_cast<unsigned>(offsetof(Stage, x)) +
                              swizzled(x_token, tid & 7);
#pragma unroll
        for (int i = 0; i < kXChunks; ++i) {
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                         :
                         : "r"(base + i * kXStride * kStepK),
                           "l"(qx + (x_off[i] + static_cast<std::uint32_t>(step * kStepK))),
                           "r"(x_token + i * kXStride < live ? 16 : 0));
        }
        if (tid < kStepGroups * Tokens) {
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                         :
                         : "r"(smem_addr(&sc) + static_cast<unsigned>(offsetof(Scales, token)) +
                               static_cast<unsigned>(tid) * 4u),
                           "l"(x_scale + (scale_off + static_cast<std::uint32_t>(
                                                          kStepGroups * step * tokens))),
                           "r"(scale_token < live ? 4 : 0));
        }
    };

    // Fragment byte offsets of the 32-wide K slice ks = 0: A x4 per 16-row tile (rows 0-7 / 8-15
    // of chunk 0, then of chunk 1), B x4 per pair of 8-token tiles (tokens of the first tile at
    // chunks 0 and 1, then of the second). Slice ks holds chunks 2 ks and 2 ks + 1, which sit at
    // the ks = 0 offset XOR (ks << 5); tiles differ from the lane's line in multiples of 8, which
    // keep the swizzle.
    const unsigned a_off = static_cast<unsigned>(offsetof(Stage, a)) +
                           swizzled(wm * Cfg::kWarpRows + (lane & 7) + ((lane >> 3) & 1) * 8,
                                    lane >> 4);
    const unsigned b_off = static_cast<unsigned>(offsetof(Stage, x)) +
                           swizzled(wn * 32 + (lane >> 4) * 8 + (lane & 7), (lane >> 3) & 1);

    // The int32 sums of group `group` (0 or 1) of the step.
    const auto multiply = [&](const Stage& s, int group, int (&g)[MT][4][4]) {
        const unsigned base = smem_addr(&s);
#pragma unroll
        for (int kh = 0; kh < 2; ++kh) {
            const unsigned slice = static_cast<unsigned>(2 * group + kh) << 5;
            unsigned b[4][2];
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                ldmatrix_x4(b[2 * np][0], b[2 * np][1], b[2 * np + 1][0], b[2 * np + 1][1],
                            base + (b_off ^ slice) + np * 16 * kStepK);
            }
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                unsigned a0, a1, a2, a3;
                ldmatrix_x4(a0, a1, a2, a3, base + (a_off ^ slice) + mt * 16 * kStepK);
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    int* d = g[mt][nt];
                    if (kh == 0) {
                        mma_s8_from(d[0], d[1], d[2], d[3], a0, a1, a2, a3, b[nt][0], b[nt][1],
                                    kFloatMagic, kFloatMagic, kFloatMagic, kFloatMagic);
                    } else {
                        mma_s8(d[0], d[1], d[2], d[3], a0, a1, a2, a3, b[nt][0], b[nt][1]);
                    }
                }
            }
        }
    };
    float acc[MT][4][4];
    // Fragment C: [0], [1] are row gid, tokens 2 lid and 2 lid + 1; [2], [3] are row gid + 8.
    // Applies the int32 sums of group `group` of the step whose scales are `sc`.
    const auto update = [&](const int (&g)[MT][4][4], const Scales& sc, int group) {
        const float* row_scale   = sc.row[group] + wm * Cfg::kWarpRows + gid;
        const float* token_scale = sc.token[group] + wn * 32 + 2 * lid;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float unit =
                        row_scale[mt * 16 + (e >> 1) * 8] * token_scale[nt * 8 + (e & 1)];
                    acc[mt][nt][e]   = fmaf(unit, __int_as_float(g[mt][nt][e]) - kFloatMagicValue,
                                            acc[mt][nt][e]);
                }
            }
        }
    };

    // Tiles tile, tile + gridDim.x, ... of the CTA. A tile's first step is prepared while the
    // previous tile's epilogue runs: its code bytes are loaded and its activation is copied into
    // stage 0 and scale buffer 0 before the epilogue stages its outputs from stage 1 on.
    set_tile(tile);
    load(0);
    decode(stages[0], scales[0]);
    if (steps > 1) { load(1); }
    issue_x(0, stages[0], scales[0]);
    cp_commit();
    int g[MT][4][4];
    for (;;) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
                for (int e = 0; e < 4; ++e) { acc[mt][nt][e] = 0.0f; }
            }
        }
        int ring = 0; // step % 3, the step's scale buffer
        for (int step = 0; step < steps; ++step) {
            const int ring_next = ring == 2 ? 0 : ring + 1;
            const int ring_prev = ring == 0 ? 2 : ring - 1;
            cp_wait<0>();
            // Stage `step` is visible and every warp is done reading step - 1's tiles and step - 2's
            // scales, whose buffers refill.
            __syncthreads();
            if (step + 1 < steps) { issue_x(step + 1, stages[(step + 1) & 1], scales[ring_next]); }
            cp_commit();
            const Stage& s   = stages[step & 1];
            const Scales& sc = scales[ring];
            const auto next  = [&] {
                if (step + 1 < steps) {
                    decode(stages[(step + 1) & 1], scales[ring_next]);
                    if (step + 2 < steps) { load(step + 2); }
                }
            };
            if (pong) {
                if (step > 0) { update(g, scales[ring_prev], 1); }
                next();
                multiply(s, 0, g);
                update(g, sc, 0);
                multiply(s, 1, g);
            } else {
                multiply(s, 0, g);
                next();
                update(g, sc, 0);
                multiply(s, 1, g);
                update(g, sc, 1);
            }
            ring = ring_next;
        }
        if (pong) { update(g, scales[ring == 0 ? 2 : ring - 1], 1); }

        // Every warp is done with the stages and scale buffers.
        __syncthreads();
        const int out_row_block = row_block;
        const int out_token0    = token0;
        const int out_live      = live;
        tile += static_cast<int>(gridDim.x);
        const bool more = Persistent && tile < total_tiles;
        if constexpr (Persistent) {
            if (more) {
                set_tile(tile);
                load(0);
                issue_x(0, stages[0], scales[0]);
            }
            cp_commit();
        }

        // Epilogue: the problem stages bf16 outputs as [token][row] in shared memory, then writes
        // 16-byte row chunks of each live token (a quantizing problem is called for every token, so
        // the lanes of a token can reduce across the chunks; it stores only the live ones).
        constexpr int kOutRows = Problem::kOutRows;
        constexpr int kLd      = rowsplit_tall::kOutLd<kOutRows>;
        __nv_bfloat16* staged = reinterpret_cast<__nv_bfloat16*>(tall_a8_smem + kStagingOffset);
        problem.template stage_outputs<MT>(acc, staged, kLd, wm * Cfg::kWarpRows, wn * 32, lane);
        __syncthreads();
        constexpr int kChunks = kOutRows / 8;
#pragma unroll
        for (int i = 0; i < Tokens * kChunks / kThreads; ++i) {
            const int item   = tid + i * kThreads;
            const int token  = item / kChunks;
            const int chunk  = item % kChunks;
            const uint4 data = *reinterpret_cast<const uint4*>(&staged[token * kLd + chunk * 8]);
            if constexpr (QuantizingProblem<Problem>) {
                problem.write_quantized(out_row_block, out_token0 + token, chunk * 8, data,
                                        token < out_live);
            } else if (token < out_live) {
                problem.write(out_row_block, out_token0 + token, chunk * 8, data);
            }
        }
        if (!more) { break; }
        // Stage 1, where the outputs were staged, refills only after the next tile's first barrier.
        decode(stages[0], scales[0]);
        if (steps > 1) { load(1); }
    }
}

// Launches `row_blocks` 128-row blocks by ceil(tokens / Tokens) token tiles, token tile fastest,
// over an activation quantized by rowsplit_a8_quantize for this `k` and `tokens`. The problem's
// weights must have k == padded_k, a multiple of 128 (every Qwen3.8 projection input). With
// Persistent, one CTA per SM walks the tiles; otherwise each CTA runs one tile.
template <int Tokens, bool Persistent, class Problem>
void launch(const Problem& problem, std::int32_t row_blocks, const std::int8_t* qx,
            const float* x_scale, std::int32_t k, std::int32_t tokens, cudaStream_t stream) {
    if (k <= 0 || k % kStepK != 0 || tokens <= 0 ||
        static_cast<std::int64_t>(tokens) * k > static_cast<std::int64_t>(UINT32_MAX)) {
        throw std::invalid_argument("rowsplit A8 pipelined GEMM: unsupported k or width");
    }
    constexpr std::size_t kSmem = Config<Tokens>::template shared_bytes<Problem, Persistent>();
    static const bool opted_in  = [] {
        CUDA_CHECK(cudaFuncSetAttribute(rowsplit_tall_a8_kernel<Tokens, Persistent, Problem>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(kSmem)));
        return true;
    }();
    (void)opted_in;
    const std::int32_t token_tiles = (tokens + Tokens - 1) / Tokens;
    const std::int64_t total_tiles = static_cast<std::int64_t>(row_blocks) * token_tiles;
    if (total_tiles > INT32_MAX) {
        throw std::invalid_argument("rowsplit A8 pipelined GEMM: too many tiles");
    }
    const unsigned grid = static_cast<unsigned>(
        Persistent ? std::min<std::int64_t>(total_tiles, device_sm_count()) : total_tiles);
    rowsplit_tall_a8_kernel<Tokens, Persistent, Problem><<<grid, kThreads, kSmem, stream>>>(
        qx, x_scale, problem, k, tokens, token_tiles, static_cast<std::int32_t>(total_tiles));
    CUDA_CHECK(cudaGetLastError());
}

// The folded Q4 gate/up SwiGLU whose output is the A8 activation of the next projection. Row
// block b produces SwiGLU rows 64 b .. 64 b + 63, i.e. 64-column group b of that activation: the
// BF16 values SwiGluQ4Problem would store are quantized per token exactly as
// rowsplit_a8_quantize does it (amax over the group, scale = amax / 127, q = rint(x * (127 /
// amax)) clamped, all in FP32) and written as int8 [intermediate, T] with FP32 scales [T, groups].
struct SwiGluQ4QuantizedProblem {
    static constexpr int kOutRows         = 64;
    static constexpr bool kQuantizesOutput = true;
    rowsplit_tall::SwiGluQ4Problem folded; // its bf16 output is unused
    std::int8_t* q;
    float* scale;
    std::int32_t tokens;

    __device__ bool q5(int row_block) const { return folded.q5(row_block); }
    __device__ RowSource source(int row_block, int row, int half, int groups) const {
        return folded.source(row_block, row, half, groups);
    }
    template <int MT>
    __device__ void stage_outputs(const float (&acc)[MT][4][4], __nv_bfloat16* staged, int ld,
                                  int warp_row, int warp_token, int lane) const {
        folded.template stage_outputs<MT>(acc, staged, ld, warp_row, warp_token, lane);
    }
    // Rows 0, 8, .., 56 of one token's group sit in eight consecutive lanes; every lane of the
    // warp calls this, and only live tokens are stored.
    __device__ void write_quantized(int row_block, int token, int row, uint4 values,
                                    bool live) const {
        const unsigned words[4] = {values.x, values.y, values.z, values.w};
        float value[8];
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 pair =
                __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&words[i]));
            value[2 * i]     = pair.x;
            value[2 * i + 1] = pair.y;
            amax             = fmaxf(amax, fmaxf(fabsf(pair.x), fabsf(pair.y)));
        }
#pragma unroll
        for (int offset_lanes = 1; offset_lanes < 8; offset_lanes <<= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset_lanes));
        }
        const float inverse = amax > 0.0f ? 127.0f / amax : 0.0f;
        unsigned packed[2]  = {0u, 0u};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int code = max(-127, min(127, __float2int_rn(value[i] * inverse)));
            packed[i >> 2] |= (static_cast<unsigned>(code) & 0xffu) << (8 * (i & 3));
        }
        if (!live) { return; }
        *reinterpret_cast<uint2*>(&q[static_cast<std::int64_t>(token) * folded.intermediate +
                                     row_block * 64 + row]) = make_uint2(packed[0], packed[1]);
        if (row == 0) { scale[static_cast<std::int64_t>(row_block) * tokens + token] = amax / 127.0f; }
    }
};

} // namespace ninfer::ops::detail::rowsplit_tall_a8
