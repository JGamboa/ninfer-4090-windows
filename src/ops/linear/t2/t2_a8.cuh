#pragma once

// A8 t2 projections: the Hadamard-rotated BF16 activation is quantized to symmetric int8 with
// one FP32 scale per 128-column group (the weight scale group), and the ternary codes multiply
// it in integer arithmetic.
//
// With a rotated weight (Weight::input_signs) the Prism rotation is fused into the quantization.
//
// Quantized activation layout. Inside every 16-column block the 4x4 byte matrix is transposed:
// int8 position 16 s + 4 j + m holds column 16 s + 4 m + j. Then `(code_word >> 2 j) & 0x03030303`
// of the 32-bit code word of those 16 columns (column c at bits 2 c) yields the codes {0,1,2}
// of exactly the four columns stored in activation word 4 s + j, byte for byte. Both the dp4a
// GEMV and the int8 MMA consume the unsigned codes directly; `(code - 1) . q = code . q - sum(q)`
// removes the offset with the stored per-slice and per-group sums of q.

#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail::t2_a8 {

constexpr int kMaxOutputs      = 4;
constexpr unsigned kCodeMask   = 0x03030303u;

struct Outputs {
    __nv_bfloat16* data[kMaxOutputs];
    int end[kMaxOutputs]; // exclusive parent row bound of each output
};

__device__ __forceinline__ void store_output(const Outputs& outputs, int row, int token,
                                             float value, bool accumulate) {
    int s = 0;
    while (row >= outputs.end[s]) ++s;
    const int begin = s ? outputs.end[s - 1] : 0;
    const int rows  = outputs.end[s] - begin;
    auto* out       = outputs.data[s] + std::int64_t(token) * rows + (row - begin);
    *out = __float2bfloat16_rn(accumulate ? value + __bfloat162float(*out) : value);
}

// One warp quantizes one (token, 128-column group). Lane l = 4 s + m holds the group columns
// 16 s + 4 i + m (i = 0..3) in `value` and writes activation word l.
// q = rint(x * 127 / amax), scale = amax / 127 (zero for an all-zero group).
__device__ __forceinline__ void quantize_group(const float (&value)[4], int lane, int token,
                                               int group, int k, std::uint32_t* __restrict__ qx,
                                               float* __restrict__ group_scale,
                                               int* __restrict__ group_sum,
                                               int* __restrict__ slice_sum) {
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) amax = fmaxf(amax, fabsf(value[i]));
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
    }
    const float inverse = amax > 0.0f ? 127.0f / amax : 0.0f;
    std::uint32_t word  = 0;
    int sum             = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int q = max(-127, min(127, __float2int_rn(value[i] * inverse)));
        word |= (static_cast<std::uint32_t>(q) & 0xffu) << (8 * i);
        sum += q;
    }
    const std::int64_t groups = k / 128;
    qx[(std::int64_t(token) * k + group * 128) / 4 + lane] = word;
    // A 32-column slice is eight consecutive lanes.
#pragma unroll
    for (int offset = 1; offset < 8; offset <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    if ((lane & 7) == 0) slice_sum[std::int64_t(token) * (k / 32) + group * 4 + (lane >> 3)] = sum;
    sum += __shfl_xor_sync(0xffffffffu, sum, 8);
    sum += __shfl_xor_sync(0xffffffffu, sum, 16);
    if (lane == 0) {
        group_scale[std::int64_t(token) * groups + group] = amax / 127.0f;
        group_sum[std::int64_t(token) * groups + group]   = sum;
    }
}

// Unrotated input: four warps, one group each.
__global__ void __launch_bounds__(128)
    quantize_kernel(const __nv_bfloat16* __restrict__ x, int k, std::uint32_t* __restrict__ qx,
                    float* __restrict__ group_scale, int* __restrict__ group_sum,
                    int* __restrict__ slice_sum) {
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int group = static_cast<int>(blockIdx.x) * 4 + warp;
    const int token = static_cast<int>(blockIdx.y);
    const __nv_bfloat16* source = x + std::int64_t(token) * k + group * 128;
    float value[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        value[i] = __bfloat162float(source[16 * (lane >> 2) + 4 * i + (lane & 3)]);
    }
    quantize_group(value, lane, token, group, k, qx, group_scale, group_sum, slice_sum);
}

// Rotated input: one CTA per (1024-column block, token) applies the Prism rotation
// (1/32) H (signs * x) in FP32 shared memory (the butterfly of ops/hadamard) and quantizes the
// rotated block directly, eight warps with one 128-column group each. The rotated activation
// is never rounded to BF16 or written back.
constexpr int kRotateThreads = 256;

__global__ void __launch_bounds__(kRotateThreads)
    rotate_quantize_kernel(const __nv_bfloat16* __restrict__ x,
                           const __nv_bfloat16* __restrict__ signs, int k,
                           std::uint32_t* __restrict__ qx, float* __restrict__ group_scale,
                           int* __restrict__ group_sum, int* __restrict__ slice_sum) {
    constexpr int kBlock = 1024;
    __shared__ float values[kBlock];
    const int tid     = static_cast<int>(threadIdx.x);
    const int token   = static_cast<int>(blockIdx.y);
    const int column0 = static_cast<int>(blockIdx.x) * kBlock;
    const __nv_bfloat16* source = x + std::int64_t(token) * k + column0;
#pragma unroll
    for (int i = 0; i < kBlock / kRotateThreads; ++i) {
        const int index = tid + i * kRotateThreads;
        values[index]   = __bfloat162float(source[index]) * __bfloat162float(signs[column0 + index]);
    }
    __syncthreads();
#pragma unroll
    for (int stride = 1; stride < kBlock; stride <<= 1) {
#pragma unroll
        for (int pair = tid; pair < kBlock / 2; pair += kRotateThreads) {
            const int low        = (pair / stride) * 2 * stride + pair % stride;
            const float a        = values[low];
            const float b        = values[low + stride];
            values[low]          = a + b;
            values[low + stride] = a - b;
        }
        __syncthreads();
    }
    const int warp = tid >> 5;
    const int lane = tid & 31;
    float value[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        value[i] = values[warp * 128 + 16 * (lane >> 2) + 4 * i + (lane & 3)] * 0x1p-5f;
    }
    quantize_group(value, lane, token, column0 / 128 + warp, k, qx, group_scale, group_sum,
                   slice_sum);
}

// ---------------------------------------------------------------------------------------------
// dp4a GEMV for decode and MTP verification. Same schedule as the A16 GEMV: a warp owns two
// rows, a lane one 32-code slice per step (1024 columns per warp step), weights streamed once.
// Per slice and row: eight shift/mask code extractions shared by all tokens, then one dp4a per
// four weights and token.

constexpr int kGemvThreads     = 256;
constexpr int kGemvWarps       = kGemvThreads / 32;
constexpr int kGemvRowsPerWarp = 2;

template <int Tile>
__global__ void __launch_bounds__(kGemvThreads)
    gemv_kernel(const uint4* __restrict__ qx, const float* __restrict__ group_scale,
                const int* __restrict__ slice_sum, const uint2* __restrict__ codes,
                const __half* __restrict__ scales, std::int64_t scale_row_halves, int n, int k,
                int tokens, Outputs outputs, bool accumulate) {
    constexpr int R  = kGemvRowsPerWarp;
    const int warp   = static_cast<int>(threadIdx.x) / 32;
    const int lane   = static_cast<int>(threadIdx.x) % 32;
    const int token0 = static_cast<int>(blockIdx.y) * Tile;
    const int live   = min(Tile, tokens - token0);
    const int row0   = (static_cast<int>(blockIdx.x) * kGemvWarps + warp) * R;
    if (row0 >= n) return;
    const bool pair           = row0 + 1 < n;
    const int slices_per_row  = k / 32;
    const int groups_per_row  = k / 128;
    const uint2* row_codes[2] = {codes + std::int64_t(row0) * slices_per_row,
                                 codes + std::int64_t(pair ? row0 + 1 : row0) * slices_per_row};
    const __half* row_scales[2] = {scales + std::int64_t(row0) * scale_row_halves,
                                   scales + std::int64_t(pair ? row0 + 1 : row0) * scale_row_halves};

    float acc[R][Tile];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < Tile; ++t) acc[r][t] = 0.0f;
    }

    uint2 next_bits[R];
    __half next_scale[R];
    const auto fetch = [&](int slice) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            next_bits[r]  = __ldcs(row_codes[r] + slice);
            next_scale[r] = __ldcs(row_scales[r] + slice / 4);
        }
    };
    if (lane < slices_per_row) fetch(lane);
    for (int slice = lane; slice < slices_per_row; slice += 32) {
        uint2 bits[R];
        float scale[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            bits[r]  = next_bits[r];
            scale[r] = __half2float(next_scale[r]);
        }
        if (slice + 32 < slices_per_row) fetch(slice + 32);

        uint4 xs[Tile][2];
        int offset[Tile];
        float step[Tile];
#pragma unroll
        for (int t = 0; t < Tile; ++t) {
            if (t < live) {
                const std::int64_t token = token0 + t;
                const uint4* row         = qx + token * (k / 16) + std::int64_t(slice) * 2;
                xs[t][0]                 = __ldg(row);
                xs[t][1]                 = __ldg(row + 1);
                offset[t] = __ldg(slice_sum + token * slices_per_row + slice);
                step[t]   = __ldg(group_scale + token * groups_per_row + slice / 4);
            } else {
                xs[t][0] = xs[t][1] = make_uint4(0, 0, 0, 0);
                offset[t]           = 0;
                step[t]             = 0.0f;
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            int dot[Tile];
#pragma unroll
            for (int t = 0; t < Tile; ++t) dot[t] = 0;
#pragma unroll
            for (int w = 0; w < 2; ++w) {
                const std::uint32_t word = w ? bits[r].y : bits[r].x;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int c = static_cast<int>((word >> (2 * j)) & kCodeMask);
#pragma unroll
                    for (int t = 0; t < Tile; ++t) {
                        const std::uint32_t lanes[4] = {xs[t][w].x, xs[t][w].y, xs[t][w].z,
                                                        xs[t][w].w};
                        dot[t] = __dp4a(c, static_cast<int>(lanes[j]), dot[t]);
                    }
                }
            }
#pragma unroll
            for (int t = 0; t < Tile; ++t) {
                acc[r][t] = fmaf(scale[r] * step[t], static_cast<float>(dot[t] - offset[t]),
                                 acc[r][t]);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < Tile; ++t) {
            float v = acc[r][t];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            acc[r][t] = v;
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row0 + r;
        if (row >= n) continue;
#pragma unroll
        for (int t = 0; t < Tile; ++t) {
            if (t < live && lane == t) store_output(outputs, row, token0 + t, acc[r][t], accumulate);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// int8 tensor-core GEMM for prefill. A CTA of four warps owns 64 rows x 64 tokens; warp
// (wm, wn) computes 32 x 32 with 2 x 4 m16n8k32 s8 MMAs per 32-column step. One K stage is one
// 128-column scale group: codes (64 x 32 bytes), int8 activations (64 x 128 bytes), and the
// group scale and sum of every token are double-buffered with cp.async. An A fragment register
// is one shift and mask of a code word (the activation layout matches it), reused by four
// token tiles; every group's integer sum is corrected and scaled once.

constexpr int kGemmWarps   = 4;
constexpr int kGemmThreads = kGemmWarps * 32;
constexpr int kGemmRows    = 64;
constexpr int kGemmTokens  = 64;
constexpr int kStageK      = 128;
constexpr int kStageBytes  = kStageK / 4;

struct GemmStage {
    std::uint8_t codes[kGemmRows][kStageBytes]; // 2 KiB
    std::uint8_t x[kGemmTokens * kStageK];      // 8 KiB, 16-byte chunks swizzled per token row
    float scale[kGemmTokens];
    int sum[kGemmTokens];
};

__global__ void __launch_bounds__(kGemmThreads)
    gemm_kernel(const std::uint8_t* __restrict__ qx, const float* __restrict__ group_scale,
                const int* __restrict__ group_sum, const std::uint8_t* __restrict__ codes,
                const __half* __restrict__ scales, std::int64_t scale_row_halves, int k,
                int tokens, Outputs outputs, bool accumulate) {
    __shared__ __align__(128) GemmStage stages[2];

    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int gid    = lane >> 2;
    const int lid    = lane & 3;
    const int wm     = warp & 1;
    const int wn     = warp >> 1;
    const int row0   = static_cast<int>(blockIdx.x) * kGemmRows;
    const int token0 = static_cast<int>(blockIdx.y) * kGemmTokens;
    const int live   = min(kGemmTokens, tokens - token0);
    const int steps  = k / kStageK;
    const std::int64_t code_row_bytes = k / 4;

    const auto stage = [&](int step, GemmStage& s) {
        {
            const int row  = tid >> 1;
            const int half = tid & 1;
            cp_async<16, Cache::cg>(&s.codes[row][half * 16],
                                    codes + std::int64_t(row0 + row) * code_row_bytes +
                                        std::int64_t(step) * kStageBytes + half * 16);
        }
#pragma unroll
        for (int i = 0; i < kGemmTokens * (kStageK / 16) / kGemmThreads; ++i) {
            const int item   = tid + i * kGemmThreads;
            const int token  = item >> 3;
            const int chunk  = item & 7;
            const int source = token < live ? token0 + token : token0;
            cp_async_zfill<16>(&s.x[token * kStageK + ((chunk ^ (token & 7)) << 4)],
                               qx + std::int64_t(source) * k + std::int64_t(step) * kStageK +
                                   chunk * 16,
                               token < live ? 16 : 0);
        }
        {
            const int token  = tid & (kGemmTokens - 1);
            const int source = token < live ? token0 + token : token0;
            const std::int64_t index = std::int64_t(source) * (k / kStageK) + step;
            if (tid < kGemmTokens) {
                cp_async_zfill<4>(&s.scale[token], group_scale + index, token < live ? 4 : 0);
            } else {
                cp_async_zfill<4>(&s.sum[token], group_sum + index, token < live ? 4 : 0);
            }
        }
    };

    float acc[2][4][4] = {};
    stage(0, stages[0]);
    cp_commit();

    for (int step = 0; step < steps; ++step) {
        if (step + 1 < steps) { stage(step + 1, stages[(step + 1) & 1]); }
        cp_commit();
        cp_wait<1>();
        __syncthreads();

        const GemmStage& s = stages[step & 1];
        int group[2][4][4] = {};
#pragma unroll
        for (int ks = 0; ks < kStageK / 32; ++ks) {
            unsigned b[4][2];
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int b_row = wn * 32 + nt * 8 + (lane & 7);
                const int chunk = ks * 2 + ((lane >> 3) & 1);
                ldmatrix_x2(b[nt][0], b[nt][1],
                            smem_addr(&s.x[b_row * kStageK + ((chunk ^ (b_row & 7)) << 4)]));
            }
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const int r       = wm * 32 + mt * 16 + gid;
                const uint2 top   = *reinterpret_cast<const uint2*>(&s.codes[r][ks * 8]);
                const uint2 below = *reinterpret_cast<const uint2*>(&s.codes[r + 8][ks * 8]);
                const unsigned a0 = (top.x >> (2 * lid)) & kCodeMask;
                const unsigned a1 = (below.x >> (2 * lid)) & kCodeMask;
                const unsigned a2 = (top.y >> (2 * lid)) & kCodeMask;
                const unsigned a3 = (below.y >> (2 * lid)) & kCodeMask;
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    mma_s8(group[mt][nt][0], group[mt][nt][1], group[mt][nt][2],
                           group[mt][nt][3], a0, a1, a2, a3, b[nt][0], b[nt][1]);
                }
            }
        }
        float token_scale[4][2];
        int token_sum[4][2];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int token   = wn * 32 + nt * 8 + 2 * lid + j;
                token_scale[nt][j] = s.scale[token];
                token_sum[nt][j]   = s.sum[token];
            }
        }
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            const int r        = row0 + wm * 32 + mt * 16 + gid;
            const float top    = __half2float(__ldg(scales + std::int64_t(r) * scale_row_halves + step));
            const float bottom =
                __half2float(__ldg(scales + std::int64_t(r + 8) * scale_row_halves + step));
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int j      = e & 1;
                    const float unit = (e < 2 ? top : bottom) * token_scale[nt][j];
                    acc[mt][nt][e] =
                        fmaf(unit, static_cast<float>(group[mt][nt][e] - token_sum[nt][j]),
                             acc[mt][nt][e]);
                }
            }
        }
        __syncthreads();
    }

    // Fragment C: [0], [1] are row gid, tokens 2*lid and 2*lid+1; [2], [3] are row gid+8.
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int row = row0 + wm * 32 + mt * 16 + gid + half * 8;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int token = wn * 32 + nt * 8 + 2 * lid + j;
                    if (token < live) {
                        store_output(outputs, row, token0 + token, acc[mt][nt][half * 2 + j],
                                     accumulate);
                    }
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail::t2_a8
