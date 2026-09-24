#pragma once

// A8 projections of T5_G128_FP16 weights (docs/maintainer/bonsai-ternary-design.md 9.1).
//
// Weight layout. A row is K / 64 units of 13 bytes. Unit byte 4 g + j (g = 0..2, j = 0..3)
// holds the trits t_m = code of column 20 g + 4 m + j (m = 0..4), byte 12 the columns 60 + m
// (m = 0..3, t_4 = 0), as q = ceil(256 v / 243) with v = sum_m t_m 3^(4 - m). Decoding one
// 32-bit group of bytes 4 g .. 4 g + 3 runs the even and odd bytes in 16-bit lanes: per m,
// r = 3 r yields the trit r >> 8 of every byte, so each m gives one word of four codes {0,1,2}
// for the natural-order columns 20 g + 4 m .. 20 g + 4 m + 3.
//
// Activation layout: int8 in natural column order with one FP32 scale per 128 columns, and the
// sums of q per 32-column slice and per 128-column group, so `(code - 1) . q = code . q - sum(q)`.
// The quantization formula is t2_a8::quantize_group's.

#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/t2/t2_a8.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail::t5_a8 {

using t2_a8::Outputs;
using t2_a8::quantize_group;
using t2_a8::store_output;

constexpr int kUnitColumns = 64;
constexpr int kUnitBytes   = 13;

// Unrotated input: four warps, one 128-column group each; lane l holds columns 4 l .. 4 l + 3.
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
    for (int i = 0; i < 4; ++i) value[i] = __bfloat162float(source[4 * lane + i]);
    quantize_group(value, lane, token, group, k, qx, group_scale, group_sum, slice_sum);
}

// Rotated input: one CTA per (1024-column block, token) applies (1/32) H (signs * x) in FP32
// shared memory and quantizes the rotated block directly (t2_a8::rotate_quantize_kernel's
// butterfly), in natural column order.
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
    for (int i = 0; i < 4; ++i) value[i] = values[warp * 128 + 4 * lane + i] * 0x1p-5f;
    quantize_group(value, lane, token, column0 / 128 + warp, k, qx, group_scale, group_sum,
                   slice_sum);
}

// The 13 bytes of one unit as four little-endian words (bytes 0-3, 4-7, 8-11, 12), read with
// four aligned 32-bit streaming loads and funnel shifts.
__device__ __forceinline__ void load_unit(const std::uint8_t* __restrict__ row_codes, int unit,
                                          std::uint32_t (&a)[4]) {
    const std::int64_t start = std::int64_t(unit) * kUnitBytes;
    const auto* base = reinterpret_cast<const std::uint32_t*>(row_codes + (start & ~std::int64_t(3)));
    const unsigned shift   = unsigned(start & 3) * 8u;
    const std::uint32_t w0 = __ldcs(base), w1 = __ldcs(base + 1), w2 = __ldcs(base + 2),
                        w3 = __ldcs(base + 3);
    a[0] = __funnelshift_r(w0, w1, shift);
    a[1] = __funnelshift_r(w1, w2, shift);
    a[2] = __funnelshift_r(w2, w3, shift);
    a[3] = w3 >> shift;
}

// words[w] = the codes of unit columns 4 w .. 4 w + 3, one per byte.
__device__ __forceinline__ void decode_unit(const std::uint32_t (&a)[4], std::uint32_t (&words)[16]) {
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        std::uint32_t even = a[g] & 0x00ff00ffu;
        std::uint32_t odd  = (a[g] >> 8) & 0x00ff00ffu;
#pragma unroll
        for (int m = 0; m < 5; ++m) {
            even *= 3u;
            odd *= 3u;
            words[5 * g + m] = ((even >> 8) & 0x00030003u) | (odd & 0x03000300u);
            even &= 0x00ff00ffu;
            odd &= 0x00ff00ffu;
        }
    }
    std::uint32_t r    = a[3] & 0xffu;
    std::uint32_t last = 0;
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        r *= 3u;
        last |= (r >> 8) << (8 * m);
        r &= 0xffu;
    }
    words[15] = last;
}

// ---------------------------------------------------------------------------------------------
// dp4a GEMV for decode and MTP verification. A CTA of two warps owns eight rows: each half-warp
// two rows, its 16 lanes striding the units (K / 64 is a multiple of 16, so no lane idles).
// Per unit and row: four streaming loads, the arithmetic decode, and one dp4a per code word and
// token; the activation words of a unit are shared by both rows.

constexpr int kGemvThreads     = 64;
constexpr int kGemvRowsPerHalf = 2;
constexpr int kGemvRowsPerCta  = kGemvThreads / 16 * kGemvRowsPerHalf;

template <int Tile>
__global__ void __launch_bounds__(kGemvThreads)
    gemv_kernel(const uint4* __restrict__ qx, const float* __restrict__ group_scale,
                const int* __restrict__ slice_sum, const std::uint8_t* __restrict__ codes,
                const __half* __restrict__ scales, std::int64_t scale_row_halves, int n, int k,
                int tokens, Outputs outputs, bool accumulate) {
    constexpr int R  = kGemvRowsPerHalf;
    const int warp   = static_cast<int>(threadIdx.x) / 32;
    const int lane   = static_cast<int>(threadIdx.x) % 32;
    const int hl     = lane & 15;
    const int token0 = static_cast<int>(blockIdx.y) * Tile;
    const int live   = min(Tile, tokens - token0);
    const int row0 =
        ((static_cast<int>(blockIdx.x) * (kGemvThreads / 32) + warp) * 2 + (lane >> 4)) * R;
    const int units              = k / kUnitColumns;
    const std::int64_t row_bytes = std::int64_t(units) * kUnitBytes;
    const int slices = k / 32, groups = k / 128;
    // Rows past n alias the last row; their results are never stored.
    const std::uint8_t* row_codes[R];
    const __half* row_scales[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const std::int64_t row = min(row0 + r, n - 1);
        row_codes[r]           = codes + row * row_bytes;
        row_scales[r]          = scales + row * scale_row_halves;
    }
    float acc[R][Tile];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < Tile; ++t) acc[r][t] = 0.0f;
    }
    // The code words of both rows are decoded once per unit; the tokens then run in chunks of
    // at most four, so an eight-token tile holds the activation of four tokens at a time.
    constexpr int kChunk = Tile < 4 ? Tile : 4;
    for (int u = hl; u < units; u += 16) {
        std::uint32_t words[R][16];
        float scale[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            std::uint32_t a[4];
            load_unit(row_codes[r], u, a);
            decode_unit(a, words[r]);
            scale[r] = __half2float(__ldcs(row_scales[r] + (u >> 1)));
        }
#pragma unroll
        for (int c0 = 0; c0 < Tile; c0 += kChunk) {
            uint4 xs[kChunk][4];
            int offset[kChunk];
            float step[kChunk];
#pragma unroll
            for (int c = 0; c < kChunk; ++c) {
                const int t = c0 + c;
                if (t < live) {
                    const std::int64_t token = token0 + t;
                    const uint4* row         = qx + token * (k / 16) + std::int64_t(u) * 4;
#pragma unroll
                    for (int w = 0; w < 4; ++w) xs[c][w] = __ldg(row + w);
                    offset[c] = __ldg(slice_sum + token * slices + 2 * u) +
                                __ldg(slice_sum + token * slices + 2 * u + 1);
                    step[c] = __ldg(group_scale + token * groups + (u >> 1));
                } else {
#pragma unroll
                    for (int w = 0; w < 4; ++w) xs[c][w] = make_uint4(0, 0, 0, 0);
                    offset[c] = 0;
                    step[c]   = 0.0f;
                }
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                int dot[kChunk];
#pragma unroll
                for (int c = 0; c < kChunk; ++c) dot[c] = 0;
#pragma unroll
                for (int w = 0; w < 16; ++w) {
#pragma unroll
                    for (int c = 0; c < kChunk; ++c) {
                        const std::uint32_t lanes[4] = {xs[c][w >> 2].x, xs[c][w >> 2].y,
                                                        xs[c][w >> 2].z, xs[c][w >> 2].w};
                        dot[c] = __dp4a(static_cast<int>(words[r][w]), static_cast<int>(lanes[w & 3]), dot[c]);
                    }
                }
#pragma unroll
                for (int c = 0; c < kChunk; ++c) {
                    acc[r][c0 + c] = fmaf(scale[r] * step[c], static_cast<float>(dot[c] - offset[c]),
                                          acc[r][c0 + c]);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < Tile; ++t) {
            float v = acc[r][t];
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            acc[r][t] = v;
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row0 + r;
        if (row >= n) continue;
#pragma unroll
        for (int t = 0; t < Tile; ++t) {
            if (t < live && hl == t) store_output(outputs, row, token0 + t, acc[r][t], accumulate);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// int8 tensor-core GEMM for prefill, t2_a8::gemm_kernel's schedule: a CTA of four warps owns 64
// rows x 64 tokens, warp (wm, wn) 32 x 32 with m16n8k32 s8 MMAs, one 128-column scale group per
// stage. Each thread decodes one (row, unit) of the stage into 16 code words of a single shared
// buffer (the next stage's bytes are loaded before the MMAs and decoded after them); A
// fragments read the words directly (a0 = columns 4 lid .. 4 lid + 3). The activation and the
// token scales and sums are double-buffered with cp.async. The word index is XORed per row so a
// warp's A reads hit 32 distinct banks.

constexpr int kGemmWarps   = 4;
constexpr int kGemmThreads = kGemmWarps * 32;
constexpr int kGemmRows    = 64;
constexpr int kGemmTokens  = 64;
constexpr int kStageK      = 128;

__device__ __forceinline__ int gemm_word(int row, int word) { return row * 32 + (word ^ ((row & 7) << 2)); }

struct GemmStage {
    std::uint8_t x[kGemmTokens * kStageK]; // 8 KiB, 16-byte chunks swizzled per token row
    float scale[kGemmTokens];
    int sum[kGemmTokens];
};

__global__ void __launch_bounds__(kGemmThreads)
    gemm_kernel(const std::uint8_t* __restrict__ qx, const float* __restrict__ group_scale,
                const int* __restrict__ group_sum, const std::uint8_t* __restrict__ codes,
                const __half* __restrict__ scales, std::int64_t scale_row_halves, int k,
                int tokens, Outputs outputs, bool accumulate) {
    __shared__ __align__(128) GemmStage stages[2];
    __shared__ __align__(128) std::uint32_t code_words[kGemmRows * 32];
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
    const std::int64_t row_bytes = std::int64_t(k / kUnitColumns) * kUnitBytes;
    const int my_row  = tid >> 1;
    const int my_unit = tid & 1;
    const std::uint8_t* my_codes = codes + std::int64_t(row0 + my_row) * row_bytes;

    std::uint32_t raw[4];
    const auto decode_store = [&]() {
        std::uint32_t words[16];
        decode_unit(raw, words);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            *reinterpret_cast<uint4*>(&code_words[gemm_word(my_row, my_unit * 16 + 4 * q)]) =
                make_uint4(words[4 * q], words[4 * q + 1], words[4 * q + 2], words[4 * q + 3]);
        }
    };
    const auto stage_x = [&](int step, GemmStage& s) {
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
        const int token  = tid & (kGemmTokens - 1);
        const int source = token < live ? token0 + token : token0;
        const std::int64_t index = std::int64_t(source) * (k / kStageK) + step;
        if (tid < kGemmTokens) {
            cp_async_zfill<4>(&s.scale[token], group_scale + index, token < live ? 4 : 0);
        } else {
            cp_async_zfill<4>(&s.sum[token], group_sum + index, token < live ? 4 : 0);
        }
    };

    float acc[2][4][4] = {};
    load_unit(my_codes, my_unit, raw);
    decode_store();
    stage_x(0, stages[0]);
    cp_commit();
    for (int step = 0; step < steps; ++step) {
        if (step + 1 < steps) {
            stage_x(step + 1, stages[(step + 1) & 1]);
            load_unit(my_codes, 2 * (step + 1) + my_unit, raw);
        }
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
                const unsigned a0 = code_words[gemm_word(r, 8 * ks + lid)];
                const unsigned a1 = code_words[gemm_word(r + 8, 8 * ks + lid)];
                const unsigned a2 = code_words[gemm_word(r, 8 * ks + 4 + lid)];
                const unsigned a3 = code_words[gemm_word(r + 8, 8 * ks + 4 + lid)];
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
                const int token    = wn * 32 + nt * 8 + 2 * lid + j;
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
        // Every warp has read this stage's code words before they are overwritten.
        __syncthreads();
        if (step + 1 < steps) decode_store();
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
                        store_output(outputs, row, token0 + token, acc[mt][nt][2 * half + j], accumulate);
                    }
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail::t5_a8
