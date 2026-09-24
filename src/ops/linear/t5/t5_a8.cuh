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
// With a rotated weight (Weight::input_signs) the Prism rotation (1/32) H (signs * x) per
// 1024-column block is fused into the quantization.

#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail::t5_a8 {

constexpr int kMaxOutputs = 4;

struct Outputs {
    __nv_bfloat16* data[kMaxOutputs];
    int end[kMaxOutputs]; // exclusive parent row bound of each output
};

// The output of parent row `row`, resolved with unrolled selects (no local copy of Outputs).
struct OutputRow {
    __nv_bfloat16* data; // element (row, token 0)
    int rows;            // token stride
};

__device__ __forceinline__ OutputRow output_row(const Outputs& outputs, int row) {
    __nv_bfloat16* data = outputs.data[0];
    int begin = 0, end = outputs.end[0];
#pragma unroll
    for (int s = 1; s < kMaxOutputs; ++s) {
        if (row >= outputs.end[s - 1]) {
            data  = outputs.data[s];
            begin = outputs.end[s - 1];
            end   = outputs.end[s];
        }
    }
    return {data + (row - begin), end - begin};
}

__device__ __forceinline__ void store_row(OutputRow out, int token, float value, bool accumulate) {
    __nv_bfloat16* p = out.data + std::int64_t(token) * out.rows;
    *p = __float2bfloat16_rn(accumulate ? value + __bfloat162float(*p) : value);
}

// One warp quantizes one (token, 128-column group); lane l holds columns 4 l .. 4 l + 3 in
// `value` and writes activation word l. q = rint(x * 127 / amax), scale = amax / 127 (zero for
// an all-zero group), then the sums of q per 32-column slice (eight lanes) and per group.
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
// shared memory (the butterfly of ops/hadamard) and quantizes the rotated block directly; the
// rotated activation is never rounded to BF16.
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
            if (t < live && hl == t) store_row(output_row(outputs, row), token0 + t, acc[r][t], accumulate);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// int8 tensor-core GEMM for prefill: a CTA of 2 x WarpsN warps owns 64 rows x 32 WarpsN tokens,
// warp (wm, wn) 32 x 32 with m16n8k32 s8 MMAs, one 128-column scale group per stage. The stage's
// 64 rows x 2 units are decoded into 16 code words each of a single shared buffer, one unit per
// thread (WarpsN = 2) or split between the two warp halves of the CTA (WarpsN = 4: bytes 0-7
// in warps 0-3, bytes 8-12 in warps 4-7, so the decode never diverges within a warp); the next
// stage's bytes are loaded before the MMAs and decoded after them. Code words and activations
// are stored in 16-byte chunks XORed with the row (token) index, so both the A and B fragments
// load with conflict-free `ldmatrix`. The activation and the token scales and sums are
// double-buffered with cp.async.

constexpr int kGemmRows  = 64;
constexpr int kStageK    = 128;

template <int WarpsN>
struct GemmConfig {
    static_assert(WarpsN == 2 || WarpsN == 4);
    static constexpr int kTokens  = 32 * WarpsN;
    static constexpr int kThreads = 64 * WarpsN;
    static constexpr int kParts   = WarpsN / 2; // decoding threads per (row, unit)
    // 64-token CTAs keep three per SM (shared memory); 128-token CTAs two, at <= 128 registers.
    static constexpr int kMinBlocks = WarpsN == 2 ? 3 : 2;
};

__device__ __forceinline__ int gemm_word(int row, int word) { return row * 32 + (word ^ ((row & 7) << 2)); }

template <int Tokens>
struct GemmStage {
    std::uint8_t x[Tokens * kStageK]; // 16-byte chunks swizzled per token row
    float scale[Tokens];
    int sum[Tokens];
};

// Part `part` of `parts` of one unit: the 13 bytes as little-endian words (bytes 0-3, 4-7, 8-11,
// 12) with two aligned streaming loads per part when split, then its words of decode_unit.
template <int Parts>
__device__ __forceinline__ void load_unit_part(const std::uint8_t* __restrict__ row_codes, int unit,
                                               int part, std::uint32_t (&a)[4]) {
    if constexpr (Parts == 1) {
        load_unit(row_codes, unit, a);
    } else {
        const std::int64_t start = std::int64_t(unit) * kUnitBytes;
        const auto* base = reinterpret_cast<const std::uint32_t*>(row_codes + (start & ~std::int64_t(3)));
        const unsigned shift = unsigned(start & 3) * 8u;
        // Part 0 needs bytes 0-7 (words 0-2 of the aligned window), part 1 bytes 8-12 (words 2-3).
        if (part == 0) {
            const std::uint32_t w0 = __ldcs(base), w1 = __ldcs(base + 1), w2 = __ldcs(base + 2);
            a[0] = __funnelshift_r(w0, w1, shift);
            a[1] = __funnelshift_r(w1, w2, shift);
        } else {
            const std::uint32_t w2 = __ldcs(base + 2), w3 = __ldcs(base + 3);
            a[2] = __funnelshift_r(w2, w3, shift);
            a[3] = w3 >> shift;
        }
    }
}

// Decodes part `part` of a unit and stores its code words: all 16 words (Parts == 1), or
// words 0-9 (part 0: byte groups 0 and 1) and words 10-15 (part 1: group 2 and byte 12).
template <int Parts>
__device__ __forceinline__ void decode_store_part(const std::uint32_t (&a)[4], int part,
                                                  std::uint32_t* __restrict__ code_words,
                                                  int row, int unit) {
    const auto group_words = [](std::uint32_t bytes, std::uint32_t* words) {
        std::uint32_t even = bytes & 0x00ff00ffu;
        std::uint32_t odd  = (bytes >> 8) & 0x00ff00ffu;
#pragma unroll
        for (int m = 0; m < 5; ++m) {
            even *= 3u;
            odd *= 3u;
            words[m] = ((even >> 8) & 0x00030003u) | (odd & 0x03000300u);
            even &= 0x00ff00ffu;
            odd &= 0x00ff00ffu;
        }
    };
    const auto last_word = [](std::uint32_t bytes) {
        std::uint32_t r = bytes & 0xffu, last = 0;
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            r *= 3u;
            last |= (r >> 8) << (8 * m);
            r &= 0xffu;
        }
        return last;
    };
    std::uint32_t* base = code_words + row * 32;
    const auto chunk    = [&](int c) { return base + ((unit * 4 + c) ^ (row & 7)) * 4; };
    if (Parts == 1 || part == 0) {
        std::uint32_t words[10];
        group_words(a[0], words);
        group_words(a[1], words + 5);
        *reinterpret_cast<uint4*>(chunk(0)) = make_uint4(words[0], words[1], words[2], words[3]);
        *reinterpret_cast<uint4*>(chunk(1)) = make_uint4(words[4], words[5], words[6], words[7]);
        *reinterpret_cast<uint2*>(chunk(2)) = make_uint2(words[8], words[9]);
    }
    if (Parts == 1 || part == 1) {
        std::uint32_t words[5];
        group_words(a[2], words);
        *reinterpret_cast<uint2*>(chunk(2) + 2) = make_uint2(words[0], words[1]);
        *reinterpret_cast<uint4*>(chunk(3)) = make_uint4(words[2], words[3], words[4], last_word(a[3]));
    }
}

template <int WarpsN>
__global__ void __launch_bounds__(GemmConfig<WarpsN>::kThreads, GemmConfig<WarpsN>::kMinBlocks)
    gemm_kernel(const std::uint8_t* __restrict__ qx, const float* __restrict__ group_scale,
                const int* __restrict__ group_sum, const std::uint8_t* __restrict__ codes,
                const __half* __restrict__ scales, std::int64_t scale_row_halves, int k,
                int tokens, Outputs outputs, bool accumulate) {
    using Config             = GemmConfig<WarpsN>;
    constexpr int kTokens    = Config::kTokens;
    constexpr int kThreads   = Config::kThreads;
    constexpr int kParts     = Config::kParts;
    __shared__ __align__(128) GemmStage<kTokens> stages[2];
    __shared__ __align__(128) std::uint32_t code_words[kGemmRows * 32];
    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int gid    = lane >> 2;
    const int lid    = lane & 3;
    const int wm     = warp & 1;
    const int wn     = warp >> 1;
    const int row0   = static_cast<int>(blockIdx.x) * kGemmRows;
    const int token0 = static_cast<int>(blockIdx.y) * kTokens;
    const int live   = min(kTokens, tokens - token0);
    const int steps  = k / kStageK;
    const std::int64_t row_bytes = std::int64_t(k / kUnitColumns) * kUnitBytes;
    // Decode item (row, unit) = (item / 2, item % 2); the part is uniform per warp.
    const int my_part = tid / (kThreads / kParts);
    const int my_item = tid % (kThreads / kParts);
    const int my_unit = my_item & 1;
    const int my_row  = my_item >> 1;
    const std::uint8_t* my_codes = codes + std::int64_t(row0 + my_row) * row_bytes;

    const auto stage_x = [&](int step, GemmStage<kTokens>& s) {
#pragma unroll
        for (int i = 0; i < kTokens * (kStageK / 16) / kThreads; ++i) {
            const int item   = tid + i * kThreads;
            const int token  = item >> 3;
            const int chunk  = item & 7;
            const int source = token < live ? token0 + token : token0;
            cp_async_zfill<16>(&s.x[token * kStageK + ((chunk ^ (token & 7)) << 4)],
                               qx + std::int64_t(source) * k + std::int64_t(step) * kStageK +
                                   chunk * 16,
                               token < live ? 16 : 0);
        }
        // kThreads = 2 kTokens: the first half stages the scales, the second the sums.
        const int token  = tid & (kTokens - 1);
        const int source = token < live ? token0 + token : token0;
        const std::int64_t index = std::int64_t(source) * (k / kStageK) + step;
        if (tid < kTokens) {
            cp_async_zfill<4>(&s.scale[token], group_scale + index, token < live ? 4 : 0);
        } else {
            cp_async_zfill<4>(&s.sum[token], group_sum + index, token < live ? 4 : 0);
        }
    };

    // ldmatrix row addresses. A (x4, per mt and ks): matrices rows 0-7 / 8-15 of chunk 2 ks,
    // then of chunk 2 ks + 1, i.e. fragment registers a0..a3. B (x4, per nt pair and ks): tokens
    // of nt at chunks 2 ks, 2 ks + 1, then of nt + 1.
    const int a_row    = wm * 32 + (lane & 7) + ((lane >> 3) & 1) * 8;
    const int a_chunk  = lane >> 4;
    const int b_token  = wn * 32 + (lane >> 4) * 8 + (lane & 7);
    const int b_chunk  = (lane >> 3) & 1;

    float acc[2][4][4] = {};
    std::uint32_t raw[4];
    load_unit_part<kParts>(my_codes, my_unit, my_part, raw);
    decode_store_part<kParts>(raw, my_part, code_words, my_row, my_unit);
    stage_x(0, stages[0]);
    cp_commit();
    for (int step = 0; step < steps; ++step) {
        if (step + 1 < steps) {
            stage_x(step + 1, stages[(step + 1) & 1]);
            load_unit_part<kParts>(my_codes, 2 * (step + 1) + my_unit, my_part, raw);
        }
        cp_commit();
        cp_wait<1>();
        __syncthreads();
        const GemmStage<kTokens>& s = stages[step & 1];
        int group[2][4][4] = {};
#pragma unroll
        for (int ks = 0; ks < kStageK / 32; ++ks) {
            unsigned b[4][2];
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                const int token = b_token + np * 16;
                const int chunk = 2 * ks + b_chunk;
                ldmatrix_x4(b[2 * np][0], b[2 * np][1], b[2 * np + 1][0], b[2 * np + 1][1],
                            smem_addr(&s.x[token * kStageK + ((chunk ^ (token & 7)) << 4)]));
            }
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                unsigned a0, a1, a2, a3;
                ldmatrix_x4(a0, a1, a2, a3,
                            smem_addr(&code_words[gemm_word(a_row + mt * 16, 4 * (2 * ks + a_chunk))]));
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
        if (step + 1 < steps) decode_store_part<kParts>(raw, my_part, code_words, my_row, my_unit);
    }

    // Fragment C: [0], [1] are row gid, tokens 2*lid and 2*lid+1; [2], [3] are row gid+8.
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const OutputRow out = output_row(outputs, row0 + wm * 32 + mt * 16 + gid + half * 8);
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int token = wn * 32 + nt * 8 + 2 * lid + j;
                    if (token < live) store_row(out, token0 + token, acc[mt][nt][2 * half + j], accumulate);
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail::t5_a8
