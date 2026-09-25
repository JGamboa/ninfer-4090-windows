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
// 1024-column block is fused into the quantization, and so is the producer of x when the caller
// passes it as an input prologue (RMSNorm of raw rows, SwiGLU of gate and up).

#include "ops/common/math.cuh"
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

// Quantization: one CTA of 256 threads per (1024-column block, token). Thread i evaluates the
// weight-input columns 4 i .. 4 i + 3 of its block through an input prologue, so warp w owns
// 128-column group w. With a rotated weight the block first goes through (1/32) H (signs * x)
// in FP32 shared memory (the butterfly of ops/hadamard). Neither the prologue's result nor its
// rotation is rounded to BF16.
constexpr int kQuantizeBlock   = 1024;
constexpr int kQuantizeThreads = 256;

__device__ __forceinline__ void unpack_bf16x4(uint2 word, float (&value)[4]) {
    const float2 low  = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&word.x));
    const float2 high = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&word.y));
    value[0]          = low.x;
    value[1]          = low.y;
    value[2]          = high.x;
    value[3]          = high.y;
}

// The input x [K,T] itself.
struct PlainInput {
    const __nv_bfloat16* x;

    __device__ __forceinline__ void operator()(int k, int token, int column,
                                               float (&value)[4]) const {
        unpack_bf16x4(load_ldg<uint2>(x + std::int64_t(token) * k + column), value);
    }
};

// ops::rmsnorm of the raw rows x [K,T]: x rsqrt(mean(x^2) + eps) gain, gain = 1 + weight with
// unit_offset, else weight. Every CTA of a token reduces the whole row in the same order.
struct RmsNormInput {
    const __nv_bfloat16* x;
    const __nv_bfloat16* weight;
    float eps;
    bool unit_offset;

    __device__ __forceinline__ void operator()(int k, int token, int column,
                                               float (&value)[4]) const {
        constexpr int kLoads = 4; // row words in flight per thread
        __shared__ float warp_sums[kQuantizeThreads / 32];
        const auto* row = reinterpret_cast<const uint2*>(x + std::int64_t(token) * k);
        const int tid   = static_cast<int>(threadIdx.x);
        const int words = k / 4;
        float sum       = 0.0f;
        uint2 own{};
        for (int first = tid; first < words; first += kLoads * kQuantizeThreads) {
            uint2 packed[kLoads];
#pragma unroll
            for (int j = 0; j < kLoads; ++j) {
                const int word = first + j * kQuantizeThreads;
                packed[j]      = word < words ? __ldg(row + word) : make_uint2(0u, 0u);
            }
#pragma unroll
            for (int j = 0; j < kLoads; ++j) {
                if (first + j * kQuantizeThreads == column / 4) own = packed[j];
                float f[4];
                unpack_bf16x4(packed[j], f);
                sum += f[0] * f[0] + f[1] * f[1] + f[2] * f[2] + f[3] * f[3];
            }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xffffffffu, sum, offset);
        }
        if ((tid & 31) == 0) warp_sums[tid >> 5] = sum;
        __syncthreads();
        float total = 0.0f;
#pragma unroll
        for (int w = 0; w < kQuantizeThreads / 32; ++w) total += warp_sums[w];
        const float inverse = rsqrtf(total / static_cast<float>(k) + eps);
        unpack_bf16x4(own, value);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float gain = __bfloat162float(weight[column + i]) + (unit_offset ? 1.0f : 0.0f);
            value[i]         = value[i] * inverse * gain;
        }
    }
};

// SwiGLU of the gate and up rows [K,T]: silu(gate) * up (exact silu, as ops::silu_mul).
struct SwiGluInput {
    const __nv_bfloat16* gate;
    const __nv_bfloat16* up;

    __device__ __forceinline__ void operator()(int k, int token, int column,
                                               float (&value)[4]) const {
        const std::int64_t offset = std::int64_t(token) * k + column;
        float g[4], u[4];
        unpack_bf16x4(load_ldg<uint2>(gate + offset), g);
        unpack_bf16x4(load_ldg<uint2>(up + offset), u);
#pragma unroll
        for (int i = 0; i < 4; ++i) value[i] = silu(g[i]) * u[i];
    }
};

template <class Input, bool Rotate>
__global__ void __launch_bounds__(kQuantizeThreads)
    quantize_kernel(Input input, const __nv_bfloat16* __restrict__ signs, int k,
                    std::uint32_t* __restrict__ qx, float* __restrict__ group_scale,
                    int* __restrict__ group_sum, int* __restrict__ slice_sum) {
    const int tid    = static_cast<int>(threadIdx.x);
    const int token  = static_cast<int>(blockIdx.y);
    const int column = static_cast<int>(blockIdx.x) * kQuantizeBlock + 4 * tid;
    float value[4];
    input(k, token, column, value);
    if constexpr (Rotate) {
        __shared__ __align__(16) float values[kQuantizeBlock];
        float4 signed_value;
        signed_value.x = value[0] * __bfloat162float(signs[column]);
        signed_value.y = value[1] * __bfloat162float(signs[column + 1]);
        signed_value.z = value[2] * __bfloat162float(signs[column + 2]);
        signed_value.w = value[3] * __bfloat162float(signs[column + 3]);
        *reinterpret_cast<float4*>(&values[4 * tid]) = signed_value;
        __syncthreads();
#pragma unroll
        for (int stride = 1; stride < kQuantizeBlock; stride <<= 1) {
#pragma unroll
            for (int pair = tid; pair < kQuantizeBlock / 2; pair += kQuantizeThreads) {
                const int low        = (pair / stride) * 2 * stride + pair % stride;
                const float a        = values[low];
                const float b        = values[low + stride];
                values[low]          = a + b;
                values[low + stride] = a - b;
            }
            __syncthreads();
        }
        const float4 rotated = *reinterpret_cast<const float4*>(&values[4 * tid]);
        value[0]             = rotated.x * 0x1p-5f;
        value[1]             = rotated.y * 0x1p-5f;
        value[2]             = rotated.z * 0x1p-5f;
        value[3]             = rotated.w * 0x1p-5f;
    }
    quantize_group(value, tid & 31, token, column / 128, k, qx, group_scale, group_sum, slice_sum);
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
// dp4a GEMV for decode and single-lane MTP verification (T <= 4). A CTA of two warps owns eight
// rows: each half-warp two rows, its 16 lanes striding the units (K / 64 is a multiple of 16, so
// no lane idles). Per unit and row: four streaming loads, the arithmetic decode, and one dp4a
// per code word and token; the activation words of a unit are shared by both rows. The dp4a work
// grows with T while the decode does not, so wider T takes the tensor-core small-T route below.

constexpr int kGemvThreads     = 64;
constexpr int kGemvRowsPerHalf = 2;
constexpr int kGemvRowsPerCta  = kGemvThreads / 16 * kGemvRowsPerHalf;
constexpr int kGemvMaxTokens   = 4;

template <int Tile>
__global__ void __launch_bounds__(kGemvThreads)
    gemv_kernel(const uint4* __restrict__ qx, const float* __restrict__ group_scale,
                const int* __restrict__ slice_sum, const std::uint8_t* __restrict__ codes,
                const __half* __restrict__ scales, std::int64_t scale_row_halves, int n, int k,
                Outputs outputs, bool accumulate) {
    static_assert(Tile >= 1 && Tile <= kGemvMaxTokens);
    constexpr int R = kGemvRowsPerHalf;
    const int warp  = static_cast<int>(threadIdx.x) / 32;
    const int lane  = static_cast<int>(threadIdx.x) % 32;
    const int hl    = lane & 15;
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
        uint4 xs[Tile][4];
        int offset[Tile];
        float step[Tile];
#pragma unroll
        for (int t = 0; t < Tile; ++t) {
            const uint4* row = qx + std::int64_t(t) * (k / 16) + std::int64_t(u) * 4;
#pragma unroll
            for (int w = 0; w < 4; ++w) xs[t][w] = __ldg(row + w);
            offset[t] = __ldg(slice_sum + std::int64_t(t) * slices + 2 * u) +
                        __ldg(slice_sum + std::int64_t(t) * slices + 2 * u + 1);
            step[t] = __ldg(group_scale + std::int64_t(t) * groups + (u >> 1));
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            int dot[Tile];
#pragma unroll
            for (int t = 0; t < Tile; ++t) dot[t] = 0;
#pragma unroll
            for (int w = 0; w < 16; ++w) {
#pragma unroll
                for (int t = 0; t < Tile; ++t) {
                    const std::uint32_t lanes[4] = {xs[t][w >> 2].x, xs[t][w >> 2].y,
                                                    xs[t][w >> 2].z, xs[t][w >> 2].w};
                    dot[t] = __dp4a(static_cast<int>(words[r][w]), static_cast<int>(lanes[w & 3]), dot[t]);
                }
            }
#pragma unroll
            for (int t = 0; t < Tile; ++t) {
                acc[r][t] = fmaf(scale[r] * step[t], static_cast<float>(dot[t] - offset[t]), acc[r][t]);
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
            if (hl == t) store_row(output_row(outputs, row), t, acc[r][t], accumulate);
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

// ---------------------------------------------------------------------------------------------
// Small-T tensor-core GEMV for the verification of concurrent MTP lanes (T = 5..32; also any T
// of a weight whose rows are not a multiple of the GEMM's 64). The dp4a GEMV's work grows with T
// and the prefill GEMM gives each 64-row CTA the whole K (80 CTAs for a 5120-row weight), so
// neither fits here. A CTA of four warps owns 16 rows and 8 NTiles tokens; warp w takes the
// 128-column groups w, w + 4, ... of the row block (K % 1024 == 0: every warp has K / 512 >= 2
// groups), and the four FP32 partial sums are added in shared memory in warp order, so the
// weights are read once per token tile with one CTA per 16 rows.
//
// Per group each lane decodes one (row, unit) = (lane / 2, lane % 2) as the GEMV does (the next
// two groups' bytes are in flight meanwhile) into the warp's shared code words, and the warp
// multiplies 16 rows x 128 columns x 8 NTiles tokens with m16n8k32 s8 MMAs, A by `ldmatrix`.
// The k order inside the MMAs is permuted so that lane `lid` reads the activation words
// 8 lid .. 8 lid + 7 of its token with two 16-byte loads: MMA step ks, fragment half h (a0/a1
// and b0 for h = 0, a2/a3 and b1 for h = 1) holds group word 8 lid + 2 ks + h, i.e. code word
// w of unit u (group word 16 u + w) is stored at chunk w % 8, word 2 u + w / 8 of its row (the
// chunk XORed with the row as in the GEMM). The group sums correct the code offset as there.

constexpr int kSmallWarps    = 4;
constexpr int kSmallThreads  = 32 * kSmallWarps;
constexpr int kSmallRows     = 16;
constexpr int kSmallMaxTiles = 4; // 32 tokens per CTA

template <int NTiles>
__global__ void __launch_bounds__(kSmallThreads, NTiles <= 2 ? 4 : 3)
    small_t_kernel(const std::uint8_t* __restrict__ qx, const float* __restrict__ group_scale,
                   const int* __restrict__ group_sum, const std::uint8_t* __restrict__ codes,
                   const __half* __restrict__ scales, std::int64_t scale_row_halves, int n, int k,
                   int tokens, Outputs outputs, bool accumulate) {
    static_assert(NTiles >= 1 && NTiles <= kSmallMaxTiles);
    constexpr int kTokens = 8 * NTiles;
    // Per warp: 16 rows x 32 code words per group; after the loop, the partial sums [token][row].
    __shared__ __align__(128) std::uint32_t warp_words[kSmallWarps][kSmallRows * 32];
    static_assert(kTokens * kSmallRows <= kSmallRows * 32);
    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int gid    = lane >> 2;
    const int lid    = lane & 3;
    const int row0   = static_cast<int>(blockIdx.x) * kSmallRows;
    const int token0 = static_cast<int>(blockIdx.y) * kTokens;
    const int live   = min(kTokens, tokens - token0);
    const int groups = k / kStageK;
    const std::int64_t row_bytes = std::int64_t(k / kUnitColumns) * kUnitBytes;
    std::uint32_t* code_words    = warp_words[warp];

    // Rows past n alias the last row and tokens past `live` the last live token; their results
    // are never stored.
    const int my_row             = lane >> 1;
    const int my_unit            = lane & 1;
    const std::uint8_t* my_codes = codes + std::int64_t(min(row0 + my_row, n - 1)) * row_bytes;
    const __half* top_scales     = scales + std::int64_t(min(row0 + gid, n - 1)) * scale_row_halves;
    const __half* bottom_scales =
        scales + std::int64_t(min(row0 + gid + 8, n - 1)) * scale_row_halves;
    // B operand: token nt * 8 + gid, bytes 32 lid .. 32 lid + 31 of each group; C fragment
    // tokens nt * 8 + 2 lid + j for the group scales and sums.
    int x_offset[NTiles], token_offset[NTiles][2];
#pragma unroll
    for (int nt = 0; nt < NTiles; ++nt) {
        x_offset[nt] = (token0 + min(nt * 8 + gid, live - 1)) * k + 32 * lid;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            token_offset[nt][j] = (token0 + min(nt * 8 + 2 * lid + j, live - 1)) * groups;
        }
    }
    // ldmatrix row addresses of the A fragments (as in the GEMM).
    const int a_row   = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int a_chunk = lane >> 4;

    float acc[NTiles][4];
#pragma unroll
    for (int nt = 0; nt < NTiles; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[nt][e] = 0.0f;
    }
    std::uint32_t raw[4], next[4];
    load_unit(my_codes, 2 * warp + my_unit, raw);
    load_unit(my_codes, 2 * (warp + kSmallWarps) + my_unit, next);
#pragma unroll 1
    for (int group = warp; group < groups; group += kSmallWarps) {
        uint4 x[NTiles][2];
#pragma unroll
        for (int nt = 0; nt < NTiles; ++nt) {
            const auto* source = reinterpret_cast<const uint4*>(qx + x_offset[nt] + group * kStageK);
            x[nt][0]           = __ldg(source);
            x[nt][1]           = __ldg(source + 1);
        }
        {
            std::uint32_t words[16];
            decode_unit(raw, words);
            std::uint32_t* row_words = code_words + my_row * 32 + 2 * my_unit;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                *reinterpret_cast<uint2*>(row_words + ((c ^ (my_row & 7)) << 2)) =
                    make_uint2(words[c], words[c + 8]);
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) raw[i] = next[i];
        if (group + 2 * kSmallWarps < groups) {
            load_unit(my_codes, 2 * (group + 2 * kSmallWarps) + my_unit, next);
        }
        __syncwarp();
        int dot[NTiles][4];
#pragma unroll
        for (int nt = 0; nt < NTiles; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) dot[nt][e] = 0;
        }
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            unsigned a0, a1, a2, a3;
            ldmatrix_x4(a0, a1, a2, a3,
                        smem_addr(&code_words[gemm_word(a_row, 4 * (2 * ks + a_chunk))]));
#pragma unroll
            for (int nt = 0; nt < NTiles; ++nt) {
                // b0, b1 = group words 8 lid + 2 ks and 8 lid + 2 ks + 1 of the lane's token.
                const uint4& xs = x[nt][ks >> 1];
                mma_s8(dot[nt][0], dot[nt][1], dot[nt][2], dot[nt][3], a0, a1, a2, a3,
                       ks & 1 ? xs.z : xs.x, ks & 1 ? xs.w : xs.y);
            }
        }
        // Every lane has its fragments before the next group's code words are stored.
        __syncwarp();
        const float top    = __half2float(__ldg(top_scales + group));
        const float bottom = __half2float(__ldg(bottom_scales + group));
        // Fragment C: [0], [1] are row gid, tokens 2 lid and 2 lid + 1; [2], [3] row gid + 8.
#pragma unroll
        for (int nt = 0; nt < NTiles; ++nt) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const float step = __ldg(group_scale + token_offset[nt][j] + group);
                const int sum    = __ldg(group_sum + token_offset[nt][j] + group);
                acc[nt][j] = fmaf(top * step, static_cast<float>(dot[nt][j] - sum), acc[nt][j]);
                acc[nt][2 + j] =
                    fmaf(bottom * step, static_cast<float>(dot[nt][2 + j] - sum), acc[nt][2 + j]);
            }
        }
    }

    // The warp's partial sums replace its code words as [token][row]; then every thread adds
    // the four warps' sums of (row, token) items in warp order, rows fastest (coalesced stores).
    auto* partial = reinterpret_cast<float*>(code_words);
#pragma unroll
    for (int nt = 0; nt < NTiles; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            partial[(nt * 8 + 2 * lid + (e & 1)) * kSmallRows + gid + (e >> 1) * 8] = acc[nt][e];
        }
    }
    __syncthreads();
    for (int item = tid; item < kTokens * kSmallRows; item += kSmallThreads) {
        const int row   = item % kSmallRows;
        const int token = item / kSmallRows;
        if (token >= live || row0 + row >= n) continue;
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < kSmallWarps; ++w) sum += reinterpret_cast<const float*>(warp_words[w])[item];
        store_row(output_row(outputs, row0 + row), token0 + token, sum, accumulate);
    }
}

} // namespace ninfer::ops::detail::t5_a8
