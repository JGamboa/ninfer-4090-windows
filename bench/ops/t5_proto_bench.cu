// Prototype: base-3 ternary codes (five trits per byte, 13 bytes per 64 columns = 1.625 bits per
// weight) against the production 2-bit t2 A8 GEMV, at the Bonsai 2 27B decode shapes. Both
// kernels read the same int8 activation (production quantization formula and layout) and the
// same FP16 group scales; weight copies rotate beyond L2. The t5 kernel decodes each byte
// through a 243-entry shared-memory table into the t2 16-column code words, then runs the same
// shift/mask + dp4a product. Output: GEMV-only microseconds, weight GB/s, and the largest
// relative difference between the two kernels' outputs.
#include "ops/linear/t2/t2_a8.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <type_traits>
#include <vector>

#define CHECK(call)                                                                        \
    do {                                                                                   \
        const cudaError_t error = (call);                                                  \
        if (error != cudaSuccess) {                                                        \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            std::exit(1);                                                                  \
        }                                                                                  \
    } while (0)

namespace t2 = ninfer::ops::detail::t2_a8;

namespace {

constexpr int kUnitColumns   = 64;
constexpr int kUnitBytes     = 13; // 12 bytes of five trits + 1 byte of four
constexpr int kT5Threads     = 64; // two warps
constexpr int kT5RowsPerHalf = 2;  // a half-warp owns two rows; 16 lanes stride the units
constexpr int kT5RowsPerBlock = kT5Threads / 16 * kT5RowsPerHalf;

constexpr std::size_t kColdBytes = 512ull << 20;
constexpr int kIterations        = 40;
constexpr int kRepeats           = 7;

__global__ void __launch_bounds__(128)
    quantize_plain(const __nv_bfloat16* __restrict__ x, int k, std::uint32_t* __restrict__ qx,
                   float* __restrict__ group_scale, int* __restrict__ group_sum,
                   int* __restrict__ slice_sum) {
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int group = static_cast<int>(blockIdx.x) * 4 + warp;
    const int token = static_cast<int>(blockIdx.y);
    const __nv_bfloat16* source = x + std::int64_t(token) * k + group * 128;
    float value[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) value[i] = __bfloat162float(source[16 * (lane >> 2) + 4 * i + (lane & 3)]);
    t2::quantize_group(value, lane, token, group, k, qx, group_scale, group_sum, slice_sum);
}

// Precomputed tables: 243 entries, and 243 x 8 copies (entry-major), 32-bit.
__device__ __align__(16) std::uint32_t g_lut1[244];
__device__ __align__(16) std::uint32_t g_lut8[243 * 8];

template <int Tile, int Mode>
__global__ void __launch_bounds__(kT5Threads)
    t5_gemv_kernel(const uint4* __restrict__ qx, const float* __restrict__ group_scale,
                   const int* __restrict__ slice_sum, const std::uint8_t* __restrict__ codes,
                   const __half* __restrict__ scales, int n, int k, int tokens,
                   __nv_bfloat16* __restrict__ out, std::uint32_t opaque_zero) {
    // Mode 3: eight 32-bit copies, lane & 7 picks one; Mode 4: 32 16-bit copies, one per lane.
    constexpr int kLutEntries = Mode == 4 ? 243 * 32 : (Mode == 3 || Mode == 6) ? 243 * 8 : 243;
    constexpr int kCopies     = Mode == 4 ? 32 : (Mode == 3 || Mode == 6) ? 8 : 1;
    using Entry               = std::conditional_t<Mode == 4, std::uint16_t, std::uint32_t>;
    __shared__ __align__(16) Entry lut[kLutEntries + (Mode == 0 || Mode >= 5 ? 1 : 0)];
    if constexpr (Mode == 5 || Mode == 6) {
        // Mode 5: 243 entries copied from global; Mode 6: 8 copies copied from global.
        constexpr int kWords = Mode == 5 ? 244 : 243 * 8;
        const auto* source   = reinterpret_cast<const uint4*>(Mode == 5 ? g_lut1 : g_lut8);
        auto* target         = reinterpret_cast<uint4*>(lut);
        for (int e = static_cast<int>(threadIdx.x); e < kWords / 4; e += kT5Threads) target[e] = source[e];
    } else
    for (int e = static_cast<int>(threadIdx.x); e < kLutEntries; e += kT5Threads) {
        int b              = e / kCopies;
        std::uint32_t word = 0;
        for (int m = 0; m < 5; ++m) {
            word |= std::uint32_t(b % 3) << (2 * m);
            b /= 3;
        }
        lut[e] = static_cast<Entry>(word);
    }
    __syncthreads();
    const std::uint32_t copy = threadIdx.x & std::uint32_t(kCopies - 1);

    constexpr int R  = kT5RowsPerHalf;
    const int warp   = static_cast<int>(threadIdx.x) / 32;
    const int lane   = static_cast<int>(threadIdx.x) % 32;
    const int hl     = lane & 15;
    const int token0 = static_cast<int>(blockIdx.y) * Tile;
    const int live   = min(Tile, tokens - token0);
    const int row0 = ((static_cast<int>(blockIdx.x) * (kT5Threads / 32) + warp) * 2 + (lane >> 4)) * R;
    const int units             = k / kUnitColumns;
    const std::int64_t row_bytes = std::int64_t(units) * kUnitBytes;
    const int slices = k / 32, groups = k / 128;
    const std::uint8_t* row_codes[R];
    const __half* row_scales[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const std::int64_t row = min(row0 + r, n - 1);
        row_codes[r]           = codes + row * row_bytes;
        row_scales[r]          = scales + row * groups;
    }
    float acc[R][Tile];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < Tile; ++t) acc[r][t] = 0.0f;
    }

    for (int u = hl; u < units; u += 16) {
        std::uint32_t words[R][4];
        float scale[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const std::int64_t start = std::int64_t(u) * kUnitBytes;
            const auto* base =
                reinterpret_cast<const std::uint32_t*>(row_codes[r] + (start & ~std::int64_t(3)));
            const unsigned shift = unsigned(start & 3) * 8u;
            const std::uint32_t w0 = __ldcs(base), w1 = __ldcs(base + 1), w2 = __ldcs(base + 2),
                                w3 = __ldcs(base + 3);
            const std::uint32_t a[4] = {__funnelshift_r(w0, w1, shift), __funnelshift_r(w1, w2, shift),
                                        __funnelshift_r(w2, w3, shift), w3 >> shift};
            std::uint32_t l[13];
#pragma unroll
            for (int i = 0; i < 13; ++i) {
                const std::uint32_t byte = (a[i >> 2] >> (8 * (i & 3))) & 0xffu;
                // Mode 1: conflict-free table index (same instruction count); Mode 2: no decode.
                if constexpr (Mode == 1) {
                    l[i] = lut[(byte & opaque_zero) + (threadIdx.x & 31u)];
                } else if constexpr (Mode == 2) {
                    l[i] = byte;
                } else {
                    l[i] = lut[byte * kCopies + copy];
                }
            }
            words[r][0] = l[0] | (l[1] << 10) | (l[2] << 20) | (l[3] << 30);
            words[r][1] = (l[3] >> 2) | (l[4] << 8) | (l[5] << 18) | (l[6] << 28);
            words[r][2] = (l[6] >> 4) | (l[7] << 6) | (l[8] << 16) | (l[9] << 26);
            words[r][3] = (l[9] >> 6) | (l[10] << 4) | (l[11] << 14) | (l[12] << 24);
            scale[r]    = __half2float(row_scales[r][u >> 1]);
        }
        uint4 xs[Tile][4];
        int offset[Tile];
        float step[Tile];
#pragma unroll
        for (int t = 0; t < Tile; ++t) {
            if (t < live) {
                const std::int64_t token = token0 + t;
                const uint4* row         = qx + token * (k / 16) + std::int64_t(u) * 4;
#pragma unroll
                for (int w = 0; w < 4; ++w) xs[t][w] = __ldg(row + w);
                offset[t] = __ldg(slice_sum + token * slices + 2 * u) +
                            __ldg(slice_sum + token * slices + 2 * u + 1);
                step[t] = __ldg(group_scale + token * groups + (u >> 1));
            } else {
#pragma unroll
                for (int w = 0; w < 4; ++w) xs[t][w] = make_uint4(0, 0, 0, 0);
                offset[t] = 0;
                step[t]   = 0.0f;
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            int dot[Tile];
#pragma unroll
            for (int t = 0; t < Tile; ++t) dot[t] = 0;
#pragma unroll
            for (int w = 0; w < 4; ++w) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int c = static_cast<int>((words[r][w] >> (2 * j)) & t2::kCodeMask);
#pragma unroll
                    for (int t = 0; t < Tile; ++t) {
                        const std::uint32_t lanes[4] = {xs[t][w].x, xs[t][w].y, xs[t][w].z, xs[t][w].w};
                        dot[t] = __dp4a(c, static_cast<int>(lanes[j]), dot[t]);
                    }
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
            if (t < live && hl == t) out[std::int64_t(token0 + t) * n + row] = __float2bfloat16_rn(acc[r][t]);
        }
    }
}

// Arithmetic (TQ1_0-style) variant. A byte holds q = ceil(v * 256 / 243) with
// v = sum_m t_m 3^(4 - m); trit m is ((q * 3^m) mod 256) * 3 >> 8. In unit bytes 4g..4g+3
// (g = 0..2), trit m of byte i is column 20 g + 4 m + i; byte 12 holds columns 60..63 as trits
// 0..3. Even and odd bytes run in 16-bit lanes, so every m yields one dp4a word of four codes
// that multiplies the activation word of the same four columns in natural order.
__device__ __forceinline__ void t5a_decode(const std::uint32_t (&a)[4], std::uint32_t (&words)[16]) {
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

template <int Tile>
__global__ void __launch_bounds__(kT5Threads)
    t5a_gemv_kernel(const uint4* __restrict__ qx, const float* __restrict__ group_scale,
                    const int* __restrict__ slice_sum, const std::uint8_t* __restrict__ codes,
                    const __half* __restrict__ scales, int n, int k, int tokens,
                    __nv_bfloat16* __restrict__ out) {
    constexpr int R  = kT5RowsPerHalf;
    const int warp   = static_cast<int>(threadIdx.x) / 32;
    const int lane   = static_cast<int>(threadIdx.x) % 32;
    const int hl     = lane & 15;
    const int token0 = static_cast<int>(blockIdx.y) * Tile;
    const int live   = min(Tile, tokens - token0);
    const int row0 = ((static_cast<int>(blockIdx.x) * (kT5Threads / 32) + warp) * 2 + (lane >> 4)) * R;
    const int units              = k / kUnitColumns;
    const std::int64_t row_bytes = std::int64_t(units) * kUnitBytes;
    const int slices = k / 32, groups = k / 128;
    const std::uint8_t* row_codes[R];
    const __half* row_scales[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const std::int64_t row = min(row0 + r, n - 1);
        row_codes[r]           = codes + row * row_bytes;
        row_scales[r]          = scales + row * groups;
    }
    float acc[R][Tile];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < Tile; ++t) acc[r][t] = 0.0f;
    }
    for (int u = hl; u < units; u += 16) {
        uint4 xs[Tile][4];
        int offset[Tile];
        float step[Tile];
#pragma unroll
        for (int t = 0; t < Tile; ++t) {
            if (t < live) {
                const std::int64_t token = token0 + t;
                const uint4* row         = qx + token * (k / 16) + std::int64_t(u) * 4;
#pragma unroll
                for (int w = 0; w < 4; ++w) xs[t][w] = __ldg(row + w);
                offset[t] = __ldg(slice_sum + token * slices + 2 * u) +
                            __ldg(slice_sum + token * slices + 2 * u + 1);
                step[t] = __ldg(group_scale + token * groups + (u >> 1));
            } else {
#pragma unroll
                for (int w = 0; w < 4; ++w) xs[t][w] = make_uint4(0, 0, 0, 0);
                offset[t] = 0;
                step[t]   = 0.0f;
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const std::int64_t start = std::int64_t(u) * kUnitBytes;
            const auto* base =
                reinterpret_cast<const std::uint32_t*>(row_codes[r] + (start & ~std::int64_t(3)));
            const unsigned shift = unsigned(start & 3) * 8u;
            const std::uint32_t w0 = __ldcs(base), w1 = __ldcs(base + 1), w2 = __ldcs(base + 2),
                                w3 = __ldcs(base + 3);
            const std::uint32_t a[4] = {__funnelshift_r(w0, w1, shift), __funnelshift_r(w1, w2, shift),
                                        __funnelshift_r(w2, w3, shift), w3 >> shift};
            std::uint32_t words[16];
            t5a_decode(a, words);
            const float scale = __half2float(row_scales[r][u >> 1]);
            int dot[Tile];
#pragma unroll
            for (int t = 0; t < Tile; ++t) dot[t] = 0;
#pragma unroll
            for (int w = 0; w < 16; ++w) {
#pragma unroll
                for (int t = 0; t < Tile; ++t) {
                    const std::uint32_t lanes[4] = {xs[t][w >> 2].x, xs[t][w >> 2].y, xs[t][w >> 2].z,
                                                    xs[t][w >> 2].w};
                    dot[t] = __dp4a(static_cast<int>(words[w]), static_cast<int>(lanes[w & 3]), dot[t]);
                }
            }
#pragma unroll
            for (int t = 0; t < Tile; ++t) {
                acc[r][t] = fmaf(scale * step[t], static_cast<float>(dot[t] - offset[t]), acc[r][t]);
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
            if (t < live && hl == t) out[std::int64_t(token0 + t) * n + row] = __float2bfloat16_rn(acc[r][t]);
        }
    }
}

// Prefill GEMM for t5a, the production t2 GEMM's schedule (64 rows x 64 tokens, four warps of
// 32 x 32, m16n8k32 s8, one 128-column scale group per double-buffered stage). Thread (row,
// unit) decodes its 13 bytes into 16 natural four-column code words in shared memory; the next
// stage's bytes are loaded before the MMAs and decoded after them. A fragments read the code
// words directly (a0 = columns 4 lid .. 4 lid + 3), B fragments are the t2 GEMM's ldmatrix of a
// natural-order activation.
constexpr int kG5Rows       = 64;
constexpr int kG5Tokens     = 64;
constexpr int kG5Threads    = 128;
constexpr int kG5StageK     = 128;
// 32 code words per row and stage; the four-word group index is XORed with the row (mod 8) so
// the A reads of a warp (gid, lid) hit 32 distinct banks. 16 KiB per stage: three CTAs per SM.
__device__ __forceinline__ int g5_word(int row, int word) { return row * 32 + (word ^ ((row & 7) << 2)); }

struct G5Stage {
    std::uint8_t x[kG5Tokens * kG5StageK];
    float scale[kG5Tokens];
    int sum[kG5Tokens];
};

// Decode 0: arithmetic decode; 1: diagnostic, raw bytes stored as code words (no decode).
template <int Decode>
__global__ void __launch_bounds__(kG5Threads)
    t5a_gemm_kernel(const std::uint8_t* __restrict__ qx, const float* __restrict__ group_scale,
                    const int* __restrict__ group_sum, const std::uint8_t* __restrict__ codes,
                    const __half* __restrict__ scales, int n, int k, int tokens,
                    __nv_bfloat16* __restrict__ out) {
    using namespace ninfer::ops;
    __shared__ __align__(128) G5Stage stages[2];
    __shared__ __align__(128) std::uint32_t code_words[kG5Rows * 32]; // single buffer
    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int gid    = lane >> 2;
    const int lid    = lane & 3;
    const int wm     = warp & 1;
    const int wn     = warp >> 1;
    const int row0   = static_cast<int>(blockIdx.x) * kG5Rows;
    const int token0 = static_cast<int>(blockIdx.y) * kG5Tokens;
    const int live   = min(kG5Tokens, tokens - token0);
    const int steps  = k / kG5StageK;
    const int groups = k / 128;
    const std::int64_t row_bytes = std::int64_t(k / kUnitColumns) * kUnitBytes;
    const int my_row  = tid >> 1;
    const int my_unit = tid & 1;
    const std::uint8_t* my_codes = codes + std::int64_t(row0 + my_row) * row_bytes;

    std::uint32_t raw[4];
    unsigned shift = 0;
    const auto fetch = [&](int step) {
        const std::int64_t start = std::int64_t(2 * step + my_unit) * kUnitBytes;
        const auto* base = reinterpret_cast<const std::uint32_t*>(my_codes + (start & ~std::int64_t(3)));
        shift = unsigned(start & 3) * 8u;
#pragma unroll
        for (int i = 0; i < 4; ++i) raw[i] = __ldcs(base + i);
    };
    const auto decode_store = [&]() {
        const std::uint32_t a[4] = {__funnelshift_r(raw[0], raw[1], shift),
                                    __funnelshift_r(raw[1], raw[2], shift),
                                    __funnelshift_r(raw[2], raw[3], shift), raw[3] >> shift};
        std::uint32_t words[16];
        if constexpr (Decode == 0) {
            t5a_decode(a, words);
        } else {
#pragma unroll
            for (int w = 0; w < 16; ++w) words[w] = a[w & 3] & 0x03030303u;
        }
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            *reinterpret_cast<uint4*>(&code_words[g5_word(my_row, my_unit * 16 + 4 * q)]) =
                make_uint4(words[4 * q], words[4 * q + 1], words[4 * q + 2], words[4 * q + 3]);
        }
    };
    const auto stage_x = [&](int step, G5Stage& s) {
#pragma unroll
        for (int i = 0; i < kG5Tokens * (kG5StageK / 16) / kG5Threads; ++i) {
            const int item   = tid + i * kG5Threads;
            const int token  = item >> 3;
            const int chunk  = item & 7;
            const int source = token < live ? token0 + token : token0;
            cp_async_zfill<16>(&s.x[token * kG5StageK + ((chunk ^ (token & 7)) << 4)],
                               qx + std::int64_t(source) * k + std::int64_t(step) * kG5StageK + chunk * 16,
                               token < live ? 16 : 0);
        }
        const int token  = tid & (kG5Tokens - 1);
        const int source = token < live ? token0 + token : token0;
        const std::int64_t index = std::int64_t(source) * groups + step;
        if (tid < kG5Tokens) {
            cp_async_zfill<4>(&s.scale[token], group_scale + index, token < live ? 4 : 0);
        } else {
            cp_async_zfill<4>(&s.sum[token], group_sum + index, token < live ? 4 : 0);
        }
    };

    float acc[2][4][4] = {};
    fetch(0);
    decode_store();
    stage_x(0, stages[0]);
    cp_commit();
    for (int step = 0; step < steps; ++step) {
        if (step + 1 < steps) {
            stage_x(step + 1, stages[(step + 1) & 1]);
            fetch(step + 1);
        }
        cp_commit();
        cp_wait<1>();
        __syncthreads();
        const G5Stage& s = stages[step & 1];
        int group[2][4][4] = {};
#pragma unroll
        for (int ks = 0; ks < kG5StageK / 32; ++ks) {
            unsigned b[4][2];
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int b_row = wn * 32 + nt * 8 + (lane & 7);
                const int chunk = ks * 2 + ((lane >> 3) & 1);
                ldmatrix_x2(b[nt][0], b[nt][1], smem_addr(&s.x[b_row * kG5StageK + ((chunk ^ (b_row & 7)) << 4)]));
            }
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const int r       = wm * 32 + mt * 16 + gid;
                const unsigned a0 = code_words[g5_word(r, 8 * ks + lid)];
                const unsigned a1 = code_words[g5_word(r + 8, 8 * ks + lid)];
                const unsigned a2 = code_words[g5_word(r, 8 * ks + 4 + lid)];
                const unsigned a3 = code_words[g5_word(r + 8, 8 * ks + 4 + lid)];
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    mma_s8(group[mt][nt][0], group[mt][nt][1], group[mt][nt][2], group[mt][nt][3], a0,
                           a1, a2, a3, b[nt][0], b[nt][1]);
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
            const float top    = __half2float(__ldg(scales + std::int64_t(r) * groups + step));
            const float bottom = __half2float(__ldg(scales + std::int64_t(r + 8) * groups + step));
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int j      = e & 1;
                    const float unit = (e < 2 ? top : bottom) * token_scale[nt][j];
                    acc[mt][nt][e] = fmaf(unit, static_cast<float>(group[mt][nt][e] - token_sum[nt][j]),
                                          acc[mt][nt][e]);
                }
            }
        }
        __syncthreads();
        if (step + 1 < steps) decode_store();
    }
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
                        out[std::int64_t(token0 + token) * n + row] = __float2bfloat16_rn(acc[mt][nt][2 * half + j]);
                    }
                }
            }
        }
    }
}

// Natural-order int8 activation for t5a: lane l of a group holds columns 4 l .. 4 l + 3.
__global__ void __launch_bounds__(128)
    quantize_natural(const __nv_bfloat16* __restrict__ x, int k, std::uint32_t* __restrict__ qx,
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
    t2::quantize_group(value, lane, token, group, k, qx, group_scale, group_sum, slice_sum);
}

template <int Tile>
void launch_t5a(const std::uint32_t* qx, const float* group_scale, const int* slice_sum,
                const std::uint8_t* codes, const __half* scales, int n, int k, int tokens,
                __nv_bfloat16* out) {
    const dim3 grid((n + kT5RowsPerBlock - 1) / kT5RowsPerBlock, (tokens + Tile - 1) / Tile);
    t5a_gemv_kernel<Tile><<<grid, kT5Threads>>>(reinterpret_cast<const uint4*>(qx), group_scale,
                                                slice_sum, codes, scales, n, k, tokens, out);
}

struct Activation {
    std::uint32_t* qx;
    float* group_scale;
    int* group_sum;
    int* slice_sum;
};

template <int Tile>
void launch_t2(const Activation& a, const std::uint8_t* codes, const __half* scales, int n, int k,
               int tokens, __nv_bfloat16* out) {
    constexpr int rows_block = t2::kGemvWarps * t2::kGemvRowsPerWarp;
    t2::Outputs outputs{};
    outputs.data[0] = out;
    outputs.end[0]  = n;
    for (int i = 1; i < t2::kMaxOutputs; ++i) outputs.end[i] = INT_MAX;
    const dim3 grid((n + rows_block - 1) / rows_block, (tokens + Tile - 1) / Tile);
    t2::gemv_kernel<Tile><<<grid, t2::kGemvThreads>>>(
        reinterpret_cast<const uint4*>(a.qx), a.group_scale, a.slice_sum,
        reinterpret_cast<const uint2*>(codes), scales, k / 128, n, k, tokens, outputs, false);
}

template <int Tile, int Mode>
void launch_t5(const Activation& a, const std::uint8_t* codes, const __half* scales, int n, int k,
               int tokens, __nv_bfloat16* out) {
    const dim3 grid((n + kT5RowsPerBlock - 1) / kT5RowsPerBlock, (tokens + Tile - 1) / Tile);
    t5_gemv_kernel<Tile, Mode><<<grid, kT5Threads>>>(reinterpret_cast<const uint4*>(a.qx),
                                                     a.group_scale, a.slice_sum, codes, scales, n,
                                                     k, tokens, out, 0u);
}

// variant -1: production t2; 0: t5; 1: t5 with conflict-free table reads; 2: t5 without decode;
// 3: table replicated 8x (32-bit); 4: table replicated 32x (16-bit).
template <int Tile>
void launch_tile(int variant, const Activation& a, const std::uint8_t* codes, const __half* scales,
                 int n, int k, int tokens, __nv_bfloat16* out) {
    switch (variant) {
    case -1: launch_t2<Tile>(a, codes, scales, n, k, tokens, out); break;
    case 0: launch_t5<Tile, 0>(a, codes, scales, n, k, tokens, out); break;
    case 1: launch_t5<Tile, 1>(a, codes, scales, n, k, tokens, out); break;
    case 2: launch_t5<Tile, 2>(a, codes, scales, n, k, tokens, out); break;
    case 3: launch_t5<Tile, 3>(a, codes, scales, n, k, tokens, out); break;
    case 4: launch_t5<Tile, 4>(a, codes, scales, n, k, tokens, out); break;
    case 5: launch_t5<Tile, 5>(a, codes, scales, n, k, tokens, out); break;
    default: launch_t5<Tile, 6>(a, codes, scales, n, k, tokens, out); break;
    }
}

void launch(int variant, const Activation& a, const std::uint8_t* codes, const __half* scales,
            int n, int k, int tokens, __nv_bfloat16* out) {
    switch (tokens) {
    case 1: launch_tile<1>(variant, a, codes, scales, n, k, tokens, out); break;
    case 2: launch_tile<2>(variant, a, codes, scales, n, k, tokens, out); break;
    case 3: launch_tile<3>(variant, a, codes, scales, n, k, tokens, out); break;
    case 4: launch_tile<4>(variant, a, codes, scales, n, k, tokens, out); break;
    default: launch_tile<8>(variant, a, codes, scales, n, k, tokens, out); break;
    }
    CHECK(cudaGetLastError());
}

float bf16_to_float(std::uint16_t bits) {
    const std::uint32_t word = std::uint32_t(bits) << 16;
    float value;
    std::memcpy(&value, &word, 4);
    return value;
}

} // namespace

int main(int argc, char** argv) {
    std::vector<int> tokens{1, 3, 4, 8};
    if (argc > 1) {
        tokens.clear();
        for (int i = 1; i < argc; ++i) tokens.push_back(std::atoi(argv[i]));
    }
    struct Shape {
        const char* role;
        int n, k;
    };
    const std::array<Shape, 5> shapes{{{"gdn in_proj", 16384, 5120},
                                       {"attn qkvg", 14336, 5120},
                                       {"mlp gate+up", 34816, 5120},
                                       {"o_proj/out", 5120, 6144},
                                       {"mlp down", 5120, 17408}}};
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    {
        void* scratch = nullptr;
        CHECK(cudaMalloc(&scratch, kColdBytes));
        for (int i = 0; i < 200; ++i) CHECK(cudaMemsetAsync(scratch, i, kColdBytes));
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaFree(scratch));
    }
    std::printf("%-12s %6s %6s %3s %8s %8s %8s %8s %8s %8s %10s %10s\n", "role", "N", "K", "T",
                "t2 us", "t5 us", "t5/t2", "arith us", "arith/t2", "ideal", "rel t5", "rel arith");
    {
        std::vector<std::uint32_t> lut1(244, 0), lut8(243 * 8);
        for (int v = 0; v < 243; ++v) {
            int b              = v;
            std::uint32_t word = 0;
            for (int m = 0; m < 5; ++m) {
                word |= std::uint32_t(b % 3) << (2 * m);
                b /= 3;
            }
            lut1[v] = word;
            for (int c = 0; c < 8; ++c) lut8[v * 8 + c] = word;
        }
        CHECK(cudaMemcpyToSymbol(g_lut1, lut1.data(), lut1.size() * 4));
        CHECK(cudaMemcpyToSymbol(g_lut8, lut8.data(), lut8.size() * 4));
    }
    std::mt19937 rng(7);
    for (const Shape& shape : shapes) {
        const int n = shape.n, k = shape.k, units = k / kUnitColumns, groups = k / 128;
        std::vector<std::uint8_t> code(std::size_t(n) * k);
        for (auto& c : code) c = static_cast<std::uint8_t>(rng() % 3);
        std::vector<__half> scale(std::size_t(n) * groups);
        std::uniform_real_distribution<float> magnitude(0.002f, 0.05f);
        for (auto& s : scale) s = __float2half(magnitude(rng));
        std::vector<std::uint8_t> t2_bytes(std::size_t(n) * (k / 4), 0);
        std::vector<std::uint8_t> t5_bytes(std::size_t(n) * units * kUnitBytes + 16, 0);
        std::vector<std::uint8_t> t5a_bytes(t5_bytes.size(), 0);
        for (int row = 0; row < n; ++row) {
            const std::uint8_t* c = &code[std::size_t(row) * k];
            for (int col = 0; col < k; ++col) {
                t2_bytes[std::size_t(row) * (k / 4) + col / 4] |= std::uint8_t(c[col] << (2 * (col % 4)));
            }
            for (int u = 0; u < units; ++u) {
                const std::uint8_t* uc = c + u * kUnitColumns;
                std::uint8_t* t5  = &t5_bytes[(std::size_t(row) * units + u) * kUnitBytes];
                std::uint8_t* t5a = &t5a_bytes[(std::size_t(row) * units + u) * kUnitBytes];
                for (int i = 0; i < kUnitBytes; ++i) {
                    // Table layout: byte i holds columns 5 i + m, least significant trit first.
                    int value = 0, power = 1;
                    for (int m = 0; m < (i < 12 ? 5 : 4); ++m) {
                        value += uc[5 * i + m] * power;
                        power *= 3;
                    }
                    t5[i] = std::uint8_t(value);
                    // Arithmetic layout: byte 4 g + j holds columns 20 g + 4 m + j (byte 12:
                    // columns 60 + m), most significant trit first, scaled to q.
                    int v = 0;
                    for (int m = 0; m < 5; ++m) {
                        const int trit = i < 12 ? uc[20 * (i / 4) + 4 * m + (i % 4)] : (m < 4 ? uc[60 + m] : 0);
                        v = 3 * v + trit;
                    }
                    t5a[i] = std::uint8_t((v * 256 + 242) / 243);
                }
            }
        }
        const std::size_t scale_bytes = scale.size() * sizeof(__half);
        const std::size_t t2_weight   = t2_bytes.size() + scale_bytes;
        const std::size_t t5_weight   = std::size_t(n) * units * kUnitBytes + scale_bytes;
        const int t2_copies = int((kColdBytes + t2_weight - 1) / t2_weight);
        const int t5_copies = int((kColdBytes + t5_weight - 1) / t5_weight);
        std::vector<std::uint8_t*> t2_codes(t2_copies), t5_codes(t5_copies), t5a_codes(t5_copies);
        std::vector<__half*> t2_scales(t2_copies), t5_scales(t5_copies);
        for (int c = 0; c < t2_copies; ++c) {
            CHECK(cudaMalloc(&t2_codes[c], t2_bytes.size()));
            CHECK(cudaMalloc(&t2_scales[c], scale_bytes));
            CHECK(cudaMemcpy(t2_codes[c], t2_bytes.data(), t2_bytes.size(), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(t2_scales[c], scale.data(), scale_bytes, cudaMemcpyHostToDevice));
        }
        for (int c = 0; c < t5_copies; ++c) {
            CHECK(cudaMalloc(&t5_codes[c], t5_bytes.size()));
            CHECK(cudaMalloc(&t5a_codes[c], t5a_bytes.size()));
            CHECK(cudaMalloc(&t5_scales[c], scale_bytes));
            CHECK(cudaMemcpy(t5_codes[c], t5_bytes.data(), t5_bytes.size(), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(t5a_codes[c], t5a_bytes.data(), t5a_bytes.size(), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(t5_scales[c], scale.data(), scale_bytes, cudaMemcpyHostToDevice));
        }
        for (const int t : tokens) {
            std::vector<std::uint16_t> x(std::size_t(k) * t);
            std::uniform_real_distribution<float> uniform(-4.0f, 4.0f);
            for (auto& v : x) {
                const __nv_bfloat16 b = __float2bfloat16(uniform(rng));
                std::memcpy(&v, &b, 2);
            }
            __nv_bfloat16* dx = nullptr;
            Activation a{}, an{};
            __nv_bfloat16 *out2 = nullptr, *out5 = nullptr;
            CHECK(cudaMalloc(&dx, x.size() * 2));
            for (Activation* act : {&a, &an}) {
                CHECK(cudaMalloc(&act->qx, std::size_t(k) * t));
                CHECK(cudaMalloc(&act->group_scale, std::size_t(groups) * t * 4));
                CHECK(cudaMalloc(&act->group_sum, std::size_t(groups) * t * 4));
                CHECK(cudaMalloc(&act->slice_sum, std::size_t(k / 32) * t * 4));
            }
            CHECK(cudaMalloc(&out2, std::size_t(n) * t * 2));
            CHECK(cudaMalloc(&out5, std::size_t(n) * t * 2));
            CHECK(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
            quantize_plain<<<dim3(k / 512, t), 128>>>(dx, k, a.qx, a.group_scale, a.group_sum, a.slice_sum);
            quantize_natural<<<dim3(k / 512, t), 128>>>(dx, k, an.qx, an.group_scale, an.group_sum, an.slice_sum);
            CHECK(cudaGetLastError());

            // 0: production t2; 1: t5 table; 2: t5 arithmetic.
            const auto run = [&](int kernel, int copy, __nv_bfloat16* out) {
                if (t > 8) {
                    // Prefill: production t2 GEMM against the t5 GEMM; no table variant.
                    const dim3 grid(n / t2::kGemmRows, (t + t2::kGemmTokens - 1) / t2::kGemmTokens);
                    if (kernel == 0) {
                        t2::Outputs outputs{};
                        outputs.data[0] = out;
                        outputs.end[0]  = n;
                        for (int i = 1; i < t2::kMaxOutputs; ++i) outputs.end[i] = INT_MAX;
                        t2::gemm_kernel<<<grid, t2::kGemmThreads>>>(
                            reinterpret_cast<const std::uint8_t*>(a.qx), a.group_scale, a.group_sum,
                            t2_codes[copy % t2_copies], t2_scales[copy % t2_copies], k / 128, k, t,
                            outputs, false);
                    } else if (kernel == 1) {
                        t5a_gemm_kernel<1><<<grid, kG5Threads>>>(
                            reinterpret_cast<const std::uint8_t*>(an.qx), an.group_scale, an.group_sum,
                            t5a_codes[copy % t5_copies], t5_scales[copy % t5_copies], n, k, t, out);
                    } else if (kernel == 2) {
                        t5a_gemm_kernel<0><<<grid, kG5Threads>>>(
                            reinterpret_cast<const std::uint8_t*>(an.qx), an.group_scale, an.group_sum,
                            t5a_codes[copy % t5_copies], t5_scales[copy % t5_copies], n, k, t, out);
                    }
                    CHECK(cudaGetLastError());
                    return;
                }
                if (kernel == 0) {
                    launch(-1, a, t2_codes[copy % t2_copies], t2_scales[copy % t2_copies], n, k, t, out);
                } else if (kernel == 1) {
                    launch(0, a, t5_codes[copy % t5_copies], t5_scales[copy % t5_copies], n, k, t, out);
                } else {
                    const std::uint8_t* codes = t5a_codes[copy % t5_copies];
                    const __half* scales      = t5_scales[copy % t5_copies];
                    switch (t) {
                    case 1: launch_t5a<1>(an.qx, an.group_scale, an.slice_sum, codes, scales, n, k, t, out); break;
                    case 2: launch_t5a<2>(an.qx, an.group_scale, an.slice_sum, codes, scales, n, k, t, out); break;
                    case 3: launch_t5a<3>(an.qx, an.group_scale, an.slice_sum, codes, scales, n, k, t, out); break;
                    case 4: launch_t5a<4>(an.qx, an.group_scale, an.slice_sum, codes, scales, n, k, t, out); break;
                    default: launch_t5a<8>(an.qx, an.group_scale, an.slice_sum, codes, scales, n, k, t, out); break;
                    }
                    CHECK(cudaGetLastError());
                }
            };
            run(0, 0, out2);
            CHECK(cudaDeviceSynchronize());
            std::vector<std::uint16_t> h2(std::size_t(n) * t), h5(std::size_t(n) * t);
            CHECK(cudaMemcpy(h2.data(), out2, h2.size() * 2, cudaMemcpyDeviceToHost));
            double norm = 0.0, max_rel[3] = {0.0, 0.0, 0.0};
            for (std::size_t i = 0; i < h2.size(); ++i) norm = std::max(norm, double(std::fabs(bf16_to_float(h2[i]))));
            for (int kernel = 1; kernel < 3; ++kernel) {
                if (t > 8 && kernel == 1) continue; // the diagnostic GEMM is not exact
                CHECK(cudaMemset(out5, 0, std::size_t(n) * t * 2));
                run(kernel, 0, out5);
                CHECK(cudaDeviceSynchronize());
                CHECK(cudaMemcpy(h5.data(), out5, h5.size() * 2, cudaMemcpyDeviceToHost));
                for (std::size_t i = 0; i < h2.size(); ++i) {
                    max_rel[kernel] = std::max(max_rel[kernel],
                                               std::fabs(double(bf16_to_float(h2[i])) - bf16_to_float(h5[i])) / norm);
                }
            }
            double us[3];
            for (int kernel = 0; kernel < 3; ++kernel) {
                std::vector<double> samples;
                for (int repeat = 0; repeat < kRepeats; ++repeat) {
                    CHECK(cudaEventRecord(start));
                    for (int i = 0; i < kIterations; ++i) run(kernel, i, kernel ? out5 : out2);
                    CHECK(cudaEventRecord(stop));
                    CHECK(cudaEventSynchronize(stop));
                    float ms = 0;
                    CHECK(cudaEventElapsedTime(&ms, start, stop));
                    samples.push_back(1000.0 * ms / kIterations);
                }
                std::sort(samples.begin(), samples.end());
                us[kernel] = samples[samples.size() / 2];
            }
            std::printf("%-12s %6d %6d %3d %8.1f %8.1f %8.3f %8.1f %8.3f %8.3f %10.2e %10.2e\n",
                        shape.role, n, k, t, us[0], us[1], us[1] / us[0], us[2], us[2] / us[0],
                        double(t5_weight) / double(t2_weight), max_rel[1], max_rel[2]);
            CHECK(cudaFree(dx));
            for (Activation* act : {&a, &an}) {
                CHECK(cudaFree(act->qx));
                CHECK(cudaFree(act->group_scale));
                CHECK(cudaFree(act->group_sum));
                CHECK(cudaFree(act->slice_sum));
            }
            CHECK(cudaFree(out2));
            CHECK(cudaFree(out5));
        }
        for (auto* p : t2_codes) CHECK(cudaFree(p));
        for (auto* p : t2_scales) CHECK(cudaFree(p));
        for (auto* p : t5_codes) CHECK(cudaFree(p));
        for (auto* p : t5a_codes) CHECK(cudaFree(p));
        for (auto* p : t5_scales) CHECK(cudaFree(p));
    }
    return 0;
}
