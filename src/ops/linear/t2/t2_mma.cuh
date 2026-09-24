#pragma once

// Tensor-core t2 projection for two or more tokens (MTP verification, prefill tiles).
//
// A CTA owns 16 parent rows and eight K-split warps. Each iteration stages 1024 columns: warp w
// multiplies columns [128 w, 128 w + 128) of the iteration, exactly one FP16 scale group, with
// eight m16n8k16 BF16 MMAs. Ternary codes become BF16 {-1, 0, +1} A fragments through a
// 16-entry table (one nibble = the two codes of one fragment register); x reaches the B
// fragments through ldmatrix. Products of the exact weights and BF16 x accumulate in FP32; each
// group sum is scaled once. Codes, scales and x are double-buffered with cp.async.

#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail::t2_mma {

constexpr int kWarps      = 8;
constexpr int kThreads    = kWarps * 32;
constexpr int kRows       = 16;              // output rows per CTA (the MMA M)
constexpr int kTokens     = 8;               // token columns per CTA (the MMA N)
constexpr int kWarpK      = 128;             // one scale group per warp and iteration
constexpr int kStepK      = kWarps * kWarpK; // columns per iteration
constexpr int kStepBytes  = kStepK / 4;      // code bytes of one row per iteration
constexpr int kMaxOutputs = 4;

struct Outputs {
    __nv_bfloat16* data[kMaxOutputs];
    int end[kMaxOutputs]; // exclusive parent row bound of each output
};

// 16-byte chunk swizzle of a 128-column BF16 row: spreads the eight token rows over banks.
__device__ __forceinline__ int swizzle_128(int row, int column) {
    return (((column >> 3) ^ (row & 7)) << 3) | (column & 7);
}

struct Stage {
    std::uint8_t codes[kRows][kStepBytes];                // 4 KiB
    __nv_bfloat16 x[kWarps][kTokens * kWarpK];            // 16 KiB
    __half scales[kRows][kWarps];                         // 256 B
};

__global__ void __launch_bounds__(kThreads)
    t2_mma_kernel(const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
                  const __half* __restrict__ scales, std::int64_t scale_row_halves, int k,
                  int tokens, Outputs outputs, bool accumulate) {
    __shared__ __align__(128) Stage stages[2];
    __shared__ unsigned pair_table[16];
    __shared__ __align__(16) float partial[kWarps][32][4];

    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int gid    = lane >> 2;
    const int lid    = lane & 3;
    const int row0   = static_cast<int>(blockIdx.x) * kRows;
    const int token0 = static_cast<int>(blockIdx.y) * kTokens;
    const int live   = min(kTokens, tokens - token0);
    const int steps  = k / kStepK;
    const std::int64_t code_row_bytes = k / 4;

    if (tid < 16) {
        // Nibble n holds codes c0 = n & 3 (lower column) and c1 = n >> 2; BF16 of c - 1.
        // Code 3 is invalid in stored data; it maps to +2 instead of branching.
        const unsigned short value[4] = {0xbf80, 0x0000, 0x3f80, 0x4000};
        pair_table[tid] = value[tid & 3] | (unsigned(value[tid >> 2]) << 16);
    }

    const auto stage = [&](int step, Stage& s) {
        // Codes: 16 rows x 256 bytes, one 16-byte copy per thread.
        {
            const int row   = tid >> 4;
            const int chunk = tid & 15;
            cp_async<16, Cache::cg>(&s.codes[row][chunk * 16],
                                    codes + std::int64_t(row0 + row) * code_row_bytes +
                                        std::int64_t(step) * kStepBytes + chunk * 16);
        }
        // Scales: the eight groups of this step for each of the 16 rows.
        if (tid < kRows) {
            cp_async<16, Cache::cg>(&s.scales[tid][0],
                                    scales + std::int64_t(row0 + tid) * scale_row_halves +
                                        std::int64_t(step) * kWarps);
        }
        // x: each warp stages its 128 columns for the live tokens (zero-filled beyond T).
        for (int item = lane; item < kTokens * (kWarpK / 8); item += 32) {
            const int token  = item / (kWarpK / 8);
            const int chunk  = item % (kWarpK / 8);
            const int source = token < live ? token0 + token : token0;
            cp_async_zfill<16>(&s.x[warp][token * kWarpK + swizzle_128(token, chunk * 8)],
                               x + std::int64_t(source) * k + std::int64_t(step) * kStepK +
                                   warp * kWarpK + chunk * 8,
                               token < live ? 16 : 0);
        }
    };

    float acc[4] = {};
    stage(0, stages[0]);
    cp_commit();

    for (int step = 0; step < steps; ++step) {
        if (step + 1 < steps) { stage(step + 1, stages[(step + 1) & 1]); }
        cp_commit();
        cp_wait<1>();
        __syncthreads();

        const Stage& s = stages[step & 1];
        float group[4] = {};
#pragma unroll
        for (int ks = 0; ks < kWarpK / 16; ++ks) {
            // Columns lid*2, lid*2+1 of this 16-column step live in one nibble of byte
            // (ks*16 + lid*2) / 4; columns +8 are two bytes later.
            const int byte  = warp * (kWarpK / 4) + ks * 4 + (lid >> 1);
            const int shift = (lid & 1) * 4;
            const unsigned a0 = pair_table[(s.codes[gid][byte] >> shift) & 15];
            const unsigned a1 = pair_table[(s.codes[gid + 8][byte] >> shift) & 15];
            const unsigned a2 = pair_table[(s.codes[gid][byte + 2] >> shift) & 15];
            const unsigned a3 = pair_table[(s.codes[gid + 8][byte + 2] >> shift) & 15];
            unsigned b0, b1;
            const int b_row    = lane & 7;
            const int b_column = ks * 16 + (((lane >> 3) & 1) << 3);
            ldmatrix_x2(b0, b1,
                        smem_addr(&s.x[warp][b_row * kWarpK + swizzle_128(b_row, b_column)]));
            mma_bf16(group[0], group[1], group[2], group[3], a0, a1, a2, a3, b0, b1);
        }
        const float top    = __half2float(s.scales[gid][warp]);
        const float bottom = __half2float(s.scales[gid + 8][warp]);
        acc[0]             = fmaf(group[0], top, acc[0]);
        acc[1]             = fmaf(group[1], top, acc[1]);
        acc[2]             = fmaf(group[2], bottom, acc[2]);
        acc[3]             = fmaf(group[3], bottom, acc[3]);
        __syncthreads(); // the buffer is refilled by the next iteration's stage
    }

    store_vec(&partial[warp][lane][0], make_float4(acc[0], acc[1], acc[2], acc[3]));
    __syncthreads();
    if (warp != 0) { return; }
    float sum[4] = {};
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
        const float4 value = load_vec<float4>(&partial[w][lane][0]);
        sum[0] += value.x;
        sum[1] += value.y;
        sum[2] += value.z;
        sum[3] += value.w;
    }
    // Fragment C: sum[0], sum[1] are row gid, tokens 2*lid and 2*lid+1; sum[2], sum[3] row gid+8.
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int row = row0 + gid + half * 8;
        int s         = 0;
        while (row >= outputs.end[s]) ++s;
        const int begin = s ? outputs.end[s - 1] : 0;
        const int rows  = outputs.end[s] - begin;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int token = 2 * lid + j;
            if (token < live) {
                auto* out = outputs.data[s] + std::int64_t(token0 + token) * rows + (row - begin);
                const float value = sum[half * 2 + j];
                *out = __float2bfloat16_rn(accumulate ? value + __bfloat162float(*out) : value);
            }
        }
    }
}

} // namespace ninfer::ops::detail::t2_mma
