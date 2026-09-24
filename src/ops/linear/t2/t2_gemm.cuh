#pragma once

// Tensor-core t2 GEMM for prefill-sized token counts.
//
// A CTA of four warps owns a 64-row x 64-token output tile; warp (wm, wn) computes rows
// [32 wm, 32 wm + 32) x tokens [32 wn, 32 wn + 32) with 2 x 4 m16n8k16 BF16 MMAs per 16-column
// step. Each K stage is one 128-column FP16 scale group: codes (64 rows x 32 bytes) and x
// (64 tokens x 128 columns) are double-buffered with cp.async, ternary codes become BF16
// {-1, 0, +1} A fragments through a 16-entry nibble table (each fragment reused by four token
// tiles), and every group sum is scaled once into the FP32 accumulator. Weights are read
// once per 64 tokens instead of once per 8.

#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail::t2_gemm {

constexpr int kWarps      = 4;
constexpr int kThreads    = kWarps * 32;
constexpr int kRows       = 64; // output rows per CTA
constexpr int kTokens     = 64; // tokens per CTA
constexpr int kStageK     = 128;
constexpr int kStageBytes = kStageK / 4;
constexpr int kMaxOutputs = 4;

struct Outputs {
    __nv_bfloat16* data[kMaxOutputs];
    int end[kMaxOutputs]; // exclusive parent row bound of each output
};

__device__ __forceinline__ int swizzle_128(int row, int column) {
    return (((column >> 3) ^ (row & 7)) << 3) | (column & 7);
}

struct Stage {
    std::uint8_t codes[kRows][kStageBytes];   // 2 KiB
    __nv_bfloat16 x[kTokens * kStageK];       // 16 KiB
};

__global__ void __launch_bounds__(kThreads)
    t2_gemm_kernel(const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
                   const __half* __restrict__ scales, std::int64_t scale_row_halves, int k,
                   int tokens, Outputs outputs, bool accumulate) {
    __shared__ __align__(128) Stage stages[2];
    __shared__ unsigned pair_table[16];

    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int gid    = lane >> 2;
    const int lid    = lane & 3;
    const int wm     = warp & 1;
    const int wn     = warp >> 1;
    const int row0   = static_cast<int>(blockIdx.x) * kRows;
    const int token0 = static_cast<int>(blockIdx.y) * kTokens;
    const int live   = min(kTokens, tokens - token0);
    const int steps  = k / kStageK;
    const std::int64_t code_row_bytes = k / 4;

    if (tid < 16) {
        // Nibble n holds codes c0 = n & 3 (lower column) and c1 = n >> 2; BF16 of c - 1.
        const unsigned short value[4] = {0xbf80, 0x0000, 0x3f80, 0x4000};
        pair_table[tid] = value[tid & 3] | (unsigned(value[tid >> 2]) << 16);
    }

    const auto stage = [&](int step, Stage& s) {
        // Codes: 64 rows x 32 bytes, 16 bytes per thread.
        {
            const int row  = tid >> 1;
            const int half = tid & 1;
            cp_async<16, Cache::cg>(&s.codes[row][half * 16],
                                    codes + std::int64_t(row0 + row) * code_row_bytes +
                                        std::int64_t(step) * kStageBytes + half * 16);
        }
        // x: 64 tokens x 128 columns, zero-filled past the live tokens.
        for (int item = tid; item < kTokens * (kStageK / 8); item += kThreads) {
            const int token  = item / (kStageK / 8);
            const int chunk  = item % (kStageK / 8);
            const int source = token < live ? token0 + token : token0;
            cp_async_zfill<16>(&s.x[token * kStageK + swizzle_128(token, chunk * 8)],
                               x + std::int64_t(source) * k + std::int64_t(step) * kStageK +
                                   chunk * 8,
                               token < live ? 16 : 0);
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

        const Stage& s       = stages[step & 1];
        float group[2][4][4] = {};
#pragma unroll
        for (int ks = 0; ks < kStageK / 16; ++ks) {
            unsigned b[4][2];
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int b_row    = wn * 32 + nt * 8 + (lane & 7);
                const int b_column = ks * 16 + (((lane >> 3) & 1) << 3);
                ldmatrix_x2(b[nt][0], b[nt][1],
                            smem_addr(&s.x[b_row * kStageK + swizzle_128(b_row, b_column)]));
            }
            const int byte  = ks * 4 + (lid >> 1);
            const int shift = (lid & 1) * 4;
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const int r       = wm * 32 + mt * 16 + gid;
                const unsigned a0 = pair_table[(s.codes[r][byte] >> shift) & 15];
                const unsigned a1 = pair_table[(s.codes[r + 8][byte] >> shift) & 15];
                const unsigned a2 = pair_table[(s.codes[r][byte + 2] >> shift) & 15];
                const unsigned a3 = pair_table[(s.codes[r + 8][byte + 2] >> shift) & 15];
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    mma_bf16(group[mt][nt][0], group[mt][nt][1], group[mt][nt][2],
                             group[mt][nt][3], a0, a1, a2, a3, b[nt][0], b[nt][1]);
                }
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
                acc[mt][nt][0] = fmaf(group[mt][nt][0], top, acc[mt][nt][0]);
                acc[mt][nt][1] = fmaf(group[mt][nt][1], top, acc[mt][nt][1]);
                acc[mt][nt][2] = fmaf(group[mt][nt][2], bottom, acc[mt][nt][2]);
                acc[mt][nt][3] = fmaf(group[mt][nt][3], bottom, acc[mt][nt][3]);
            }
        }
        __syncthreads(); // the buffer is refilled by the next iteration's stage
    }

    // Fragment C: [0], [1] are row gid, tokens 2*lid and 2*lid+1; [2], [3] are row gid+8.
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int row = row0 + wm * 32 + mt * 16 + gid + half * 8;
            int s         = 0;
            while (row >= outputs.end[s]) ++s;
            const int begin = s ? outputs.end[s - 1] : 0;
            const int rows  = outputs.end[s] - begin;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int token = wn * 32 + nt * 8 + 2 * lid + j;
                    if (token < live) {
                        auto* out = outputs.data[s] + std::int64_t(token0 + token) * rows +
                                    (row - begin);
                        const float value = acc[mt][nt][half * 2 + j];
                        *out = __float2bfloat16_rn(accumulate ? value + __bfloat162float(*out)
                                                              : value);
                    }
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail::t2_gemm
