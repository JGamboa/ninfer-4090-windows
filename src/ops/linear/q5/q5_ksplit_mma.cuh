#pragma once

// Q5_G64_FP16 small-T tensor-core projection (T = 7..16): the q4_ksplit_mma schedule with the
// Q5 high-bit plane. A CTA owns 16 output rows; its eight warps split each 512-column group into
// 64-column slices, decode their row codes to BF16 A fragments (exact integers -16..15) and
// multiply them with m16n8k16 BF16 MMAs against up to 16 staged activation columns. Each 64-column
// slice is one scale group, so the scale is applied to the slice's FP32 MMA sum; the eight
// slices then reduce in shared memory. The residual epilogue adds the BF16 residual in FP32 and
// rounds once.
//
// Row storage (q5_rowsplit_storage.cuh): per 64-weight group, 32 code bytes (weight 2 i in the
// low nibble of byte i, 2 i + 1 in the high nibble), 8 high-bit bytes (weight w at bit w & 7 of
// byte w >> 3) and one FP16 scale; groups of a row are contiguous, so a row's codes, high bits
// and scales are three contiguous byte runs of K / 2, K / 8 and K / 32 bytes.

#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct Q5KSplitMmaSchedule {
    static constexpr int kKWarps            = 8;
    static constexpr int kThreads           = kKWarps * 32;
    static constexpr int kTileKPerWarp      = 64;
    static constexpr int kGroupK            = kKWarps * kTileKPerWarp;
    static constexpr int kRowsPerCta        = 16;
    static constexpr int kRowsPerLoaderWarp = kRowsPerCta / kKWarps;
};

__device__ __forceinline__ int q5_ksplit_swizzle_64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

// Weights 2 j and 2 j + 1 of a slice: code byte j and their two high bits -> one BF16 pair.
__device__ __forceinline__ unsigned q5_ksplit_bf16_pair(std::uint8_t packed, unsigned high_bits) {
    const int q0 = ((static_cast<int>(packed & 0x0fu) | static_cast<int>((high_bits & 1u) << 4)) ^
                    0x10) -
                   0x10;
    const int q1 = ((static_cast<int>(packed >> 4) | static_cast<int>((high_bits & 2u) << 3)) ^
                    0x10) -
                   0x10;
    const __nv_bfloat162 pair = __floats2bfloat162_rn(static_cast<float>(q0), static_cast<float>(q1));
    return *reinterpret_cast<const unsigned*>(&pair);
}

// out = W x (Residual = false) or out += W x (Residual = true), for `columns` <= TileCols
// activation columns (TileCols - 8 < columns). x is [K, columns] BF16, out [OutputRows, columns].
template <int OutputRows, int InputRows, int TileCols, bool Residual>
__launch_bounds__(Q5KSplitMmaSchedule::kThreads) __global__
    void q5_ksplit_mma_kernel(const __nv_bfloat16* __restrict__ x,
                              const std::uint8_t* __restrict__ codes,
                              const std::uint8_t* __restrict__ high,
                              const std::uint8_t* __restrict__ scales,
                              __nv_bfloat16* __restrict__ out, int columns) {
    using Schedule              = Q5KSplitMmaSchedule;
    constexpr int kHidden       = InputRows;
    constexpr int kTileK        = Schedule::kTileKPerWarp;
    constexpr int kWarps        = Schedule::kKWarps;
    constexpr int kRowsPerCta   = Schedule::kRowsPerCta;
    constexpr int kGroupK       = Schedule::kGroupK;
    constexpr int kGroups       = kHidden / kGroupK;
    constexpr int kCodeRowBytes = kHidden / 2;
    constexpr int kHighRowBytes = kHidden / 8;
    constexpr int kScalesPerRow = kHidden / 64;
    constexpr int kNt           = TileCols / 8;
    static_assert(TileCols == 8 || TileCols == 16);
    static_assert(kHidden % kGroupK == 0 && OutputRows % kRowsPerCta == 0);

    union SharedStorage {
        struct {
            std::uint8_t codes[kRowsPerCta][kGroupK / 2];
            std::uint8_t high[kRowsPerCta][kGroupK / 8];
            __nv_bfloat16 activations[kWarps][TileCols * kTileK];
            std::uint16_t scales[kRowsPerCta][kWarps];
        } staging;

        float partial[kWarps * kNt * 32 * 4];
    };

    __shared__ __align__(16) SharedStorage shared;
    auto& code_shared  = shared.staging.codes;
    auto& high_shared  = shared.staging.high;
    auto& x_shared     = shared.staging.activations;
    auto& scale_shared = shared.staging.scales;

    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int gid     = lane >> 2;
    const int lid     = lane & 3;
    const int k_split = warp;
    const int row0    = static_cast<int>(blockIdx.x) * kRowsPerCta;

    // Each warp stages its own 64-column slice of every live activation column (dead columns of
    // the tile are zero-filled, so their MMA outputs are finite and never stored).
    const auto stage_x = [&](int group_k0) {
        constexpr int kItems = TileCols * (kTileK / 8);
        for (int item = lane; item < kItems; item += 32) {
            const int col    = item / (kTileK / 8);
            const int k8     = item - col * (kTileK / 8);
            const int source = col < columns ? col : 0;
            cp_async_zfill<16>(&x_shared[warp][col * kTileK + q5_ksplit_swizzle_64(col, k8 * 8)],
                               &x[static_cast<std::int64_t>(source) * kHidden + group_k0 +
                                  warp * kTileK + k8 * 8],
                               col < columns ? 16 : 0);
        }
    };

    const auto stage_weight = [&](int group_k0) {
#pragma unroll
        for (int row_item = 0; row_item < Schedule::kRowsPerLoaderWarp; ++row_item) {
            const int row                   = warp * Schedule::kRowsPerLoaderWarp + row_item;
            const std::int64_t weight_row   = row0 + row;
            // 256 code bytes (16 chunks) and 64 high-bit bytes (4 chunks) per row and group.
            if (lane < kGroupK / 32) {
                cp_async<16, Cache::cg>(&code_shared[row][lane * 16],
                                        codes + weight_row * kCodeRowBytes + group_k0 / 2 +
                                            lane * 16);
            } else if (lane < kGroupK / 32 + kGroupK / 128) {
                const int chunk = lane - kGroupK / 32;
                cp_async<16, Cache::cg>(&high_shared[row][chunk * 16],
                                        high + weight_row * kHighRowBytes + group_k0 / 8 +
                                            chunk * 16);
            }
        }
        for (int row = tid; row < kRowsPerCta; row += kWarps * 32) {
            cp_async<16>(&scale_shared[row][0],
                         scales + (static_cast<std::int64_t>(row0 + row) * kScalesPerRow +
                                   group_k0 / 64) *
                                      2);
        }
    };

    const int b_rin     = lane & 7;
    const int b_koff    = ((lane >> 3) & 1) << 3;
    const int warp_koff = k_split * kTileK;
    float acc[kNt][4]   = {};

    stage_weight(0);
    stage_x(0);
    cp_commit();
    cp_wait<0>();
    __syncthreads();

#pragma unroll 1
    for (int group_index = 0; group_index < kGroups; ++group_index) {
        const int group_k0      = group_index * kGroupK;
        float group_acc[kNt][4] = {};
        // The slice's 64 high bits: byte 2 ks holds weights ks*16 .. ks*16+7, byte 2 ks + 1 the
        // next eight; A fragment register a0 needs weights ks*16 + 2 lid (+1), a2 the same + 8.
        const std::uint8_t* top_high = &high_shared[gid][warp_koff / 8];
        const std::uint8_t* bot_high = &high_shared[gid + 8][warp_koff / 8];

#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const int byte_col = warp_koff / 2 + ks * 8 + lid;
            const int shift    = 2 * lid;
            const unsigned af0 = q5_ksplit_bf16_pair(code_shared[gid][byte_col],
                                                     top_high[2 * ks] >> shift);
            const unsigned af1 = q5_ksplit_bf16_pair(code_shared[gid + 8][byte_col],
                                                     bot_high[2 * ks] >> shift);
            const unsigned af2 = q5_ksplit_bf16_pair(code_shared[gid][byte_col + 4],
                                                     top_high[2 * ks + 1] >> shift);
            const unsigned af3 = q5_ksplit_bf16_pair(code_shared[gid + 8][byte_col + 4],
                                                     bot_high[2 * ks + 1] >> shift);
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                unsigned bf0, bf1;
                const int br = nt * 8 + b_rin;
                ldmatrix_x2(bf0, bf1,
                            smem_addr(&x_shared[k_split][br * kTileK + q5_ksplit_swizzle_64(
                                                                           br, ks * 16 + b_koff)]));
                mma_bf16(group_acc[nt][0], group_acc[nt][1], group_acc[nt][2], group_acc[nt][3],
                         af0, af1, af2, af3, bf0, bf1);
            }
        }

        const float top_scale = __half2float(__ushort_as_half(scale_shared[gid][k_split]));
        const float bot_scale = __half2float(__ushort_as_half(scale_shared[gid + 8][k_split]));
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            acc[nt][0] = fmaf(group_acc[nt][0], top_scale, acc[nt][0]);
            acc[nt][1] = fmaf(group_acc[nt][1], top_scale, acc[nt][1]);
            acc[nt][2] = fmaf(group_acc[nt][2], bot_scale, acc[nt][2]);
            acc[nt][3] = fmaf(group_acc[nt][3], bot_scale, acc[nt][3]);
        }

        if (group_index + 1 < kGroups) {
            __syncthreads();
            stage_weight(group_k0 + kGroupK);
            stage_x(group_k0 + kGroupK);
            cp_commit();
            cp_wait<0>();
            __syncthreads();
        }
    }

    // Reduce the eight K slices: odd warps publish, even warps fold their partner, warp 0 sums.
    __syncthreads();
    auto* partial = shared.partial;
    if ((k_split & 1) != 0) {
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            store_vec(partial + ((k_split * kNt + nt) * 32 + lane) * 4,
                      make_float4(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]));
        }
    }
    __syncthreads();
    if ((k_split & 1) == 0) {
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            const float4 partner =
                load_vec<float4>(partial + (((k_split + 1) * kNt + nt) * 32 + lane) * 4);
            acc[nt][0] += partner.x;
            acc[nt][1] += partner.y;
            acc[nt][2] += partner.z;
            acc[nt][3] += partner.w;
            if (k_split != 0) {
                store_vec(partial + ((k_split * kNt + nt) * 32 + lane) * 4,
                          make_float4(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]));
            }
        }
    }
    __syncthreads();

    if (k_split == 0) {
        const auto store = [&](int row, int col, float value) {
            if (col >= columns) return;
            __nv_bfloat16* destination = out + static_cast<std::int64_t>(col) * OutputRows + row;
            if constexpr (Residual) value += __bfloat162float(*destination);
            *destination = __float2bfloat16_rn(value);
        };
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            float4 sum = make_float4(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]);
#pragma unroll
            for (int split = 2; split < kWarps; split += 2) {
                const float4 value =
                    load_vec<float4>(partial + ((split * kNt + nt) * 32 + lane) * 4);
                sum.x += value.x;
                sum.y += value.y;
                sum.z += value.z;
                sum.w += value.w;
            }
            const int col0 = nt * 8 + 2 * lid;
            store(row0 + gid, col0, sum.x);
            store(row0 + gid + 8, col0, sum.z);
            store(row0 + gid, col0 + 1, sum.y);
            store(row0 + gid + 8, col0 + 1, sum.w);
        }
    }
}

} // namespace ninfer::ops::detail
