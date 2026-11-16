#pragma once

// Q5_G64_FP16 small-T tensor-core projection (T = 1..16). Adapted from iamwavecut/ninfer-all
// (Apache-2.0), src/ops/linear/q5/q5_small_t_mma.cuh, specialized to the eight-K-warp layout.
//
// For narrow extents -- a DFlash2 verification is T = K + 1 = 7..16 -- the SIMT kernels grow with T
// because every column is its own FMA stream, while an MMA over an eight-column tile costs the
// same from T = 1 to T = 8.
//
// A CTA owns 16 weight rows (one m16 tile) and splits each 512-wide K slab over its eight warps,
// one 64-k quantization group per warp. Nibbles, high bits and scales are staged with cp.async;
// each lane decodes its eight contiguous code bytes per row straight into A fragments (exactly: a
// Q5 code is an integer in [-16, 15]) and multiplies them against the staged activations with
// bf16 mma.sync in a permuted k order (see the kernel); each warp folds its group's fp16 scale
// into an fp32 accumulator, and the eight K partials are reduced through shared memory at the end.
//
// Row storage (q5_rowsplit_storage.cuh): per 64-weight group, 32 code bytes (weight 2 i in the low
// nibble of byte i, 2 i + 1 in the high nibble), 8 high-bit bytes (weight w at bit w & 7 of byte
// w >> 3) and one FP16 scale; groups of a row are contiguous, so a row's codes, high bits and
// scales are three contiguous byte runs of K / 2, K / 8 and K / 32 bytes.

#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct Q5SmallTSchedule {
    static constexpr int kWarps        = 8; // K warps of a CTA, one 64-k group apiece
    static constexpr int kThreads      = kWarps * 32;
    static constexpr int kRowsPerCta   = 16;
    static constexpr int kTileK        = 64;
    static constexpr int kGroupK       = kWarps * kTileK; // one slab: 512 k
    static constexpr int kCodeRowBytes = kGroupK / 2;
    static constexpr int kHighRowBytes = kGroupK / 8;
    // A lane reads eight contiguous code bytes per row (LDS.64) and two high bytes (LDS.16); a
    // half-warp of 64-bit loads spans four rows, so code rows are staggered by 32 B (8 banks) and
    // the eight rows of a warp's high-bit loads by 16 B, which puts every row a warp touches in
    // its own banks.
    static constexpr int kCodeRowPad = 32;
    static constexpr int kHighRowPad = 16;
    // Activation columns are 64 bf16 (128 B) per K warp; 16 B of padding staggers adjacent
    // columns by four banks for the quarter-warp 128-bit B loads.
    static constexpr int kXColPad = 8;
};

// Shared memory of q5_small_t_mma_kernel: a ring of Stages slabs, reused for the final K
// reduction. A ring of two stages at 16 columns is exactly the 48 KB static limit.
template <int XCols, int Stages>
struct Q5SmallTStorage {
    using Schedule = Q5SmallTSchedule;
    struct Stage {
        std::uint8_t codes[Schedule::kRowsPerCta][Schedule::kCodeRowBytes + Schedule::kCodeRowPad];
        std::uint8_t high[Schedule::kRowsPerCta][Schedule::kHighRowBytes + Schedule::kHighRowPad];
        __nv_bfloat16 activations[Schedule::kWarps][XCols][64 + Schedule::kXColPad];
        std::uint16_t scales[Schedule::kRowsPerCta][Schedule::kWarps];
    };
    union Shared {
        Stage stages[Stages];
        float partial[Schedule::kWarps * (XCols < 8 ? 1 : XCols / 8) * 32 * 4];
    };
    static constexpr int kBytes = sizeof(Shared);
};

// Eight Q5 codes -> four bf16 pairs, exactly, without int-to-float conversion. `word` holds four
// code bytes (weight 2j in byte j's low nibble, 2j + 1 in its high nibble) and `high` their eight
// high bits (bit i for weight i). A code is the five-bit two's-complement value nib - 16 * h; with
// the high bit inverted into bit 4 it becomes v = nib + 16 * (1 - h) in [0, 31], which placed in
// the mantissa of bf16 128.0 reads 128 + v, and subtracting 144 leaves the code. out[j] holds
// weights (2j, 2j + 1).
__device__ __forceinline__ void q5_small_t_decode_eight(unsigned word, unsigned high,
                                                        unsigned (&out)[4]) {
    const unsigned kMagic = 0x43004300u; // bf16 128.0 in both halves
    const unsigned kBias  = 0x43104310u; // bf16 144.0 in both halves
    const unsigned lo     = word & 0x0f0f0f0fu;
    const unsigned hi     = (word >> 4) & 0x0f0f0f0fu;
    unsigned even         = __byte_perm(lo, hi, 0x5140); // weights 0..3, one per byte
    unsigned odd          = __byte_perm(lo, hi, 0x7362); // weights 4..7
    const unsigned inv    = ~high;
    // (b & 0xf) * 0x00204081 lays bits 0..3 at 0, 8, 16, 24 with no carries between the copies.
    even |= (((inv & 0xfu) * 0x00204081u) & 0x01010101u) << 4;
    odd |= ((((inv >> 4) & 0xfu) * 0x00204081u) & 0x01010101u) << 4;
    const unsigned biased[4] = {
        __byte_perm(even, kMagic, 0x7150), __byte_perm(even, kMagic, 0x7372),
        __byte_perm(odd, kMagic, 0x7150), __byte_perm(odd, kMagic, 0x7372)};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const __nv_bfloat162 value = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&biased[j]),
                                             *reinterpret_cast<const __nv_bfloat162*>(&kBias));
        out[j]                     = *reinterpret_cast<const unsigned*>(&value);
    }
}

// Rows: output rows, a multiple of 16; the grid is Rows / 16. K: reduction length, a multiple of
// the 512-k slab. XCols: activation columns staged (4, 8 or 16), covered by max(1, XCols / 8)
// eight-column MMA tiles; the B fragments of columns at or past `columns` are zero. Stages: depth
// of the cp.async ring (1..2); a CTA keeps at most four resident (two with a 48 KB ring).
// Epilogue::store(row, col0, v) receives, for the lane's rows row and row + 8, v.x = (row, col0),
// v.y = (row, col0 + 1), v.z = (row + 8, col0) and v.w = (row + 8, col0 + 1); columns at or past
// `columns` must be ignored by the epilogue. x is [K, columns] BF16, columns contiguous.
//
// K order. The MMA's k slots may be any permutation of a group's 64 k, provided A and B agree.
// Lane lid is given k 16*lid .. 16*lid + 15 of its warp's group: for step ks, slots (2*lid,
// 2*lid + 1) carry k 16*lid + 4*ks + {0, 1} and slots (2*lid + 8, 2*lid + 9) carry
// 16*lid + 4*ks + {2, 3}. The lane's A operand over the four steps is then eight contiguous code
// bytes per row -- one 64-bit load, decoded by q5_small_t_decode_eight -- and its B operand
// sixteen contiguous activations of one column -- two 128-bit loads, no ldmatrix -- where the
// natural slot order needs a byte load, a decode and an ldmatrix per fragment.
template <int Rows, int K, int XCols, int Stages, class Epilogue>
__launch_bounds__(Q5SmallTSchedule::kThreads, XCols >= 16 || Stages == 2 ? 4 : 6) __global__
    void q5_small_t_mma_kernel(const __nv_bfloat16* __restrict__ x,
                               const std::uint8_t* __restrict__ codes,
                               const std::uint8_t* __restrict__ high_bits,
                               const std::uint8_t* __restrict__ scales, Epilogue epilogue,
                               int columns) {
    using Schedule               = Q5SmallTSchedule;
    constexpr int kWarps         = Schedule::kWarps;
    constexpr int kTileK         = Schedule::kTileK;
    constexpr int kGroupK        = Schedule::kGroupK;
    constexpr int kSlabs         = K / kGroupK;
    constexpr int kGroupsPerRow  = K / 64;
    constexpr int kNt            = XCols < 8 ? 1 : XCols / 8;
    static_assert(XCols == 4 || XCols == 8 || XCols == 16);
    static_assert(Stages >= 1 && Stages <= 2);
    static_assert(K % kGroupK == 0, "K must be a whole number of slabs");
    static_assert(Rows % Schedule::kRowsPerCta == 0);

    using Storage = Q5SmallTStorage<XCols, Stages>;
    using Stage   = typename Storage::Stage;
    static_assert(Storage::kBytes <= 48 * 1024, "the ring must fit static shared memory");
    __shared__ __align__(16) typename Storage::Shared shared;

    const int tid      = static_cast<int>(threadIdx.x);
    const int warp     = tid >> 5; // the K warp
    const int lane     = tid & 31;
    const int gid      = lane >> 2;
    const int lid      = lane & 3;
    const int row0     = static_cast<int>(blockIdx.x) * Schedule::kRowsPerCta;

    const auto stage_x = [&](int slab_k0, Stage& stage) {
        constexpr int kChunksPerCol = kGroupK / 8;
        const int items             = columns * kChunksPerCol;
        for (int item = tid; item < items; item += Schedule::kThreads) {
            const int col   = item / kChunksPerCol;
            const int chunk = item - col * kChunksPerCol;
            cp_async<16>(&stage.activations[chunk / 8][col][(chunk % 8) * 8],
                         &x[static_cast<std::int64_t>(col) * K + slab_k0 + chunk * 8]);
        }
    };

    // Per row: the code chunks, then the high-bit chunks, then one scale copy.
    const auto stage_weight = [&](int slab_k0, Stage& stage) {
        constexpr int kCodeChunks  = Schedule::kCodeRowBytes / 16;
        constexpr int kHighChunks  = Schedule::kHighRowBytes / 16;
        constexpr int kItemsPerRow = kCodeChunks + kHighChunks + 1;
        for (int item = tid; item < Schedule::kRowsPerCta * kItemsPerRow;
             item += Schedule::kThreads) {
            const int row        = item / kItemsPerRow;
            const int j          = item - row * kItemsPerRow;
            const std::int64_t r = row0 + row;
            if (j < kCodeChunks) {
                cp_async<16, Cache::cg>(&stage.codes[row][j * 16],
                                        codes + r * (K / 2) + slab_k0 / 2 + j * 16);
            } else if (j < kCodeChunks + kHighChunks) {
                const int chunk = j - kCodeChunks;
                cp_async<16, Cache::cg>(&stage.high[row][chunk * 16],
                                        high_bits + r * (K / 8) + slab_k0 / 8 + chunk * 16);
            } else {
                cp_async<2 * kWarps>(&stage.scales[row][0],
                                     scales + (r * kGroupsPerRow + slab_k0 / 64) * 2);
            }
        }
    };

    const auto issue_slab = [&](int slab) {
        if (slab < kSlabs) {
            Stage& stage = shared.stages[slab % Stages];
            stage_weight(slab * kGroupK, stage);
            stage_x(slab * kGroupK, stage);
        }
        cp_commit(); // empty commits keep cp_wait<Stages - 1> exact through the tail
    };

    const int code_byte = warp * (kTileK / 2) + 8 * lid;
    const int high_byte = warp * (kTileK / 8) + 2 * lid;
    float acc[kNt][4]   = {};

#pragma unroll
    for (int prefetch = 0; prefetch < Stages - 1; ++prefetch) { issue_slab(prefetch); }

#pragma unroll 1
    for (int slab = 0; slab < kSlabs; ++slab) {
        issue_slab(slab + Stages - 1);
        cp_wait<Stages - 1>();
        __syncthreads();

        const Stage& stage = shared.stages[slab % Stages];
        const uint2 top    = load_vec<uint2>(&stage.codes[gid][code_byte]);
        const uint2 bot    = load_vec<uint2>(&stage.codes[gid + 8][code_byte]);
        const unsigned top_high = *reinterpret_cast<const std::uint16_t*>(&stage.high[gid][high_byte]);
        const unsigned bot_high =
            *reinterpret_cast<const std::uint16_t*>(&stage.high[gid + 8][high_byte]);

        float group_acc[kNt][4] = {};
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            // Four code bytes and one high byte per row cover steps ks = 2*half and 2*half + 1.
            unsigned a_top[4], a_bot[4];
            q5_small_t_decode_eight(half == 0 ? top.x : top.y, (top_high >> (8 * half)) & 0xffu,
                                    a_top);
            q5_small_t_decode_eight(half == 0 ? bot.x : bot.y, (bot_high >> (8 * half)) & 0xffu,
                                    a_bot);
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                const int col = nt * 8 + gid;
                uint4 b       = make_uint4(0u, 0u, 0u, 0u);
                if (col < columns) {
                    b = load_vec<uint4>(&stage.activations[warp][col][16 * lid + 8 * half]);
                }
                float(&g)[4] = group_acc[nt];
                mma_bf16(g[0], g[1], g[2], g[3], a_top[0], a_bot[0], a_top[1], a_bot[1], b.x, b.y);
                mma_bf16(g[0], g[1], g[2], g[3], a_top[2], a_bot[2], a_top[3], a_bot[3], b.z, b.w);
            }
        }

        const float top_scale = __half2float(__ushort_as_half(stage.scales[gid][warp]));
        const float bot_scale = __half2float(__ushort_as_half(stage.scales[gid + 8][warp]));
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            acc[nt][0] = fmaf(group_acc[nt][0], top_scale, acc[nt][0]);
            acc[nt][1] = fmaf(group_acc[nt][1], top_scale, acc[nt][1]);
            acc[nt][2] = fmaf(group_acc[nt][2], bot_scale, acc[nt][2]);
            acc[nt][3] = fmaf(group_acc[nt][3], bot_scale, acc[nt][3]);
        }
        // The next iteration's issue_slab writes the stage this one just read.
        __syncthreads();
    }
    cp_wait<0>();

    // Reduce the eight K partials: odd warps publish, even ones fold their neighbour, then warp 0
    // sums the even ones.
    __syncthreads();
    auto* partial   = shared.partial;
    const auto slot = [&](int w, int nt) { return partial + ((w * kNt + nt) * 32 + lane) * 4; };
    if ((warp & 1) != 0) {
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            store_vec(slot(warp, nt), make_float4(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]));
        }
    }
    __syncthreads();
    if ((warp & 1) == 0) {
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            const float4 partner = load_vec<float4>(slot(warp + 1, nt));
            acc[nt][0] += partner.x;
            acc[nt][1] += partner.y;
            acc[nt][2] += partner.z;
            acc[nt][3] += partner.w;
            if (warp != 0) {
                store_vec(slot(warp, nt),
                          make_float4(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]));
            }
        }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            float4 sum = make_float4(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]);
#pragma unroll
            for (int split = 2; split < kWarps; split += 2) {
                const float4 value = load_vec<float4>(slot(split, nt));
                sum.x += value.x;
                sum.y += value.y;
                sum.z += value.z;
                sum.w += value.w;
            }
            epilogue.store(row0 + gid, nt * 8 + 2 * lid, sum);
        }
    }
}

// Launches q5_small_t_mma_kernel over Rows / 16 CTAs.
template <int Rows, int K, int XCols, int Stages, class Epilogue>
void q5_small_t_mma_launch(cudaStream_t stream, const __nv_bfloat16* x, const std::uint8_t* codes,
                           const std::uint8_t* high_bits, const std::uint8_t* scales,
                           Epilogue epilogue, int columns) {
    q5_small_t_mma_kernel<Rows, K, XCols, Stages, Epilogue>
        <<<Rows / Q5SmallTSchedule::kRowsPerCta, Q5SmallTSchedule::kThreads, 0, stream>>>(
            x, codes, high_bits, scales, epilogue, columns);
}

} // namespace ninfer::ops::detail
