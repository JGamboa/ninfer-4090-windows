#include "ops/linear/t2/t2_project.h"

#include "core/device.h"
#include "ops/linear/t2/t2_mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

constexpr int kThreads     = 256;
constexpr int kWarps       = kThreads / 32;
constexpr int kRowsPerWarp = 2;
constexpr int kMaxTile     = 8; // tokens per launch column; larger T uses grid.y
constexpr int kMaxOutputs  = 4;

struct Outputs {
    __nv_bfloat16* data[kMaxOutputs];
    int end[kMaxOutputs]; // exclusive parent row bound of each output
};

// Each warp owns two parent rows. A lane reads 32 codes (8 bytes) of each row per step, so a
// warp step covers 1024 columns; the codes of both rows are loaded before any arithmetic to
// keep several global loads in flight. The 32 decoded weights (code - 1) are reused for every
// token; x comes from global memory through L1 (the whole activation is shared by all warps).
// FP32 accumulation, one FP16 group scale per 32-code slice (a slice never crosses a group).
template <int Tile>
__global__ void __launch_bounds__(kThreads)
    t2_project_kernel(const __nv_bfloat16* __restrict__ x, const uint2* __restrict__ codes,
                      const __half* __restrict__ scales, std::int64_t scale_row_halves, int n,
                      int k, int tokens, Outputs outputs, bool accumulate) {
    const int warp   = static_cast<int>(threadIdx.x) / 32;
    const int lane   = static_cast<int>(threadIdx.x) % 32;
    const int token0 = static_cast<int>(blockIdx.y) * Tile;
    const int live   = min(Tile, tokens - token0);
    const int row0   = (static_cast<int>(blockIdx.x) * kWarps + warp) * kRowsPerWarp;
    if (row0 >= n) return;
    const bool pair           = row0 + 1 < n;
    const int slices_per_row  = k / 32;
    const uint2* row_codes[2] = {codes + std::int64_t(row0) * slices_per_row,
                                 codes + std::int64_t(pair ? row0 + 1 : row0) * slices_per_row};
    const __half* row_scales[2] = {scales + std::int64_t(row0) * scale_row_halves,
                                   scales + std::int64_t(pair ? row0 + 1 : row0) * scale_row_halves};
    const __nv_bfloat16* xt = x + std::int64_t(token0) * k;

    float acc[kRowsPerWarp][Tile];
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
#pragma unroll
        for (int t = 0; t < Tile; ++t) acc[r][t] = 0.0f;
    }

    // Weights are read once: stream them past L1 so the reused x stays resident. The next
    // slice's codes are requested before the current slice is computed.
    uint2 next_bits[kRowsPerWarp];
    __half next_scale[kRowsPerWarp];
    const auto fetch = [&](int slice) {
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            next_bits[r]  = __ldcs(row_codes[r] + slice);
            next_scale[r] = __ldcs(row_scales[r] + slice / 4);
        }
    };
    if (lane < slices_per_row) fetch(lane);
    for (int slice = lane; slice < slices_per_row; slice += 32) {
        uint2 bits[kRowsPerWarp];
        float scale[kRowsPerWarp];
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            bits[r]  = next_bits[r];
            scale[r] = __half2float(next_scale[r]);
        }
        if (slice + 32 < slices_per_row) fetch(slice + 32);
        const int column = slice * 32;
        float partial[kRowsPerWarp][Tile];
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
#pragma unroll
            for (int t = 0; t < Tile; ++t) partial[r][t] = 0.0f;
        }
        // Eight columns at a time: load them for every token, then decode each weight once and
        // apply it to all tokens (the loop order, not the compiler, guarantees the reuse).
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            uint4 xs[Tile];
#pragma unroll
            for (int t = 0; t < Tile; ++t) {
                xs[t] = t < live ? __ldg(reinterpret_cast<const uint4*>(xt + std::int64_t(t) * k +
                                                                        column) + q)
                                 : make_uint4(0, 0, 0, 0);
            }
#pragma unroll
            for (int h = 0; h < 8; ++h) {
                const int j = q * 8 + h; // code index within the 32-code slice
#pragma unroll
                for (int r = 0; r < kRowsPerWarp; ++r) {
                    const std::uint32_t word = j < 16 ? bits[r].x : bits[r].y;
                    const float weight =
                        static_cast<float>(static_cast<int>((word >> ((j % 16) * 2)) & 3u) - 1);
#pragma unroll
                    for (int t = 0; t < Tile; ++t) {
                        const std::uint32_t pairs[4] = {xs[t].x, xs[t].y, xs[t].z, xs[t].w};
                        const std::uint32_t pair     = pairs[h / 2];
                        // BF16 pair: the low half is the lower column.
                        const float value = __uint_as_float(h % 2 ? pair & 0xffff0000u : pair << 16);
                        partial[r][t]     = fmaf(weight, value, partial[r][t]);
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
#pragma unroll
            for (int t = 0; t < Tile; ++t) acc[r][t] = fmaf(scale[r], partial[r][t], acc[r][t]);
        }
    }

#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
#pragma unroll
        for (int t = 0; t < Tile; ++t) {
            float v = acc[r][t];
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                v += __shfl_xor_sync(0xffffffffu, v, offset);
            }
            acc[r][t] = v;
        }
    }
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        const int row = row0 + r;
        if (row >= n) continue;
        int s = 0;
        while (row >= outputs.end[s]) ++s;
        const int begin = s ? outputs.end[s - 1] : 0;
        const int rows  = outputs.end[s] - begin;
#pragma unroll
        for (int t = 0; t < Tile; ++t) {
            if (t < live && lane == t) {
                auto* out = outputs.data[s] + static_cast<std::int64_t>(token0 + t) * rows +
                            (row - begin);
                const float value = accumulate ? acc[r][t] + __bfloat162float(*out) : acc[r][t];
                *out              = __float2bfloat16_rn(value);
            }
        }
    }
}

template <int Tile>
void launch(const Tensor& x, const Weight& w, const Outputs& outputs, bool accumulate,
            cudaStream_t stream) {
    const int tokens         = x.ne[1];
    constexpr int kRowsBlock = kWarps * kRowsPerWarp;
    const dim3 grid(static_cast<unsigned>((w.n + kRowsBlock - 1) / kRowsBlock),
                    static_cast<unsigned>((tokens + Tile - 1) / Tile));
    t2_project_kernel<Tile><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const uint2*>(w.qdata),
        static_cast<const __half*>(w.scales), w.scale_nb[1] / 2, w.n, w.k, tokens, outputs,
        accumulate);
    CUDA_CHECK(cudaGetLastError());
}

bool aligned16(const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 15) == 0; }

} // namespace

void validate_t2_weight(const Weight& w, const char* op) {
    if (w.qtype != QType::T2_G128_FP16 || w.layout != QuantLayout::TernaryRowK128 ||
        w.scale_dtype != DType::FP16 || w.group_size != 128 || w.n <= 0 || w.k % 1024 ||
        w.qdata == nullptr || w.qhigh != nullptr || w.scales == nullptr ||
        (reinterpret_cast<std::uintptr_t>(w.qdata) & 15) ||
        (reinterpret_cast<std::uintptr_t>(w.scales) & 15) || w.scale_nb[1] != w.k / 64) {
        throw std::invalid_argument(std::string(op) +
                                    ": weight must be T2_G128_FP16 ternary rows with K%1024=0");
    }
}

void t2_project(const Tensor& x, const Weight& w, std::span<Tensor* const> outputs,
                bool accumulate, cudaStream_t stream) {
    validate_t2_weight(w, "t2_project");
    const int tokens = x.ne[1];
    if (x.dtype != DType::BF16 || x.ne[0] != w.k || tokens <= 0 || x.ne[2] != 1 ||
        x.ne[3] != 1 || !x.is_contiguous() || !aligned16(x.data)) {
        throw std::invalid_argument("t2_project: x must be contiguous aligned BF16 [K,T]");
    }
    if (outputs.empty() || outputs.size() > kMaxOutputs) {
        throw std::invalid_argument("t2_project: expected one to four outputs");
    }
    Outputs packed{};
    int end = 0;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        const Tensor& out = *outputs[i];
        if (out.dtype != DType::BF16 || out.ne[1] != tokens || out.ne[2] != 1 || out.ne[3] != 1 ||
            !out.is_contiguous() || out.data == nullptr) {
            throw std::invalid_argument("t2_project: outputs must be contiguous BF16 [rows,T]");
        }
        end += out.ne[0];
        packed.data[i] = static_cast<__nv_bfloat16*>(out.data);
        packed.end[i]  = end;
    }
    for (std::size_t i = outputs.size(); i < kMaxOutputs; ++i) {
        packed.data[i] = packed.data[outputs.size() - 1];
        packed.end[i]  = end;
    }
    if (end != w.n) {
        throw std::invalid_argument("t2_project: output rows must cover the weight rows");
    }
    if (tokens >= 2 && w.n % t2_mma::kRows == 0) {
        // Tensor cores keep the cost per weight byte flat in T (MTP verification, prefill).
        t2_mma::Outputs mma_outputs{};
        for (int i = 0; i < kMaxOutputs; ++i) {
            mma_outputs.data[i] = packed.data[i];
            mma_outputs.end[i]  = packed.end[i];
        }
        const dim3 grid(static_cast<unsigned>(w.n / t2_mma::kRows),
                        static_cast<unsigned>((tokens + t2_mma::kTokens - 1) / t2_mma::kTokens));
        t2_mma::t2_mma_kernel<<<grid, t2_mma::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const __half*>(w.scales), w.scale_nb[1] / 2, w.k, tokens, mma_outputs,
            accumulate);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    switch (tokens) {
    case 1: launch<1>(x, w, packed, accumulate, stream); break;
    case 2: launch<2>(x, w, packed, accumulate, stream); break;
    case 3: launch<3>(x, w, packed, accumulate, stream); break;
    case 4: launch<4>(x, w, packed, accumulate, stream); break;
    default: launch<kMaxTile>(x, w, packed, accumulate, stream); break;
    }
}

} // namespace ninfer::ops::detail
