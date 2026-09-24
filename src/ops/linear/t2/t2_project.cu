#include "ops/linear/t2/t2_project.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

constexpr int kChunk         = 1024;        // x columns staged per pass
constexpr int kWordsPerChunk = kChunk / 16; // 16 two-bit codes per 32-bit word
constexpr int kThreads       = 256;
constexpr int kRowsPerWarp   = 2;
constexpr int kRowsPerBlock  = kThreads / 32 * kRowsPerWarp;
constexpr int kMaxOutputs    = 4;

struct Outputs {
    __nv_bfloat16* data[kMaxOutputs];
    int end[kMaxOutputs]; // exclusive parent row bound of each output
};

__device__ __forceinline__ void unpack8(const uint4& v, float (&out)[8]) {
    const auto* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(h[i]);
        out[2 * i]     = f.x;
        out[2 * i + 1] = f.y;
    }
}

// Each warp owns kRowsPerWarp parent rows and one tile of Tile tokens; lanes split the 64 code
// words of every staged 1024-column chunk. Code c contributes (c - 1) * x; the per-word partial
// sum is scaled once by its group's FP16 scale.
template <int Tile>
__global__ void __launch_bounds__(kThreads)
    t2_project_kernel(const __nv_bfloat16* __restrict__ x, const std::uint32_t* __restrict__ codes,
                      const __half* __restrict__ scales, std::int64_t scale_row_halves, int n,
                      int k, int tokens, Outputs outputs, bool accumulate) {
    __shared__ __align__(16) __nv_bfloat16 staged[Tile][kChunk];
    const int warp   = static_cast<int>(threadIdx.x) / 32;
    const int lane   = static_cast<int>(threadIdx.x) % 32;
    const int token0 = static_cast<int>(blockIdx.y) * Tile;
    const int live   = min(Tile, tokens - token0);
    const int row0   = static_cast<int>(blockIdx.x) * kRowsPerBlock + warp * kRowsPerWarp;
    const std::int64_t words_per_row = k / 16;

    float acc[kRowsPerWarp][Tile];
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
#pragma unroll
        for (int t = 0; t < Tile; ++t) acc[r][t] = 0.0f;
    }

    for (int chunk = 0; chunk < k / kChunk; ++chunk) {
        constexpr int kVectors = kChunk / 8;
        for (int i = static_cast<int>(threadIdx.x); i < live * kVectors; i += kThreads) {
            const int t = i / kVectors, v = i % kVectors;
            reinterpret_cast<uint4*>(staged[t])[v] = reinterpret_cast<const uint4*>(
                x + static_cast<std::int64_t>(token0 + t) * k + chunk * kChunk)[v];
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            const int row = row0 + r;
            if (row < n) {
                for (int word = lane; word < kWordsPerChunk; word += 32) {
                    const std::int64_t index = chunk * kWordsPerChunk + word;
                    const std::uint32_t bits = codes[row * words_per_row + index];
                    const float scale = __half2float(scales[row * scale_row_halves + index / 8]);
                    float weight[16];
#pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        weight[j] = static_cast<float>(static_cast<int>((bits >> (2 * j)) & 3u) - 1);
                    }
#pragma unroll
                    for (int t = 0; t < Tile; ++t) {
                        if (t < live) {
                            const auto* xv = reinterpret_cast<const uint4*>(&staged[t][word * 16]);
                            float lo[8], hi[8];
                            unpack8(xv[0], lo);
                            unpack8(xv[1], hi);
                            float partial = 0.0f;
#pragma unroll
                            for (int j = 0; j < 8; ++j) {
                                partial = fmaf(weight[j], lo[j], partial);
                                partial = fmaf(weight[8 + j], hi[j], partial);
                            }
                            acc[r][t] = fmaf(scale, partial, acc[r][t]);
                        }
                    }
                }
            }
        }
        __syncthreads();
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
            if (t < live && lane == t % 32) {
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
    const int tokens = x.ne[1];
    const dim3 grid(static_cast<unsigned>((w.n + kRowsPerBlock - 1) / kRowsPerBlock),
                    static_cast<unsigned>((tokens + Tile - 1) / Tile));
    t2_project_kernel<Tile><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint32_t*>(w.qdata),
        static_cast<const __half*>(w.scales), w.scale_nb[1] / 2, w.n, w.k, tokens, outputs,
        accumulate);
    CUDA_CHECK(cudaGetLastError());
}

bool aligned16(const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 15) == 0; }

} // namespace

void validate_t2_weight(const Weight& w, const char* op) {
    if (w.qtype != QType::T2_G128_FP16 || w.layout != QuantLayout::TernaryRowK128 ||
        w.scale_dtype != DType::FP16 || w.group_size != 128 || w.n <= 0 || w.k % kChunk ||
        w.qdata == nullptr || w.qhigh != nullptr || w.scales == nullptr ||
        (reinterpret_cast<std::uintptr_t>(w.qdata) & 3) || w.scale_nb[1] != w.k / 64) {
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
    if (tokens <= 8) {
        launch<8>(x, w, packed, accumulate, stream);
    } else {
        launch<16>(x, w, packed, accumulate, stream);
    }
}

} // namespace ninfer::ops::detail
