#include "ops/linear/t5/t5_project.h"

#include "core/device.h"
#include "ops/linear/t5/t5_a8.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

// Routes by T (design 9.1): the dp4a GEMV through T = 4, the small-T tensor-core route through
// T = 32 (MTP verification of up to eight lanes at draft 3), the prefill GEMM beyond.
constexpr int kSmallTMaxTokens = 8 * t5_a8::kSmallMaxTiles;

bool aligned16(const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 15) == 0; }

std::size_t round_up_256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

struct QuantizedX {
    Tensor q, scale, group_sum, slice_sum;
};

// Quantizes the weight input of `input` (a t5_a8 prologue), rotated when the weight carries
// input signs, in one kernel.
template <class Input>
QuantizedX quantize(const Input& input, int k, int tokens, const void* signs,
                    WorkspaceArena& workspace, cudaStream_t stream) {
    QuantizedX result{workspace.alloc(DType::I8, {k, tokens}),
                      workspace.alloc(DType::FP32, {k / 128, tokens}),
                      workspace.alloc(DType::I32, {k / 128, tokens}),
                      workspace.alloc(DType::I32, {k / 32, tokens})};
    auto* q     = static_cast<std::uint32_t*>(result.q.data);
    auto* scale = static_cast<float*>(result.scale.data);
    auto* gsum  = static_cast<int*>(result.group_sum.data);
    auto* ssum  = static_cast<int*>(result.slice_sum.data);
    const auto* sign = static_cast<const __nv_bfloat16*>(signs);
    const dim3 grid(static_cast<unsigned>(k / t5_a8::kQuantizeBlock),
                    static_cast<unsigned>(tokens));
    if (sign != nullptr) {
        t5_a8::quantize_kernel<Input, true>
            <<<grid, t5_a8::kQuantizeThreads, 0, stream>>>(input, sign, k, q, scale, gsum, ssum);
    } else {
        t5_a8::quantize_kernel<Input, false>
            <<<grid, t5_a8::kQuantizeThreads, 0, stream>>>(input, sign, k, q, scale, gsum, ssum);
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}

template <int Tile>
void launch_gemv(const QuantizedX& q, const Weight& w, const t5_a8::Outputs& outputs,
                 bool accumulate, cudaStream_t stream) {
    const unsigned grid =
        static_cast<unsigned>((w.n + t5_a8::kGemvRowsPerCta - 1) / t5_a8::kGemvRowsPerCta);
    t5_a8::gemv_kernel<Tile><<<grid, t5_a8::kGemvThreads, 0, stream>>>(
        static_cast<const uint4*>(q.q.data), static_cast<const float*>(q.scale.data),
        static_cast<const int*>(q.slice_sum.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const __half*>(w.scales), w.scale_nb[1] / 2, w.n, w.k, outputs, accumulate);
    CUDA_CHECK(cudaGetLastError());
}

template <int NTiles>
void launch_small_t(const QuantizedX& q, const Weight& w, int tokens,
                    const t5_a8::Outputs& outputs, bool accumulate, cudaStream_t stream) {
    constexpr int kTokens = 8 * NTiles;
    const dim3 grid(static_cast<unsigned>((w.n + t5_a8::kSmallRows - 1) / t5_a8::kSmallRows),
                    static_cast<unsigned>((tokens + kTokens - 1) / kTokens));
    t5_a8::small_t_kernel<NTiles><<<grid, t5_a8::kSmallThreads, 0, stream>>>(
        static_cast<const std::uint8_t*>(q.q.data), static_cast<const float*>(q.scale.data),
        static_cast<const int*>(q.group_sum.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const __half*>(w.scales), w.scale_nb[1] / 2, w.n, w.k, tokens, outputs,
        accumulate);
    CUDA_CHECK(cudaGetLastError());
}

template <int WarpsM, int WarpsN>
void launch_gemm(const QuantizedX& q, const Weight& w, int tokens, const t5_a8::Outputs& outputs,
                 bool accumulate, cudaStream_t stream) {
    using Config                = t5_a8::GemmConfig<WarpsM, WarpsN>;
    constexpr std::size_t kSmem = t5_a8::gemm_shared_bytes<WarpsM, WarpsN>();
    static_assert(kSmem <= 48 * 1024);
    const int token_tiles = (tokens + Config::kTokens - 1) / Config::kTokens;
    const unsigned grid   = static_cast<unsigned>(w.n / Config::kRows * token_tiles);
    t5_a8::gemm_kernel<WarpsM, WarpsN><<<grid, Config::kThreads, kSmem, stream>>>(
        static_cast<const std::uint8_t*>(q.q.data), static_cast<const float*>(q.scale.data),
        static_cast<const int*>(q.group_sum.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const __half*>(w.scales), w.scale_nb[1] / 2, w.k, tokens, token_tiles, outputs,
        accumulate);
    CUDA_CHECK(cudaGetLastError());
}

void launch_tall_gemm(const QuantizedX& q, const Weight& w, int tokens,
                      const t5_a8::Outputs& outputs, bool accumulate, cudaStream_t stream) {
    constexpr std::size_t kSmem = t5_a8::gemm_tall_shared_bytes();
    static const bool opted_in  = [] {
        CUDA_CHECK(cudaFuncSetAttribute(t5_a8::gemm_tall_kernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(kSmem)));
        return true;
    }();
    (void)opted_in;
    const int token_tiles = (tokens + t5_a8::kTallTokens - 1) / t5_a8::kTallTokens;
    const unsigned grid   = static_cast<unsigned>(w.n / t5_a8::kTallRows * token_tiles);
    t5_a8::gemm_tall_kernel<<<grid, t5_a8::kTallThreads, kSmem, stream>>>(
        static_cast<const std::uint8_t*>(q.q.data), static_cast<const float*>(q.scale.data),
        static_cast<const int*>(q.group_sum.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const __half*>(w.scales), w.scale_nb[1] / 2, w.k, tokens, token_tiles, outputs,
        accumulate);
    CUDA_CHECK(cudaGetLastError());
}

// Short prefills (MTP/DFlash2 tails, chunk remainders) keep 64-token CTAs; longer ones share
// each decoded weight stage across 128 tokens. The exception is a 128-token grid that leaves SMs
// idle over a short K: one eight-warp CTA per SM then runs longer than the 64-token CTAs it
// replaces (o_proj 5120 x 6144 at T = 128: 80 CTAs, +14 %), while a long K still gains (down
// 5120 x 17408, -3 %) and grids of at least one CTA per SM gain 25-30 % (design 9.1).
bool use_small_gemm_tile(const Weight& w, int tokens) {
    using Large = t5_a8::GemmConfig<2, 4>;
    if (tokens <= t5_a8::GemmConfig<2, 2>::kTokens) return true;
    constexpr int kLongK = 16384;
    const std::int64_t large_ctas =
        std::int64_t(w.n / Large::kRows) * ((tokens + Large::kTokens - 1) / Large::kTokens);
    return large_ctas < device_sm_count() && w.k < kLongK;
}

// Weights of at least 8192 rows take the 128 x 128 kernel: each staged activation tile then
// serves 128 rows, halving the L2 activation reads per output (gate+up, GDN in_proj and attention
// qkvg). The 5120-row weights keep 64-row CTAs: 40 row blocks leave the one-CTA-per-SM tile too
// few CTAs per wave (o_proj +8 % at T = 1024, design 9.1).
bool use_tall_gemm_tile(const Weight& w) {
    return w.n >= 8192 && w.n % t5_a8::kTallRows == 0;
}

void require_input(const Tensor& x, const Weight& w, const char* what) {
    if (x.dtype != DType::BF16 || x.ne[0] != w.k || x.ne[1] <= 0 || x.ne[2] != 1 || x.ne[3] != 1 ||
        !x.is_contiguous() || !aligned16(x.data)) {
        throw std::invalid_argument(std::string("t5_project: ") + what +
                                    " must be contiguous aligned BF16 [K,T]");
    }
}

// Validates the weight, policy and outputs, then quantizes the input and multiplies: dp4a GEMV
// through T = 4, the small-T MMA route through T = 32 (and at any T when N % 64 != 0, in 32-token
// tiles), the int8 MMA GEMM beyond.
template <class Input>
void project(const Input& input, int tokens, const Weight& w, std::span<Tensor* const> outputs,
             bool accumulate, LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) {
    if (!allows_a8(policy) || workspace == nullptr) {
        throw std::invalid_argument("t5_project: a T5 weight requires AllowA8 and a workspace");
    }
    if (outputs.empty() || outputs.size() > t5_a8::kMaxOutputs) {
        throw std::invalid_argument("t5_project: expected one to four outputs");
    }
    t5_a8::Outputs packed{};
    int end = 0;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        const Tensor& out = *outputs[i];
        if (out.dtype != DType::BF16 || out.ne[1] != tokens || out.ne[2] != 1 || out.ne[3] != 1 ||
            !out.is_contiguous() || out.data == nullptr) {
            throw std::invalid_argument("t5_project: outputs must be contiguous BF16 [rows,T]");
        }
        end += out.ne[0];
        packed.data[i] = static_cast<__nv_bfloat16*>(out.data);
        packed.end[i]  = end;
    }
    for (std::size_t i = outputs.size(); i < t5_a8::kMaxOutputs; ++i) {
        packed.data[i] = packed.data[outputs.size() - 1];
        packed.end[i]  = end;
    }
    if (end != w.n) {
        throw std::invalid_argument("t5_project: output rows must cover the weight rows");
    }

    auto scope         = workspace->scope();
    const QuantizedX q = quantize(input, w.k, tokens, w.input_signs, *workspace, stream);
    switch (tokens) {
    case 1: launch_gemv<1>(q, w, packed, accumulate, stream); return;
    case 2: launch_gemv<2>(q, w, packed, accumulate, stream); return;
    case 3: launch_gemv<3>(q, w, packed, accumulate, stream); return;
    case 4: launch_gemv<4>(q, w, packed, accumulate, stream); return;
    default: break;
    }
    if (tokens <= kSmallTMaxTokens || w.n % t5_a8::GemmConfig<2, 2>::kRows != 0) {
        switch ((std::min(tokens, kSmallTMaxTokens) + 7) / 8) {
        case 1: launch_small_t<1>(q, w, tokens, packed, accumulate, stream); break;
        case 2: launch_small_t<2>(q, w, tokens, packed, accumulate, stream); break;
        case 3: launch_small_t<3>(q, w, tokens, packed, accumulate, stream); break;
        default: launch_small_t<4>(q, w, tokens, packed, accumulate, stream); break;
        }
    } else if (use_small_gemm_tile(w, tokens)) {
        launch_gemm<2, 2>(q, w, tokens, packed, accumulate, stream);
    } else if (use_tall_gemm_tile(w)) {
        launch_tall_gemm(q, w, tokens, packed, accumulate, stream);
    } else {
        launch_gemm<2, 4>(q, w, tokens, packed, accumulate, stream);
    }
}

} // namespace

void validate_t5_weight(const Weight& w, const char* op) {
    if (w.qtype != QType::T5_G128_FP16 || w.layout != QuantLayout::TernaryRowK128 ||
        w.scale_dtype != DType::FP16 || w.group_size != 128 || w.n <= 0 || w.k % 1024 ||
        w.qdata == nullptr || w.qhigh != nullptr || w.scales == nullptr ||
        (reinterpret_cast<std::uintptr_t>(w.qdata) & 15) ||
        (reinterpret_cast<std::uintptr_t>(w.scales) & 15) || w.scale_nb[1] != w.k / 64 ||
        (reinterpret_cast<std::uintptr_t>(w.input_signs) & 1)) {
        throw std::invalid_argument(std::string(op) +
                                    ": weight must be T5_G128_FP16 ternary rows with K%1024=0");
    }
}

std::size_t t5_workspace_capacity_bytes(LinearPolicy policy, std::int32_t input_rows,
                                        std::int32_t max_tokens) {
    if (input_rows <= 0 || input_rows % 1024 || max_tokens <= 0) {
        throw std::invalid_argument("t5 workspace: requires K % 1024 == 0 and positive T");
    }
    if (!allows_a8(policy)) { throw std::invalid_argument("t5 workspace: requires AllowA8"); }
    const std::size_t k = static_cast<std::size_t>(input_rows);
    const std::size_t t = static_cast<std::size_t>(max_tokens);
    return round_up_256(k * t) + 2 * round_up_256(k / 128 * t * 4) + round_up_256(k / 32 * t * 4);
}

void t5_project(const Tensor& x, const Weight& w, std::span<Tensor* const> outputs, bool accumulate,
                LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) {
    validate_t5_weight(w, "t5_project");
    require_input(x, w, "x");
    project(t5_a8::PlainInput{static_cast<const __nv_bfloat16*>(x.data)}, x.ne[1], w, outputs,
            accumulate, policy, workspace, stream);
}

void t5_project_rmsnorm(const Tensor& x, const Tensor& norm_weight, float eps, bool unit_offset,
                        const Weight& w, std::span<Tensor* const> outputs, bool accumulate,
                        LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) {
    validate_t5_weight(w, "t5_project_rmsnorm");
    require_input(x, w, "x");
    if (norm_weight.dtype != DType::BF16 || norm_weight.ne[0] != w.k || norm_weight.ne[1] != 1 ||
        norm_weight.ne[2] != 1 || norm_weight.ne[3] != 1 || !norm_weight.is_contiguous() ||
        norm_weight.data == nullptr) {
        throw std::invalid_argument("t5_project_rmsnorm: norm weight must be contiguous BF16 [K]");
    }
    if (!(eps > 0.0f) || !std::isfinite(eps)) {
        throw std::invalid_argument("t5_project_rmsnorm: eps must be positive and finite");
    }
    project(t5_a8::RmsNormInput{static_cast<const __nv_bfloat16*>(x.data),
                                static_cast<const __nv_bfloat16*>(norm_weight.data), eps,
                                unit_offset},
            x.ne[1], w, outputs, accumulate, policy, workspace, stream);
}

void t5_project_swiglu(const Tensor& gate, const Tensor& up, const Weight& w,
                       std::span<Tensor* const> outputs, bool accumulate, LinearPolicy policy,
                       WorkspaceArena* workspace, cudaStream_t stream) {
    validate_t5_weight(w, "t5_project_swiglu");
    require_input(gate, w, "gate");
    require_input(up, w, "up");
    if (up.ne[1] != gate.ne[1]) {
        throw std::invalid_argument("t5_project_swiglu: gate and up must have the same T");
    }
    project(t5_a8::SwiGluInput{static_cast<const __nv_bfloat16*>(gate.data),
                               static_cast<const __nv_bfloat16*>(up.data)},
            gate.ne[1], w, outputs, accumulate, policy, workspace, stream);
}

} // namespace ninfer::ops::detail
