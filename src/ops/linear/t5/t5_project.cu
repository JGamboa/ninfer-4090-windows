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

// Bytes of code rows that a decode-width quantization asks L2 for ahead of the GEMV
// (t5_a8::WeightPrefetch). One MiB is what DRAM moves while the quantization runs: per MTP 2
// round on the RTX 4090, 0.75 MiB saved 1.5 %, 1-1.5 MiB 1.9 %, 4 MiB 0.9 % (design 9.1 item 33).
constexpr std::int64_t kGemvPrefetchBytes = std::int64_t(1) << 20;

t5_a8::WeightPrefetch gemv_prefetch(const Weight& w) {
    const std::int64_t row_bytes = std::int64_t(w.k / t5_a8::kUnitColumns) * t5_a8::kUnitBytes;
    const std::int64_t rows =
        std::min<std::int64_t>(w.n, std::max<std::int64_t>(1, kGemvPrefetchBytes / row_bytes));
    // Line i starts at base + 128 i, inside the first `bytes` of the plane for every i counted.
    const auto lines = [](std::int64_t bytes) { return static_cast<unsigned>((bytes + 127) / 128); };
    return {static_cast<const char*>(w.qdata), static_cast<const char*>(w.scales),
            lines(rows * row_bytes), lines(rows * w.scale_nb[1])};
}

// Quantizes the weight input of `input` (a t5_a8 prologue), rotated when the weight carries
// input signs, in one kernel. At GEMV widths the kernel also prefetches the weight's leading rows.
template <class Input>
QuantizedX quantize(const Input& input, const Weight& w, int tokens, WorkspaceArena& workspace,
                    cudaStream_t stream) {
    const int k = w.k;
    QuantizedX result{workspace.alloc(DType::I8, {k, tokens}),
                      workspace.alloc(DType::FP32, {k / 128, tokens}),
                      workspace.alloc(DType::I32, {k / 128, tokens}),
                      workspace.alloc(DType::I32, {k / 32, tokens})};
    auto* q     = static_cast<std::uint32_t*>(result.q.data);
    auto* scale = static_cast<float*>(result.scale.data);
    auto* gsum  = static_cast<int*>(result.group_sum.data);
    auto* ssum  = static_cast<int*>(result.slice_sum.data);
    const auto* sign = static_cast<const __nv_bfloat16*>(w.input_signs);
    // A token-state prologue visits all blocks from one CTA per token once the tokens alone fill
    // the SMs (prefill); with fewer tokens each CTA prepares the state for its own block, so a
    // decode or verification quantization is not K / 1024 blocks in series.
    const bool whole_row = Input::kTokenState && tokens >= device_sm_count();
    const bool prefetch  = tokens <= t5_a8::kGemvMaxTokens;
    const t5_a8::WeightPrefetch lines =
        prefetch ? gemv_prefetch(w) : t5_a8::WeightPrefetch{nullptr, nullptr, 0u, 0u};
    const dim3 grid(whole_row ? 1u : static_cast<unsigned>(k / t5_a8::kQuantizeBlock),
                    static_cast<unsigned>(tokens + (prefetch ? 1 : 0)));
    if (sign != nullptr) {
        t5_a8::quantize_kernel<Input, true><<<grid, t5_a8::kQuantizeThreads, 0, stream>>>(
            input, sign, k, whole_row, q, scale, gsum, ssum, tokens, lines);
    } else {
        t5_a8::quantize_kernel<Input, false><<<grid, t5_a8::kQuantizeThreads, 0, stream>>>(
            input, sign, k, whole_row, q, scale, gsum, ssum, tokens, lines);
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

void launch_gemm(const QuantizedX& q, const Weight& w, int tokens, const t5_a8::Outputs& outputs,
                 bool accumulate, cudaStream_t stream) {
    constexpr std::size_t kSmem = t5_a8::gemm_shared_bytes();
    static_assert(kSmem <= 48 * 1024);
    const int token_tiles = (tokens + t5_a8::kGemmTokens - 1) / t5_a8::kGemmTokens;
    const unsigned grid   = static_cast<unsigned>(w.n / t5_a8::kGemmRows * token_tiles);
    t5_a8::gemm_kernel<<<grid, t5_a8::kGemmThreads, kSmem, stream>>>(
        static_cast<const std::uint8_t*>(q.q.data), static_cast<const float*>(q.scale.data),
        static_cast<const int*>(q.group_sum.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const __half*>(w.scales), w.scale_nb[1] / 2, w.k, tokens, token_tiles, outputs,
        accumulate);
    CUDA_CHECK(cudaGetLastError());
}

// The token tiles of each 128-row block: `wide` 128-token tiles, then `narrow` 64-token tiles.
struct TallSplit {
    int wide, narrow;
};

// A 128-token CTA of gemm_tall_kernel costs ~1.6x a 64-token one (the decode per step is the
// same, the MMAs double; measured 1.6-1.75x on the 5120-row weights at T = 512..2048).
constexpr double kWideCost = 1.6;

// Time until the last CTA ends when `wide` CTAs of cost kWideCost and then `narrow` CTAs of cost 1
// are started one per SM, each on the SM that frees first (the order of the launch).
double tall_makespan(std::int64_t wide, std::int64_t narrow, int sms) {
    const std::int64_t waves = wide / sms, rest = wide % sms;
    // sms - rest SMs free at `early`, rest SMs at `late`.
    const double early = double(waves) * kWideCost, late = early + kWideCost;
    if (narrow == 0) return rest > 0 ? late : early;
    // The narrow CTAs end at early + i and late + j (i, j >= 1); the last one is the
    // narrow-th smallest of these ends counted with their SM multiplicities.
    std::int64_t i = 1, j = 1;
    for (;;) {
        const double at_early = early + double(i);
        const double at_late  = rest > 0 ? late + double(j) : at_early + 1.0;
        if (at_early <= at_late) {
            narrow -= sms - rest;
            ++i;
            if (narrow <= 0) return rest > 0 ? std::max(at_early, late) : at_early;
        } else {
            narrow -= rest;
            ++j;
            if (narrow <= 0) return at_late;
        }
    }
}

// The 5120-row weights have only 40 row blocks, so whole 128-token tiles quantize the last wave
// coarsely (T = 1024: 320 CTAs, 2.5 waves) and whole 64-token tiles pay the decode twice per
// output. Covering the first tokens of every row block with 128-token tiles and the rest with
// 64-token tiles lets the cheaper CTAs fill the last wave; the split with the shortest estimated
// makespan is taken (T = 1024: 4 + 8 tiles per row block, 4-10 % faster than either uniform
// width). Arithmetic per output does not depend on the split.
TallSplit choose_tall_split(int row_blocks, int tokens) {
    const int sms   = device_sm_count();
    TallSplit best  = {(tokens + 127) / 128, 0};
    double shortest = tall_makespan(std::int64_t(row_blocks) * best.wide, 0, sms);
    for (int wide = 0; 128 * wide < tokens; ++wide) {
        const int narrow  = (tokens - 128 * wide + 63) / 64;
        const double time = tall_makespan(std::int64_t(row_blocks) * wide,
                                          std::int64_t(row_blocks) * narrow, sms);
        if (time < shortest) {
            shortest = time;
            best     = {wide, narrow};
        }
    }
    return best;
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
    const int row_blocks  = w.n / t5_a8::kTallRows;
    const TallSplit split = choose_tall_split(row_blocks, tokens);
    const unsigned grid   = static_cast<unsigned>(row_blocks * (split.wide + split.narrow));
    t5_a8::gemm_tall_kernel<<<grid, t5_a8::kTallThreads, kSmem, stream>>>(
        static_cast<const std::uint8_t*>(q.q.data), static_cast<const float*>(q.scale.data),
        static_cast<const int*>(q.group_sum.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const __half*>(w.scales), w.scale_nb[1] / 2, w.k, tokens, split.wide,
        split.narrow, outputs, accumulate);
    CUDA_CHECK(cudaGetLastError());
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
// tiles), the 64 x 64 GEMM through T = 64 (and beyond when N % 128 != 0 or an output is not made
// of aligned 8-row chunks), the pipelined 128-row GEMM beyond (every Bonsai weight).
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
    // The pipelined GEMM stores 8-row chunks as 16-byte words.
    bool chunked_outputs = true;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        const Tensor& out = *outputs[i];
        if (out.dtype != DType::BF16 || out.ne[1] != tokens || out.ne[2] != 1 || out.ne[3] != 1 ||
            !out.is_contiguous() || out.data == nullptr) {
            throw std::invalid_argument("t5_project: outputs must be contiguous BF16 [rows,T]");
        }
        end += out.ne[0];
        chunked_outputs = chunked_outputs && out.ne[0] % 8 == 0 && aligned16(out.data);
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
    const QuantizedX q = quantize(input, w, tokens, *workspace, stream);
    switch (tokens) {
    case 1: launch_gemv<1>(q, w, packed, accumulate, stream); return;
    case 2: launch_gemv<2>(q, w, packed, accumulate, stream); return;
    case 3: launch_gemv<3>(q, w, packed, accumulate, stream); return;
    case 4: launch_gemv<4>(q, w, packed, accumulate, stream); return;
    default: break;
    }
    if (tokens <= kSmallTMaxTokens || w.n % t5_a8::kGemmRows != 0) {
        switch ((std::min(tokens, kSmallTMaxTokens) + 7) / 8) {
        case 1: launch_small_t<1>(q, w, tokens, packed, accumulate, stream); break;
        case 2: launch_small_t<2>(q, w, tokens, packed, accumulate, stream); break;
        case 3: launch_small_t<3>(q, w, tokens, packed, accumulate, stream); break;
        default: launch_small_t<4>(q, w, tokens, packed, accumulate, stream); break;
        }
    } else if (tokens > t5_a8::kGemmTokens && w.n % t5_a8::kTallRows == 0 && chunked_outputs) {
        launch_tall_gemm(q, w, tokens, packed, accumulate, stream);
    } else {
        launch_gemm(q, w, tokens, packed, accumulate, stream);
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
