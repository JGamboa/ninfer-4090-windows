#include "ninfer/ops/rmsnorm_swiglu_mlp.h"

#include "ops/linear/t5/t5_project.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

std::size_t round_up_256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

void validate_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("rmsnorm_swiglu_mlp: invalid compute policy");
}

} // namespace

bool rmsnorm_swiglu_mlp_accepts(QType gate_up_qtype, LinearPolicy gate_up_policy, QType down_qtype,
                                LinearPolicy down_policy) {
    validate_policy(gate_up_policy);
    validate_policy(down_policy);
    return gate_up_qtype == QType::T5_G128_FP16 && down_qtype == QType::T5_G128_FP16 &&
           allows_a8(gate_up_policy) && allows_a8(down_policy);
}

std::size_t rmsnorm_swiglu_mlp_workspace_capacity_bytes(
    QType qtype, std::int32_t gate_up_rows, std::int32_t input_rows, LinearPolicy gate_up_policy,
    LinearPolicy down_policy, std::int32_t min_tokens, std::int32_t max_tokens) {
    if (!rmsnorm_swiglu_mlp_accepts(qtype, gate_up_policy, qtype, down_policy)) {
        throw std::invalid_argument("rmsnorm_swiglu_mlp workspace: unregistered profile");
    }
    if (min_tokens <= 0 || max_tokens < min_tokens || gate_up_rows <= 0 || gate_up_rows % 2048 ||
        input_rows <= 0 || input_rows % 1024) {
        throw std::invalid_argument("rmsnorm_swiglu_mlp workspace: invalid profile or interval");
    }
    const std::int32_t rows = gate_up_rows / 2;
    // g and u live across both projections; each projection's quantized input is scoped to it.
    return 2 * round_up_256(static_cast<std::size_t>(rows) * max_tokens * 2) +
           std::max(detail::t5_workspace_capacity_bytes(gate_up_policy, input_rows, max_tokens),
                    detail::t5_workspace_capacity_bytes(down_policy, rows, max_tokens));
}

void rmsnorm_swiglu_mlp(const RmsNormPrologue& norm, const Weight& gate_up,
                        LinearPolicy gate_up_policy, const Weight& down, LinearPolicy down_policy,
                        Tensor& residual, WorkspaceArena& ws, cudaStream_t stream) {
    if (!rmsnorm_swiglu_mlp_accepts(gate_up.qtype, gate_up_policy, down.qtype, down_policy)) {
        throw std::invalid_argument("rmsnorm_swiglu_mlp: unregistered weight format or policy");
    }
    const std::int32_t hidden = residual.ne[0];
    const std::int32_t tokens = residual.ne[1];
    if (gate_up.n % 2048 || gate_up.k != hidden || down.n != hidden || down.k != gate_up.n / 2) {
        throw std::invalid_argument("rmsnorm_swiglu_mlp: gate/up [2M,D] and down [D,M] expected");
    }
    if (tokens <= 0) { throw std::invalid_argument("rmsnorm_swiglu_mlp: T must be positive"); }
    // The residual is both the normalized input and the accumulated output; the t5 projections
    // validate it, the norm weight and their weights.
    auto scope             = ws.scope();
    Tensor gate            = ws.alloc(DType::BF16, {down.k, tokens});
    Tensor up              = ws.alloc(DType::BF16, {down.k, tokens});
    Tensor* gate_up_rows[] = {&gate, &up};
    detail::t5_project_rmsnorm(residual, norm.weight, norm.eps, norm.unit_offset, gate_up,
                               gate_up_rows, /*accumulate=*/false, gate_up_policy, &ws, stream);
    Tensor* delta[] = {&residual};
    detail::t5_project_swiglu(gate, up, down, delta, /*accumulate=*/true, down_policy, &ws, stream);
}

} // namespace ninfer::ops
