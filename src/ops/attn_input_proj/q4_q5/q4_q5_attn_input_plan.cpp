#include "core/weight.h"
#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.h"

#include "core/layout.h"
#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

bool supported_shape(const Q4Q5AttnInputProblem& problem) noexcept {
    return problem.input_rows == 5120 && problem.query_rows == 6144 && problem.kv_rows == 1024 &&
           problem.padded_k == 5120;
}

// With an A8 permission, prefill widths from here on run the int8 GEMM; narrower extents
// (decode and speculative verification) keep the A16 routes.
constexpr std::int32_t kA8MinCols = 129;

std::size_t a8_workspace_bytes(std::int32_t k, std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_a8_g64_activation(layout, k, cols);
    return layout.peak_bytes(1);
}

} // namespace

const char* q4_q5_attn_input_schedule_name(Q4Q5AttnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4Q5AttnInputScheduleId::ParentSplitFixed:
        return "attn_input_proj.q4_q5.parent_split_fixed";
    case Q4Q5AttnInputScheduleId::MixedR32C32S2:
        return "attn_input_proj.q4_q5.mixed.r32.c32.s2";
    case Q4Q5AttnInputScheduleId::MixedR32C64S3:
        return "attn_input_proj.q4_q5.mixed.r32.c64.s3";
    case Q4Q5AttnInputScheduleId::PairR32C64S3:
        return "attn_input_proj.q4_q5.pair.r32.c64.s3";
    case Q4Q5AttnInputScheduleId::MixedPipelinedR128C128:
        return "attn_input_proj.q4_q5.mixed.pipelined.r128.c128";
    case Q4Q5AttnInputScheduleId::PairR32C64S4:
        return "attn_input_proj.q4_q5.pair.r32.c64.s4";
    case Q4Q5AttnInputScheduleId::A8MixedPipelinedR128C128:
        return "attn_input_proj.q4_q5.a8.mixed.pipelined.r128.c128";
    }
    return "attn_input_proj.q4_q5.unknown";
}

bool q4_q5_attn_input_admits(const Q4Q5AttnInputProblem& problem) noexcept {
    switch (problem.policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return supported_shape(problem) && problem.cols >= 1;
    }
    return false;
}

Q4Q5AttnInputPlan q4_q5_attn_input_resolve_plan(const Q4Q5AttnInputProblem& problem) {
    if (!q4_q5_attn_input_admits(problem)) {
        throw std::invalid_argument(
            "Q4/Q5 attention input: exact problem or column count is not admitted");
    }

    if (allows_a8(problem.policy) && problem.cols >= kA8MinCols) {
        return {Q4Q5AttnInputScheduleId::A8MixedPipelinedR128C128,
                a8_workspace_bytes(problem.input_rows, problem.cols)};
    }
    if (problem.cols <= 16) return {Q4Q5AttnInputScheduleId::ParentSplitFixed};
    if (problem.cols <= 32) return {Q4Q5AttnInputScheduleId::MixedR32C32S2};
    if (problem.cols <= 64) return {Q4Q5AttnInputScheduleId::MixedR32C64S3};
    if (problem.cols <= 104) return {Q4Q5AttnInputScheduleId::PairR32C64S3};
    if (problem.cols <= 128 || problem.cols >= 193)
        return {Q4Q5AttnInputScheduleId::MixedPipelinedR128C128};
    return {Q4Q5AttnInputScheduleId::PairR32C64S4};
}

std::size_t q4_q5_attn_input_capacity_workspace_bytes(LinearPolicy policy, std::int32_t min_cols,
                                                      std::int32_t max_cols) {
    if (min_cols <= 0 || max_cols < min_cols) {
        throw std::invalid_argument("Q4/Q5 attention input: invalid column interval");
    }
    const auto plan = [&](std::int32_t cols) {
        return q4_q5_attn_input_resolve_plan({5120, 6144, 1024, 5120, cols, policy});
    };
    (void)plan(min_cols);
    // Only the A8 route has scratch, and it grows with the width.
    return plan(max_cols).workspace_bytes;
}

void q4_q5_attn_input_execute_plan(const Q4Q5AttnInputPlan& plan, const Tensor& x,
                                   const Weight& query_key_weight, const Weight& gate_value_weight,
                                   Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                   LinearPolicy policy, WorkspaceArena* ws, cudaStream_t stream) {
    const Q4Q5AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], query_key_weight.padded_shape[1],
                                       x.ne[1], policy};
    const Q4Q5AttnInputPlan resolved = q4_q5_attn_input_resolve_plan(problem);
    if (resolved.schedule != plan.schedule || resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("Q4/Q5 attention input: plan does not match exact problem");
    }

    switch (plan.schedule) {
    case Q4Q5AttnInputScheduleId::ParentSplitFixed:
        q4_q5_attn_input_small_t_launch(x, query_key_weight, gate_value_weight, q, gate, k, v,
                                        stream);
        return;
    case Q4Q5AttnInputScheduleId::MixedR32C32S2:
        q4_q5_attn_input_mixed_r32_c32_s2_launch(x, query_key_weight, gate_value_weight, q, gate, k,
                                                 v, stream);
        return;
    case Q4Q5AttnInputScheduleId::MixedR32C64S3:
        q4_q5_attn_input_mixed_r32_c64_s3_launch(x, query_key_weight, gate_value_weight, q, gate, k,
                                                 v, stream);
        return;
    case Q4Q5AttnInputScheduleId::PairR32C64S3:
        q4_q5_attn_input_pair_r32_c64_s3_launch(x, query_key_weight, gate_value_weight, q, gate, k,
                                                v, stream);
        return;
    case Q4Q5AttnInputScheduleId::MixedPipelinedR128C128:
        q4_q5_attn_input_mixed_pipelined_r128_c128_launch(x, query_key_weight, gate_value_weight,
                                                          q, gate, k, v, stream);
        return;
    case Q4Q5AttnInputScheduleId::PairR32C64S4:
        q4_q5_attn_input_grouped_mma_r32_c64_s4_launch(x, query_key_weight, gate_value_weight, q,
                                                       gate, k, v, stream);
        return;
    case Q4Q5AttnInputScheduleId::A8MixedPipelinedR128C128: {
        if (ws == nullptr) {
            throw std::invalid_argument("Q4/Q5 attention input: the A8 route requires a workspace");
        }
        auto scratch_scope        = ws->scope();
        A8G64Activation quantized = allocate_a8_g64_activation(*ws, problem.input_rows, problem.cols);
        a8_g64_quantize(x, quantized, stream);
        q4_q5_attn_input_a8_mixed_pipelined_r128_c128_launch(quantized, query_key_weight,
                                                             gate_value_weight, q, gate, k, v,
                                                             stream);
        return;
    }
    }
    throw std::logic_error("Q4/Q5 attention input: unknown schedule");
}

void q4_q5_attn_input_dispatch(const Tensor& x, const Weight& query_key_weight,
                               const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k,
                               Tensor& v, LinearPolicy policy, WorkspaceArena* ws,
                               cudaStream_t stream) {
    const Q4Q5AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], query_key_weight.padded_shape[1],
                                       x.ne[1], policy};
    const Q4Q5AttnInputPlan plan = q4_q5_attn_input_resolve_plan(problem);
    q4_q5_attn_input_execute_plan(plan, x, query_key_weight, gate_value_weight, q, gate, k, v,
                                  policy, ws, stream);
}

} // namespace ninfer::ops::detail
