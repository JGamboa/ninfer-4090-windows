#include "core/weight.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h"

#include "core/layout.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct ColsSet {
    std::int32_t first;
    std::int32_t last;

    constexpr bool contains(std::int32_t cols) const noexcept {
        return cols >= first && cols <= last;
    }
};

struct RouteSpec {
    ColsSet cols;
    Q4Q5GdnInputScheduleId schedule;
};

constexpr std::array<RouteSpec, 4> kRoutes{{
    {{1, 16}, Q4Q5GdnInputScheduleId::IndependentDirectFixed},
    {{17, 32}, Q4Q5GdnInputScheduleId::GroupedMixedMmaR32C32S2},
    {{33, 64}, Q4Q5GdnInputScheduleId::GroupedMixedMmaR32C64S4},
    {{65, kAnyCols}, Q4Q5GdnInputScheduleId::GroupedMixedPipelinedR128C128},
}};

constexpr bool catalog_is_closed() noexcept {
    bool closed = kRoutes[0].cols.first == 1;
    for (std::size_t index = 0; index + 1 < kRoutes.size(); ++index) {
        closed = closed && (kRoutes[index].cols.last + 1 == kRoutes[index + 1].cols.first);
    }
    return closed && kRoutes[kRoutes.size() - 1].cols.last == kAnyCols;
}

static_assert(catalog_is_closed(), "GDN input routes must be exact and closed");

// With an A8 permission, prefill widths from here on run the int8 GEMM; narrower extents
// (decode and speculative verification) keep the A16 routes.
constexpr std::int32_t kA8MinCols = 129;

std::size_t a8_workspace_bytes(std::int32_t k, std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_a8_g64_activation(layout, k, cols);
    return layout.peak_bytes(1);
}

bool supported_shape(const Q4Q5GdnInputProblem& problem) noexcept {
    return problem.input_rows == 5120 && problem.qk_rows == 4096 && problem.value_z_rows == 12288 &&
           problem.qkv_rows == 10240 && problem.z_rows == 6144 && problem.padded_k == 5120;
}

} // namespace

const char* q4_q5_gdn_input_schedule_name(Q4Q5GdnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4Q5GdnInputScheduleId::IndependentDirectFixed:
        return "gdn_input_proj.q4_q5.independent_direct_fixed";
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR32C32S2:
        return "gdn_input_proj.q4_q5.grouped_mixed.mma.r32.c32.s2";
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR32C64S4:
        return "gdn_input_proj.q4_q5.grouped_mixed.mma.r32.c64.s4";
    case Q4Q5GdnInputScheduleId::GroupedMixedPipelinedR128C128:
        return "gdn_input_proj.q4_q5.grouped_mixed.mma.pipelined.r128.c128";
    case Q4Q5GdnInputScheduleId::A8GroupedMixedPipelinedR128C128:
        return "gdn_input_proj.q4_q5.a8.grouped_mixed.mma.pipelined.r128.c128";
    }
    return "gdn_input_proj.q4_q5.unknown";
}

const char* q4_q5_gdn_input_conv_schedule_name(Q4Q5GdnInputConvScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4Q5GdnInputConvScheduleId::ProjectionEpilogueFused:
        return "gdn_input_proj_conv.q4_q5.projection_epilogue_fused";
    case Q4Q5GdnInputConvScheduleId::Materialized:
        return "gdn_input_proj_conv.q4_q5.materialized";
    }
    return "gdn_input_proj_conv.q4_q5.unknown";
}

bool q4_q5_gdn_input_admits(const Q4Q5GdnInputProblem& problem) noexcept {
    switch (problem.policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return supported_shape(problem) && problem.cols >= 1;
    }
    return false;
}

Q4Q5GdnInputPlan q4_q5_gdn_input_resolve_plan(const Q4Q5GdnInputProblem& problem) {
    if (!q4_q5_gdn_input_admits(problem)) {
        throw std::invalid_argument(
            "Q4/Q5 GDN input: exact problem or column count is not admitted");
    }
    if (allows_a8(problem.policy) && problem.cols >= kA8MinCols) {
        return {Q4Q5GdnInputScheduleId::A8GroupedMixedPipelinedR128C128,
                a8_workspace_bytes(problem.input_rows, problem.cols)};
    }

    for (const RouteSpec& route : kRoutes) {
        if (!route.cols.contains(problem.cols)) { continue; }
        return {route.schedule};
    }
    throw std::logic_error("Q4/Q5 GDN input: admitted problem has no covering route");
}

Q4Q5GdnInputConvPlan q4_q5_gdn_input_conv_resolve_plan(const Q4Q5GdnInputProblem& problem,
                                                       std::int32_t batch_size) {
    if (!q4_q5_gdn_input_admits(problem) || batch_size <= 0 || batch_size > 8) {
        throw std::invalid_argument(
            "Q4/Q5 GDN input conv: exact problem or column count is not admitted");
    }
    if (batch_size > 1) { return {Q4Q5GdnInputConvScheduleId::Materialized}; }
    switch (problem.cols) {
    case 1:
    case 2:
    case 3:
    case 5:
    case 6:
        return {Q4Q5GdnInputConvScheduleId::ProjectionEpilogueFused};
    default:
        return {Q4Q5GdnInputConvScheduleId::Materialized};
    }
}

std::size_t q4_q5_gdn_input_capacity_workspace_bytes(LinearPolicy policy, std::int32_t min_cols,
                                                     std::int32_t max_cols) {
    if (min_cols <= 0 || max_cols < min_cols) {
        throw std::invalid_argument("Q4/Q5 GDN input: invalid column interval");
    }
    const auto plan = [&](std::int32_t cols) {
        return q4_q5_gdn_input_resolve_plan({5120, 4096, 12288, 10240, 6144, 5120, cols, policy});
    };
    (void)plan(min_cols);
    // Only the A8 route has scratch, and it grows with the width.
    return plan(max_cols).workspace_bytes;
}

void q4_q5_gdn_input_execute_plan(const Q4Q5GdnInputPlan& plan, const Tensor& x,
                                  const Weight& qk_weight, const Weight& value_z_weight,
                                  Tensor& qkv, Tensor& z, LinearPolicy policy, WorkspaceArena* ws,
                                  cudaStream_t stream) {
    const Q4Q5GdnInputProblem problem{x.ne[0],   qk_weight.n, value_z_weight.n,
                                      qkv.ne[0], z.ne[0],     qk_weight.padded_shape[1],
                                      x.ne[1],   policy};
    const Q4Q5GdnInputPlan resolved = q4_q5_gdn_input_resolve_plan(problem);
    if (resolved.schedule != plan.schedule || resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("Q4/Q5 GDN input: plan does not match exact problem");
    }

    switch (plan.schedule) {
    case Q4Q5GdnInputScheduleId::IndependentDirectFixed: {
        Tensor qk    = qkv.slice(0, 0, problem.qk_rows);
        Tensor value = qkv.slice(0, problem.qk_rows, problem.z_rows);
        q4_q5_gdn_input_independent_launch(x, qk_weight, value_z_weight, qk, value, z, stream);
        return;
    }
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR32C32S2:
    case Q4Q5GdnInputScheduleId::GroupedMixedMmaR32C64S4:
    case Q4Q5GdnInputScheduleId::GroupedMixedPipelinedR128C128:
        q4_q5_gdn_input_grouped_mma_launch(x, qk_weight, value_z_weight, qkv, z, plan.schedule,
                                           stream);
        return;
    case Q4Q5GdnInputScheduleId::A8GroupedMixedPipelinedR128C128: {
        if (ws == nullptr) {
            throw std::invalid_argument("Q4/Q5 GDN input: the A8 route requires a workspace");
        }
        auto scratch_scope        = ws->scope();
        A8G64Activation quantized = allocate_a8_g64_activation(*ws, problem.input_rows, problem.cols);
        a8_g64_quantize(x, quantized, stream);
        q4_q5_gdn_input_a8_grouped_mma_launch(quantized, qk_weight, value_z_weight, qkv, z, stream);
        return;
    }
    }
    throw std::logic_error("Q4/Q5 GDN input: unknown schedule");
}

void q4_q5_gdn_input_dispatch(const Tensor& x, const Weight& qk_weight,
                              const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                              LinearPolicy policy, WorkspaceArena* ws, cudaStream_t stream) {
    const Q4Q5GdnInputProblem problem{x.ne[0],   qk_weight.n, value_z_weight.n,
                                      qkv.ne[0], z.ne[0],     qk_weight.padded_shape[1],
                                      x.ne[1],   policy};
    const Q4Q5GdnInputPlan plan = q4_q5_gdn_input_resolve_plan(problem);
    q4_q5_gdn_input_execute_plan(plan, x, qk_weight, value_z_weight, qkv, z, policy, ws, stream);
}

} // namespace ninfer::ops::detail
