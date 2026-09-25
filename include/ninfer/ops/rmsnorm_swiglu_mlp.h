#pragma once

// ninfer::ops - residual += Down(SwiGLU(GateUp(RMSNorm(residual)))), one dense MLP block.

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/rmsnorm.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Whether rmsnorm_swiglu_mlp is registered for these weight formats and policies: both weights
 * T5_G128_FP16 and both policies allowing A8. Other dense MLPs compose rmsnorm(),
 * linear_swiglu() and linear_add().
 */
[[nodiscard]] bool rmsnorm_swiglu_mlp_accepts(QType gate_up_qtype, LinearPolicy gate_up_policy,
                                              QType down_qtype, LinearPolicy down_policy);

/**
 * Returns the transient capacity of rmsnorm_swiglu_mlp for every T in the inclusive
 * [min_tokens,max_tokens] interval, for a gate/up parent [gate_up_rows,input_rows] and its down
 * projection [input_rows,gate_up_rows/2]. Unregistered profiles or invalid intervals throw.
 */
[[nodiscard]] std::size_t rmsnorm_swiglu_mlp_workspace_capacity_bytes(
    QType qtype, std::int32_t gate_up_rows, std::int32_t input_rows, LinearPolicy gate_up_policy,
    LinearPolicy down_policy, std::int32_t min_tokens, std::int32_t max_tokens);

/**
 * Op: rmsnorm_swiglu_mlp
 *
 * Math / indexing, for the BF16 residual [D,T] and M = gate_up_rows / 2:
 *   n[:,t]     = rmsnorm(residual, norm)[:,t]            (RmsNormPrologue, rmsnorm() semantics)
 *   g[i,t]     = Linear(n, gate_up)[i,t],  u[i,t] = Linear(n, gate_up)[M + i,t]
 *   a[i,t]     = SiLU(g[i,t]) * u[i,t]
 *   ideal[:,t] = residual[:,t] + Linear(a, down)[:,t].
 *
 * Registered domain (rmsnorm_swiglu_mlp_accepts()):
 *   T5_G128_FP16 TernaryRowK128 gate_up [2M,D] (gate rows [0,M) before up rows [M,2M)) and
 *   down [D,M], D and M multiples of 1024, both under AllowA8/AllowA4 (Bonsai: D = 5120,
 *   M = 17408). A weight carrying input signs is stored in the Prism-rotated basis and rotates
 *   its input itself. T may be any positive value. The residual is contiguous, 16-byte aligned
 *   BF16 [D,T]; the norm weight is contiguous BF16 [D].
 *
 * Numeric:
 *   The oracle evaluates n, g, u, a and ideal naively in FP64 from the represented residual,
 *   norm weight and decoded weights. n and a are private arithmetic, not semantic rounding
 *   boundaries: each is evaluated inside the int8 quantization of the following projection and
 *   never rounded to BF16. g and u are staged in BF16. The updated BF16 residual is compared
 *   with ideal under the A8 criterion (activation quantization plus BF16 storage).
 *
 * Effects:
 *   Updates the full residual in place; the weights, the norm weight and the workspace must not
 *   overlap it.
 *
 * Workspace:
 *   Caller-owned transient storage reported by rmsnorm_swiglu_mlp_workspace_capacity_bytes(),
 *   scoped to the call: g and u, and the quantized activation of each projection. There is no
 *   persistent state side effect.
 */
void rmsnorm_swiglu_mlp(const RmsNormPrologue& norm, const Weight& gate_up,
                        LinearPolicy gate_up_policy, const Weight& down, LinearPolicy down_policy,
                        Tensor& residual, WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::ops
