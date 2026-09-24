#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "ninfer/ops/hadamard.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"

namespace ninfer::models::qwen3_5::execution {

inline void project(const Tensor& input, const LinearParameters& p, Tensor& output,
                    WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();
    ops::linear(input, p.weight, output, p.policy, workspace, stream);
}

// Output heads: a Prism t2 head reads a rotated copy of its input, which other consumers (MTP,
// DFlash taps) still read in the primal basis.
inline void project_head(const Tensor& input, const LinearParameters& p,
                         const InputRotation& rotation, Tensor& output, WorkspaceArena& workspace,
                         cudaStream_t stream) {
    auto scope = workspace.scope();
    if (!rotation) {
        ops::linear(input, p.weight, output, p.policy, workspace, stream);
        return;
    }
    Tensor rotated = workspace.alloc(DType::BF16, {input.ne[0], input.ne[1]});
    ops::hadamard_1024(input, *rotation, rotated, stream);
    ops::linear(rotated, p.weight, output, p.policy, workspace, stream);
}

inline void project_add(const Tensor& input, const LinearParameters& p, Tensor& residual,
                        WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();
    ops::linear_add(input, p.weight, residual, p.policy, workspace, stream);
}

inline void project_swiglu(const Tensor& input, const LinearParameters& p, Tensor& output,
                           WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();
    ops::linear_swiglu(input, p.weight, output, p.policy, workspace, stream);
}

} // namespace ninfer::models::qwen3_5::execution
