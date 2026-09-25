#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "ninfer/ops/rmsnorm.h"

namespace ninfer::models::qwen3_5::execution {

[[nodiscard]] std::size_t ffn_workspace_bytes(const FfnParameters& parameters, std::int32_t first,
                                              std::int32_t last, bool mtp = false);
void ffn(const Tensor& hidden, const FfnParameters& parameters, Tensor& residual,
         const ops::SparseMoeHints& hints, WorkspaceArena& workspace, cudaStream_t stream,
         bool mtp = false);

// Whether the dense FFN registers rmsnorm_swiglu_mlp, which normalizes the residual itself
// instead of reading a materialized normalized hidden state. ffn_workspace_bytes (mtp = false)
// then sizes normalized_ffn.
[[nodiscard]] bool ffn_fuses_rmsnorm(const FfnParameters& parameters);
// residual += FFN(rmsnorm(residual, norm)).
void normalized_ffn(const ops::RmsNormPrologue& norm, const FfnParameters& parameters,
                    Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::models::qwen3_5::execution
