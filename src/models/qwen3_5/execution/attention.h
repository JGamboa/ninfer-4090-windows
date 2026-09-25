#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "ninfer/ops/rmsnorm.h"

namespace ninfer::models::qwen3_5::execution {

[[nodiscard]] std::size_t
attention_projection_workspace_bytes(const AttentionParameters& parameters, std::int32_t first,
                                     std::int32_t last);
void attention_projection(const Tensor& hidden, const AttentionParameters& parameters,
                          Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                          WorkspaceArena& workspace, cudaStream_t stream);

// Whether the projection registers the RMSNorm-input form, which normalizes the raw residual
// rows itself instead of reading a materialized normalized hidden state.
[[nodiscard]] bool attention_projection_fuses_rmsnorm(const AttentionParameters& parameters);
void attention_projection(const Tensor& x, const ops::RmsNormPrologue& norm,
                          const AttentionParameters& parameters, Tensor& query, Tensor& gate,
                          Tensor& key, Tensor& value, WorkspaceArena& workspace,
                          cudaStream_t stream);

void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query,
               cudaStream_t stream);
void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query, Tensor& key,
               cudaStream_t stream);

} // namespace ninfer::models::qwen3_5::execution
