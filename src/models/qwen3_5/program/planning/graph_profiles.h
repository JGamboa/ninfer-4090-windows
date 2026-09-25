#pragma once
#include "models/qwen3_5/program/program.h"
#include "ninfer/ops/softmax_attention.h"

namespace ninfer::models::qwen3_5::detail {

[[nodiscard]] std::vector<GraphExecutionProfile> ordinary_graph_profiles(std::uint32_t capacity);
[[nodiscard]] std::vector<GraphExecutionProfile> mtp_graph_profiles(std::uint32_t capacity,
                                                                    std::uint32_t draft_window);
// MTP rounds verifying the n-gram window V (width V+1) at one batch size. At that width the
// attention route of target verify and MTP alignment changes with the batch, the KV storage and the
// visible keys, so each profile's topology class is the attention Op's class at its envelope.
[[nodiscard]] ops::AttentionHeadGeometry
text_attention_geometry(const execution::Parameters& parameters);
[[nodiscard]] std::vector<GraphExecutionProfile>
mtp_wide_graph_profiles(std::uint32_t capacity, std::uint32_t draft_window,
                        std::uint32_t verify_window, std::uint32_t batch_size,
                        ops::AttentionHeadGeometry attention, KvCacheStorage kv_storage);
[[nodiscard]] std::vector<GraphExecutionProfile> dflash_graph_profiles(SpeculativeBackend backend,
                                                                       std::uint32_t capacity,
                                                                       std::uint32_t draft_window,
                                                                       std::uint32_t batch_size);

} // namespace ninfer::models::qwen3_5::detail
