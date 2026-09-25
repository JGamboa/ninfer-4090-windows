#include "models/qwen3_5/program/planning/graph_profiles.h"
#include "models/qwen3_5/execution/parameters.h"
#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {
namespace {
std::vector<GraphExecutionProfile>
graph_profiles_through(std::uint32_t max_frontier,
                       const std::vector<std::uint32_t>& preferred_ends) {
    std::vector<GraphExecutionProfile> out;
    std::uint32_t begin = 0;
    for (const std::uint32_t preferred_end : preferred_ends) {
        if (begin > max_frontier) { break; }
        const std::uint32_t end = std::min(preferred_end, max_frontier);
        out.push_back({begin, end});
        if (end == max_frontier) { return out; }
        begin = end + 1;
    }
    if (begin <= max_frontier) { out.push_back({begin, max_frontier}); }
    return out;
}

std::vector<GraphExecutionProfile> dflash_base_profiles(std::uint32_t capacity,
                                                        std::uint32_t draft_window) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    const std::uint32_t block        = draft_window + 1;
    const std::uint32_t max_frontier = capacity - 1;
    std::vector<std::uint32_t> ends{
        96U, 127U, 511U, 1023U, 2047U, 4095U, 8191U, 16383U, 32767U, 65536U, 131072U, 196608U,
    };
    const auto add_target_boundary = [&](std::uint32_t visible_end) {
        if (visible_end >= block) { ends.push_back(visible_end - block); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_target_boundary(visible_end);
    }
    if (draft_window >= 6 && draft_window <= 15) {
        add_target_boundary(draft_window <= 11 ? 512U : 1024U);
    }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    return graph_profiles_through(max_frontier, ends);
}

bool dflash_target_uses_chunked_small_t(std::uint32_t draft_window, std::uint32_t batch_size,
                                        std::uint32_t max_visible_keys) {
    const std::uint32_t tokens = draft_window + 1;
    if (tokens <= 6) { return false; }
    if (batch_size > 1) { return true; }
    const std::uint32_t prompt_visible_limit = tokens <= 12 ? 512U : 1024U;
    return max_visible_keys > prompt_visible_limit;
}

} // namespace

std::vector<GraphExecutionProfile> ordinary_graph_profiles(std::uint32_t capacity) {
    // E+1 is the one-token visible window. Early ranges limit empty producer CTAs; later ranges
    // follow measured split-policy transitions until the producer grid reaches its fixed cap.
    return graph_profiles_through(capacity - 1, {127, 511, 2047, 4095, 8197, 16389, 32767});
}

namespace {

// Profile ends for rounds verifying `verify_drafts` drafts with MTP depth K.
std::vector<std::uint32_t> mtp_profile_ends(std::uint32_t draft_window,
                                            std::uint32_t verify_drafts) {
    // Bound the final AR window E+V+K at split-policy transitions until the grid reaches its cap.
    std::vector<std::uint32_t> ends;
    const auto add_shifted = [&](std::uint32_t visible_end, std::uint32_t offset) {
        if (visible_end >= offset) { ends.push_back(visible_end - offset); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_shifted(visible_end, verify_drafts + draft_window);
    }
    if (verify_drafts != draft_window) { return ends; }
    // Target verify and MTP batch both have T=K+1 and W=E+K+1. Preserve one concrete INT8
    // implementation per range at the T=4/5/6 launch boundaries.
    if (draft_window == 3) {
        add_shifted(1029, draft_window + 1);
    } else if (draft_window == 4) {
        for (const std::uint32_t visible_end : {128U, 512U, 1029U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    } else if (draft_window == 5) {
        for (const std::uint32_t visible_end : {128U, 160U, 2054U, 8198U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    }
    return ends;
}

} // namespace

std::vector<GraphExecutionProfile> mtp_graph_profiles(std::uint32_t capacity,
                                                      std::uint32_t draft_window) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    std::vector<std::uint32_t> ends = mtp_profile_ends(draft_window, draft_window);
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    return graph_profiles_through(capacity - 1, ends);
}

ops::AttentionHeadGeometry text_attention_geometry(const execution::Parameters& parameters) {
    const auto& attention = parameters.model.config().text.attention;
    if (!attention) { throw std::logic_error("Qwen3.5 text model has no attention geometry"); }
    return {static_cast<std::int32_t>(attention->head_dim),
            static_cast<std::int32_t>(attention->num_attention_heads),
            static_cast<std::int32_t>(attention->num_key_value_heads)};
}

std::vector<GraphExecutionProfile> mtp_wide_graph_profiles(std::uint32_t capacity,
                                                           std::uint32_t draft_window,
                                                           std::uint32_t verify_window,
                                                           std::uint32_t batch_size,
                                                           ops::AttentionHeadGeometry attention,
                                                           KvCacheStorage kv_storage) {
    if (capacity == 0 || draft_window == 0 || verify_window <= draft_window ||
        verify_window > 15 || batch_size == 0) {
        throw std::invalid_argument("invalid MTP wide graph dimensions");
    }
    const std::uint32_t block = verify_window + 1;
    const auto topology = [&](std::uint32_t frontier) {
        // Target verify and MTP alignment both see E+V+1 keys at width V+1.
        const auto visible = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(capacity, static_cast<std::uint64_t>(frontier) + block));
        return ops::causal_softmax_attention_topology_class(
            attention, kv_storage, {1, visible}, static_cast<std::int32_t>(block),
            static_cast<std::int32_t>(batch_size));
    };
    std::vector<std::uint32_t> ends = mtp_profile_ends(draft_window, verify_window);
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    // Split a range where the attention class changes (it changes once, at a visible-key limit),
    // so short contexts keep their cheaper route instead of the class at the range's maximum.
    std::vector<std::uint32_t> split_ends;
    for (const GraphExecutionProfile range : graph_profiles_through(capacity - 1, ends)) {
        if (topology(range.min) != topology(range.max)) {
            std::uint32_t low = range.min;
            std::uint32_t high = range.max;
            while (low + 1 < high) {
                const std::uint32_t middle = low + (high - low) / 2;
                (topology(middle) == topology(range.min) ? low : high) = middle;
            }
            split_ends.push_back(low);
        }
        split_ends.push_back(range.max);
    }
    std::vector<GraphExecutionProfile> profiles = graph_profiles_through(capacity - 1, split_ends);
    for (GraphExecutionProfile& profile : profiles) {
        profile.topology_class = topology(profile.max);
    }
    return profiles;
}

std::vector<GraphExecutionProfile> dflash_graph_profiles(SpeculativeBackend backend,
                                                         std::uint32_t capacity,
                                                         std::uint32_t draft_window,
                                                         std::uint32_t batch_size) {
    if (capacity == 0 || draft_window == 0 || draft_window > 15) {
        throw std::invalid_argument("invalid masked draft graph dimensions");
    }
    if (backend == SpeculativeBackend::DFlash2) {
        auto profiles = graph_profiles_through(capacity - 1, {96, 511, 2047, 8191, 32767});
        for (std::size_t i = 0; i < profiles.size(); ++i) {
            profiles[i].topology_class = static_cast<std::uint32_t>(i);
        }
        return profiles;
    }
    std::vector<GraphExecutionProfile> profiles = dflash_base_profiles(capacity, draft_window);
    for (GraphExecutionProfile& profile : profiles) {
        const std::uint32_t target_max = static_cast<std::uint32_t>(std::min<std::uint64_t>(
            capacity, static_cast<std::uint64_t>(profile.max) + draft_window + 1ULL));
        const bool split_swa           = profile.max > 96U;
        const bool chunked_target =
            dflash_target_uses_chunked_small_t(draft_window, batch_size, target_max);
        profile.topology_class = (chunked_target ? 2U : 0U) | (split_swa ? 1U : 0U);
    }
    return profiles;
}

} // namespace ninfer::models::qwen3_5::detail
