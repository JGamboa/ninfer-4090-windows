#include "models/qwen3_5/execution/mtp.h"

#include "core/layout.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/mtp_pack.h"

#include <algorithm>
#include <stdexcept>
#include <variant>

namespace ninfer::models::qwen3_5::execution {
namespace {

std::size_t linear_bytes(const LinearParameters& p, std::int32_t first, std::int32_t last) {
    const auto& w = p.weight;
    return ops::linear_workspace_capacity_bytes(w.qtype, w.n, w.k, p.policy, first, last);
}

} // namespace

std::size_t mtp_projection_workspace_bytes(const MtpProjectionParameters& parameters,
                                           std::int32_t first, std::int32_t last) {
    if (std::holds_alternative<ops::PairedProjectionWeights>(parameters.complete)) {
        if (first <= 0 || last < first) {
            throw std::invalid_argument("MTP projection: invalid column interval");
        }
        return 0; // the paired attention-input routes use no transient storage
    }
    const auto& p = std::get<LinearParameters>(parameters.complete);
    const auto& w = p.weight;
    if (!parameters.rows) {
        return ops::attn_input_proj_workspace_capacity_bytes(w.qtype, w.n, w.k, p.policy, first,
                                                             last);
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {w.n, last});
    (void)layout.alloc_bytes(linear_bytes(p, first, last));
    return layout.peak_bytes(1);
}

std::size_t mtp_kv_workspace_bytes(const MtpProjectionParameters& parameters,
                                   const AttentionConfig& config, std::int32_t first,
                                   std::int32_t last) {
    if (parameters.rows) {
        return std::max(linear_bytes((*parameters.rows)[1], first, last),
                        linear_bytes((*parameters.rows)[3], first, last));
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {dimension(config.query_width()), last});
    (void)layout.alloc(DType::BF16, {dimension(config.query_width()), last});
    (void)layout.alloc_bytes(mtp_projection_workspace_bytes(parameters, first, last));
    return layout.peak_bytes(1);
}

std::size_t mtp_query_gate_workspace_bytes(const MtpProjectionParameters& parameters,
                                           const AttentionConfig& config, std::int32_t first,
                                           std::int32_t last) {
    if (parameters.rows) {
        return std::max(linear_bytes((*parameters.rows)[0], first, last),
                        linear_bytes((*parameters.rows)[2], first, last));
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {dimension(config.key_width()), last});
    (void)layout.alloc(DType::BF16, {dimension(config.key_width()), last});
    (void)layout.alloc_bytes(mtp_projection_workspace_bytes(parameters, first, last));
    return layout.peak_bytes(1);
}

void mtp_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                    const AttentionConfig& config, Tensor& query, Tensor& gate, Tensor& key,
                    Tensor& value, WorkspaceArena& workspace, cudaStream_t stream) {
    if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&parameters.complete)) {
        ops::attn_input_proj(hidden, pair->first, pair->second, query, gate, key, value, stream);
        return;
    }
    const auto& p = std::get<LinearParameters>(parameters.complete);
    if (!parameters.rows) {
        ops::attn_input_proj(hidden, p.weight, query, gate, key, value, p.policy, workspace,
                             stream);
        return;
    }
    auto scope         = workspace.scope();
    const auto columns = hidden.ne[1];
    Tensor packed      = workspace.alloc(DType::BF16, {p.weight.n, columns});
    ops::linear(hidden, p.weight, packed, p.policy, workspace, stream);
    Tensor q =
        query.view({dimension(config.head_dim), dimension(config.num_attention_heads), columns});
    Tensor k =
        key.view({dimension(config.head_dim), dimension(config.num_key_value_heads), columns});
    Tensor g =
        gate.view({dimension(config.head_dim), dimension(config.num_attention_heads), columns});
    Tensor v =
        value.view({dimension(config.head_dim), dimension(config.num_key_value_heads), columns});
    ops::mtp_split_attn_in(packed, q, k, g, v, stream);
}

void mtp_kv_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                       const AttentionConfig& config, Tensor& key, Tensor& value,
                       WorkspaceArena& workspace, cudaStream_t stream) {
    if (parameters.rows) {
        // Row views of whichever parent holds K and V: one Linear each, in every stored format.
        const auto& k = (*parameters.rows)[1];
        const auto& v = (*parameters.rows)[3];
        {
            auto scope = workspace.scope();
            ops::linear(hidden, k.weight, key, k.policy, workspace, stream);
        }
        auto scope = workspace.scope();
        ops::linear(hidden, v.weight, value, v.policy, workspace, stream);
        return;
    }
    auto scope   = workspace.scope();
    Tensor query = workspace.alloc(DType::BF16, {dimension(config.query_width()), hidden.ne[1]});
    Tensor gate  = workspace.alloc(DType::BF16, {dimension(config.query_width()), hidden.ne[1]});
    mtp_projection(hidden, parameters, config, query, gate, key, value, workspace, stream);
}

void mtp_query_gate_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                               const AttentionConfig& config, Tensor& query, Tensor& gate,
                               WorkspaceArena& workspace, cudaStream_t stream) {
    if (parameters.rows) {
        const auto& q = (*parameters.rows)[0];
        const auto& g = (*parameters.rows)[2];
        {
            auto scope = workspace.scope();
            ops::linear(hidden, q.weight, query, q.policy, workspace, stream);
        }
        auto scope = workspace.scope();
        ops::linear(hidden, g.weight, gate, g.policy, workspace, stream);
        return;
    }
    auto scope   = workspace.scope();
    Tensor key   = workspace.alloc(DType::BF16, {dimension(config.key_width()), hidden.ne[1]});
    Tensor value = workspace.alloc(DType::BF16, {dimension(config.key_width()), hidden.ne[1]});
    mtp_projection(hidden, parameters, config, query, gate, key, value, workspace, stream);
}

} // namespace ninfer::models::qwen3_5::execution
