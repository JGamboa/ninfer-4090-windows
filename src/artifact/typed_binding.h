#pragma once

#include "artifact/binder.h"
#include "core/tensor.h"

#include <cstdint>
#include <initializer_list>
#include <span>
#include <string_view>

namespace ninfer::artifact {

class MaterializedArtifact;

[[nodiscard]] ObjectHandle bind_tensor(Binder& binder, std::string_view name, NumericFormat format,
                                       std::initializer_list<std::uint64_t> shape,
                                       TensorPlacement placement);

// Like bind_tensor, but for a tensor that a v2 artifact stores as one whole object
// (`v2_name`) while a v3 artifact may instead split across several logical binding names
// that jointly cover it (`v3_leaf_names`, order does not matter). Tries `v2_name` first;
// only consults `v3_leaf_names` (via Binder::require_tensor_fused) when that name doesn't
// exist, so this works unchanged against either artifact version.
[[nodiscard]] ObjectHandle bind_tensor_fused(Binder& binder, std::string_view v2_name,
                                             std::span<const std::string_view> v3_leaf_names,
                                             NumericFormat format,
                                             std::initializer_list<std::uint64_t> shape,
                                             TensorPlacement placement);

[[nodiscard]] ObjectHandle bind_device_tensor(Binder& binder, std::string_view name,
                                              NumericFormat format,
                                              std::initializer_list<std::uint64_t> shape);

[[nodiscard]] ObjectHandle bind_raw_resource(Binder& binder, std::string_view name);

[[nodiscard]] Tensor materialized_tensor(const MaterializedArtifact& materialized,
                                         ObjectHandle handle, NumericFormat format,
                                         std::initializer_list<std::int32_t> internal_shape);

[[nodiscard]] Weight materialized_weight(const MaterializedArtifact& materialized,
                                         ObjectHandle handle, NumericFormat format,
                                         std::int32_t rows, std::int32_t columns);

} // namespace ninfer::artifact
