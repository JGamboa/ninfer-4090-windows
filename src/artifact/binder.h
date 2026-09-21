#pragma once

#include "artifact/reader.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::artifact {

enum class TensorPlacement : std::uint8_t {
    Device,
    ValidateOnly,
};

struct ObjectHandle {
    std::size_t index = 0;
};

struct DeviceMaterialization {
    ObjectHandle object;
    std::uint64_t offset    = 0;
    std::uint64_t bytes     = 0;
    std::uint64_t alignment = 0;
};

struct HostMaterialization {
    ObjectHandle object;
};

struct MaterializationPlan {
    std::size_t object_count            = 0;
    std::uint64_t device_capacity_bytes = 0;
    std::vector<DeviceMaterialization> device_objects;
    std::vector<HostMaterialization> host_objects;
};

class Binder {
public:
    explicit Binder(const Reader& reader);

    [[nodiscard]] bool has_object(std::string_view name) const noexcept;

    ObjectHandle require_tensor(std::string_view name, NumericFormat format, StorageLayout layout,
                                std::span<const std::uint64_t> shape);
    ObjectHandle require_resource(std::string_view name, ResourceEncoding encoding);

    // Like require_tensor, but resolves against a group of logical binding names that
    // together make up one physical tensor. On a v2 artifact, or a v3 artifact where the
    // whole tensor has one binding name, pass a single name and this behaves exactly like
    // require_tensor. On a v3 artifact where a converter split the tensor across several
    // logical parameters that each cover a contiguous row range (e.g. a fused Q/K
    // projection exposed as separate "attention/query" and "attention/key" bindings), pass
    // all of their names (order does not matter) to get back the one underlying tensor.
    ObjectHandle require_tensor_fused(std::span<const std::string_view> names,
                                      NumericFormat format, StorageLayout layout,
                                      std::span<const std::uint64_t> shape);

    [[nodiscard]] bool contains(std::string_view name) const noexcept;
    const ObjectDescriptor& descriptor(ObjectHandle handle) const;
    PayloadSpan payload(ObjectHandle handle) const;
    void materialize_on_device(ObjectHandle handle);
    void retain_on_host(ObjectHandle handle);
    void validate_only(ObjectHandle handle);
    MaterializationPlan finish();

private:
    ObjectHandle find_unconsumed(std::string_view name);

    const Reader& reader_;
    std::vector<bool> consumed_;
    std::vector<bool> planned_;
    MaterializationPlan materialization_;
};

} // namespace ninfer::artifact
