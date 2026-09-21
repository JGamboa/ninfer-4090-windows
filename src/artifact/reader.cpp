#include "artifact/reader.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <functional>
#include <limits>
#include <span>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ninfer::artifact {
namespace {

using Json = nlohmann::json;

constexpr std::array<std::byte, 8> kMagicV2 = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'F'},
    std::byte{'E'}, std::byte{'R'}, std::byte{0},   std::byte{2},
};
constexpr std::array<std::byte, 8> kMagicV3 = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'F'},
    std::byte{'E'}, std::byte{'R'}, std::byte{0},   std::byte{3},
};
constexpr std::uint64_t kPrefixBytesV2     = 16;
constexpr std::uint64_t kEntryHeaderBytesV3 = 32;
constexpr std::uint64_t kPayloadAlignment  = 4096;

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, std::string_view label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw ArtifactError(std::string(label) + " overflows u64");
    }
    return a + b;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, std::string_view label) {
    const auto biased = checked_add(value, alignment - 1, label);
    return biased / alignment * alignment;
}

std::uint64_t tensor_elements(std::span<const std::uint64_t> shape) noexcept {
    std::uint64_t total = 1;
    for (const auto dim : shape) { total *= dim; }
    return total;
}

std::uint64_t read_u64_le(const std::byte* data) noexcept {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) {
        value |= std::uint64_t(std::to_integer<unsigned char>(data[i])) << (i * 8);
    }
    return value;
}

template <std::size_t N>
void require_members(const Json& value, const std::array<const char*, N>& members,
                     std::string_view label) {
    if (!value.is_object() || value.size() != N) {
        throw ArtifactError(std::string(label) + " has missing or extra members");
    }
    for (const char* member : members) {
        if (!value.contains(member)) {
            throw ArtifactError(std::string(label) + " has missing or extra members");
        }
    }
}

const std::string& require_string(const Json& value, std::string_view label) {
    if (!value.is_string()) {
        throw ArtifactError(std::string(label) + " must be a nonempty string");
    }
    const auto& result = value.get_ref<const std::string&>();
    if (result.empty()) { throw ArtifactError(std::string(label) + " must be a nonempty string"); }
    return result;
}

std::uint64_t require_unsigned(const Json& value, std::string_view label, bool positive) {
    if (!value.is_number_unsigned()) {
        throw ArtifactError(std::string(label) + " must be an integer");
    }
    const auto result = value.get<std::uint64_t>();
    if (positive && result == 0) { throw ArtifactError(std::string(label) + " must be positive"); }
    return result;
}

NumericFormat parse_format(std::string_view name) {
    if (name == "BF16") { return NumericFormat::BF16; }
    if (name == "FP32") { return NumericFormat::FP32; }
    if (name == "I32") { return NumericFormat::I32; }
    if (name == "Q4G64_F16S") { return NumericFormat::Q4G64_F16S; }
    if (name == "Q5G64_F16S") { return NumericFormat::Q5G64_F16S; }
    if (name == "Q6G64_F16S") { return NumericFormat::Q6G64_F16S; }
    if (name == "W8G32_F16S") { return NumericFormat::W8G32_F16S; }
    if (name == "NVFP4") { return NumericFormat::NVFP4; }
    if (name == "FP8_E4M3FN_ROW_BF16S") { return NumericFormat::FP8_E4M3FN_ROW_BF16S; }
    throw ArtifactError("unknown tensor format: " + std::string(name));
}

StorageLayout parse_layout(std::string_view name) {
    if (name == "contiguous-le-v1") { return StorageLayout::ContiguousLeV1; }
    if (name == "row-split-k128-v1") { return StorageLayout::RowSplitK128V1; }
    if (name == "blockscale-k16-m128x4-v1") { return StorageLayout::BlockScaleK16M128x4V1; }
    if (name == "row-scale-v1") { return StorageLayout::RowScaleV1; }
    throw ArtifactError("unknown tensor layout: " + std::string(name));
}

ResourceEncoding parse_encoding(std::string_view name) {
    if (name == "raw-bytes-v1") { return ResourceEncoding::RawBytesV1; }
    throw ArtifactError("unknown resource encoding: " + std::string(name));
}

// v3 spells format/layout/encoding names differently (lowercase snake_case, matching the
// upstream container spec) than this fork's v2 artifacts (kebab-case / enum-like names).
// Both map onto the same NumericFormat/StorageLayout/ResourceEncoding enums.
NumericFormat parse_format_v3(std::string_view name) {
    if (name == "bf16") { return NumericFormat::BF16; }
    if (name == "fp32") { return NumericFormat::FP32; }
    if (name == "int32") { return NumericFormat::I32; }
    if (name == "q4_g64_fp16") { return NumericFormat::Q4G64_F16S; }
    if (name == "q5_g64_fp16") { return NumericFormat::Q5G64_F16S; }
    if (name == "q6_g64_fp16") { return NumericFormat::Q6G64_F16S; }
    if (name == "q8_g32_fp16") { return NumericFormat::W8G32_F16S; }
    if (name == "nvfp4") { return NumericFormat::NVFP4; }
    if (name == "fp8_e4m3fn_row_bf16") { return NumericFormat::FP8_E4M3FN_ROW_BF16S; }
    throw ArtifactError("unknown v3 tensor format: " + std::string(name));
}

StorageLayout parse_layout_v3(std::string_view name) {
    if (name == "contiguous_le_v1") { return StorageLayout::ContiguousLeV1; }
    if (name == "row_split_k128_v1") { return StorageLayout::RowSplitK128V1; }
    if (name == "block_scale_k16_m128x4_v1") { return StorageLayout::BlockScaleK16M128x4V1; }
    if (name == "row_scale_v1") { return StorageLayout::RowScaleV1; }
    throw ArtifactError("unknown v3 tensor layout: " + std::string(name));
}

ResourceEncoding parse_encoding_v3(std::string_view name) {
    if (name == "raw_bytes_v1") { return ResourceEncoding::RawBytesV1; }
    throw ArtifactError("unknown v3 resource encoding: " + std::string(name));
}

TensorDescriptor parse_tensor_v3(const Json& value, const std::string& id) {
    const auto format = parse_format_v3(require_string(value.at("format"), "tensor format"));
    const auto layout = parse_layout_v3(require_string(value.at("layout"), "tensor layout"));
    const auto offset = require_unsigned(value.at("offset"), "tensor offset", false);
    const auto stored_size = require_unsigned(value.at("bytes"), "tensor bytes", true);

    const auto& raw_shape = value.at("shape");
    if (!raw_shape.is_array()) { throw ArtifactError("tensor shape must be an array"); }
    std::vector<std::uint64_t> shape;
    shape.reserve(raw_shape.size());
    for (const auto& dim : raw_shape) {
        shape.push_back(require_unsigned(dim, "shape dimension", true));
    }

    const auto expected_size = tensor_encoded_size(layout, format, shape);
    if (stored_size != expected_size) {
        throw ArtifactError("object " + id + " stores " + std::to_string(stored_size) +
                            " bytes; layout requires " + std::to_string(expected_size));
    }
    return {id, std::move(shape), format, layout, offset, stored_size};
}

ResourceDescriptor parse_resource_v3(const Json& value, const std::string& id) {
    return {
        id,
        parse_encoding_v3(require_string(value.at("encoding"), "resource encoding")),
        require_unsigned(value.at("offset"), "resource offset", false),
        require_unsigned(value.at("bytes"), "resource bytes", true),
    };
}

ObjectDescriptor parse_object_v3(const Json& value) {
    if (!value.is_object()) { throw ArtifactError("each v3 object entry must be a JSON object"); }
    const auto id = require_string(value.at("id"), "object id");
    const auto it = value.find("kind");
    if (it == value.end() || !it->is_string()) {
        throw ArtifactError("v3 object kind must be 'tensor' or 'resource'");
    }
    const auto& kind = it->get_ref<const std::string&>();
    if (kind == "tensor") { return parse_tensor_v3(value, id); }
    if (kind == "resource") { return parse_resource_v3(value, id); }
    throw ArtifactError("v3 object kind must be 'tensor' or 'resource'");
}

// A v3 "binding" maps one logical parameter name to either a whole physical object, or a
// single contiguous row range within one (this fork's converter never splits a bound
// parameter's element range across more than one part, and never leaves a plain row-major
// tensor's range mid-row; see docs/artifact-v3-port-notes.md).
struct V3BindingRef {
    std::size_t object_index = 0;
    std::uint64_t row_begin  = 0;
    std::uint64_t row_count  = 0;
    bool whole                = false;
};

TensorDescriptor parse_tensor(const Json& value) {
    static constexpr std::array members = {
        "name", "kind", "shape", "format", "layout", "offset", "bytes",
    };
    require_members(value, members, "tensor entry");

    const auto name        = require_string(value.at("name"), "tensor name");
    const auto format      = parse_format(require_string(value.at("format"), "tensor format"));
    const auto layout      = parse_layout(require_string(value.at("layout"), "tensor layout"));
    const auto offset      = require_unsigned(value.at("offset"), "tensor offset", false);
    const auto stored_size = require_unsigned(value.at("bytes"), "tensor bytes", true);

    const auto& raw_shape = value.at("shape");
    if (!raw_shape.is_array()) { throw ArtifactError("tensor shape must be an array"); }
    std::vector<std::uint64_t> shape;
    shape.reserve(raw_shape.size());
    for (const auto& dim : raw_shape) {
        shape.push_back(require_unsigned(dim, "shape dimension", true));
    }

    const auto expected_size = tensor_encoded_size(layout, format, shape);
    if (stored_size != expected_size) {
        throw ArtifactError("tensor " + name + " stores " + std::to_string(stored_size) +
                            " bytes; layout requires " + std::to_string(expected_size));
    }
    return {name, std::move(shape), format, layout, offset, stored_size};
}

ResourceDescriptor parse_resource(const Json& value) {
    static constexpr std::array members = {
        "name", "kind", "encoding", "offset", "bytes",
    };
    require_members(value, members, "resource entry");
    return {
        require_string(value.at("name"), "resource name"),
        parse_encoding(require_string(value.at("encoding"), "resource encoding")),
        require_unsigned(value.at("offset"), "resource offset", false),
        require_unsigned(value.at("bytes"), "resource bytes", true),
    };
}

ObjectDescriptor parse_object(const Json& value) {
    if (!value.is_object()) { throw ArtifactError("each object entry must be a JSON object"); }
    const auto it = value.find("kind");
    if (it == value.end() || !it->is_string()) {
        throw ArtifactError("object kind must be 'tensor' or 'resource'");
    }
    const auto& kind = it->get_ref<const std::string&>();
    if (kind == "tensor") { return parse_tensor(value); }
    if (kind == "resource") { return parse_resource(value); }
    throw ArtifactError("object kind must be 'tensor' or 'resource'");
}

struct TransparentStringHash {
    using is_transparent = void;

    std::size_t operator()(std::string_view value) const noexcept {
        return std::hash<std::string_view>{}(value);
    }

    std::size_t operator()(const std::string& value) const noexcept {
        return (*this)(std::string_view(value));
    }
};

class MappedFile {
public:
    explicit MappedFile(const std::filesystem::path& path) {
#ifdef _WIN32
        mapping_file_ = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (mapping_file_ == INVALID_HANDLE_VALUE) {
            throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
                                    "CreateFileW " + path.string());
        }

        LARGE_INTEGER file_size{};
        if (!::GetFileSizeEx(mapping_file_, &file_size)) {
            const auto error = ::GetLastError();
            ::CloseHandle(mapping_file_);
            mapping_file_ = INVALID_HANDLE_VALUE;
            throw std::system_error(static_cast<int>(error), std::system_category(),
                                    "GetFileSizeEx " + path.string());
        }
        if (file_size.QuadPart < 0 ||
            static_cast<std::uint64_t>(file_size.QuadPart) >
                static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            ::CloseHandle(mapping_file_);
            mapping_file_ = INVALID_HANDLE_VALUE;
            throw ArtifactError("artifact size does not fit the process address space");
        }

        size_ = static_cast<std::size_t>(file_size.QuadPart);
        if (size_ != 0) {
            mapping_ = ::CreateFileMappingW(mapping_file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
            if (mapping_ == nullptr) {
                const auto error = ::GetLastError();
                ::CloseHandle(mapping_file_);
                mapping_file_ = INVALID_HANDLE_VALUE;
                throw std::system_error(static_cast<int>(error), std::system_category(),
                                        "CreateFileMappingW " + path.string());
            }
            const void* view = ::MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0);
            if (view == nullptr) {
                const auto error = ::GetLastError();
                ::CloseHandle(mapping_);
                ::CloseHandle(mapping_file_);
                mapping_      = nullptr;
                mapping_file_ = INVALID_HANDLE_VALUE;
                throw std::system_error(static_cast<int>(error), std::system_category(),
                                        "MapViewOfFile " + path.string());
            }
            data_ = static_cast<const std::byte*>(view);
        }

        direct_file_ = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                     OPEN_EXISTING,
                                     FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING |
                                         FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN,
                                     nullptr);
        if (direct_file_ == INVALID_HANDLE_VALUE) {
            const auto error = ::GetLastError();
            if (data_ != nullptr) { ::UnmapViewOfFile(data_); }
            if (mapping_ != nullptr) { ::CloseHandle(mapping_); }
            ::CloseHandle(mapping_file_);
            data_         = nullptr;
            mapping_      = nullptr;
            mapping_file_ = INVALID_HANDLE_VALUE;
            throw std::system_error(static_cast<int>(error), std::system_category(),
                                    "CreateFileW direct " + path.string());
        }
#else
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (fd < 0) {
            throw std::system_error(errno, std::generic_category(), "open " + path.string());
        }

        struct stat status {};

        if (::fstat(fd, &status) != 0) {
            const int error = errno;
            ::close(fd);
            throw std::system_error(error, std::generic_category(), "fstat " + path.string());
        }
        if (status.st_size < 0 ||
            static_cast<std::uintmax_t>(status.st_size) > std::numeric_limits<std::size_t>::max()) {
            ::close(fd);
            throw ArtifactError("artifact size does not fit the process address space");
        }

        const auto size = static_cast<std::size_t>(status.st_size);
        void* mapping   = nullptr;
        if (size != 0) {
            mapping = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (mapping == MAP_FAILED) {
                const int error = errno;
                ::close(fd);
                throw std::system_error(error, std::generic_category(), "mmap " + path.string());
            }
        }
        fd_   = fd;
        data_ = static_cast<const std::byte*>(mapping);
        size_ = size;
#endif
    }

    ~MappedFile() {
#ifdef _WIN32
        if (data_ != nullptr) { ::UnmapViewOfFile(data_); }
        if (mapping_ != nullptr) { ::CloseHandle(mapping_); }
        if (direct_file_ != INVALID_HANDLE_VALUE) { ::CloseHandle(direct_file_); }
        if (mapping_file_ != INVALID_HANDLE_VALUE) { ::CloseHandle(mapping_file_); }
#else
        if (data_ != nullptr) { ::munmap(const_cast<std::byte*>(data_), size_); }
        if (fd_ >= 0) { ::close(fd_); }
#endif
    }

    MappedFile(const MappedFile&)            = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    const std::byte* data() const noexcept { return data_; }

    std::size_t size() const noexcept { return size_; }

    std::size_t read_direct(std::uint64_t absolute_offset, std::span<std::byte> destination) const {
        constexpr std::size_t alignment = Reader::direct_io_alignment;
        if (absolute_offset % alignment != 0 || destination.size() % alignment != 0 ||
            reinterpret_cast<std::uintptr_t>(destination.data()) % alignment != 0) {
            throw ArtifactError("direct artifact read is not 4096-byte aligned");
        }
#ifdef _WIN32
        std::size_t total = 0;
        while (total < destination.size()) {
            constexpr std::size_t max_read = 1ULL << 30;
            const auto amount = static_cast<DWORD>(std::min(max_read, destination.size() - total));
            const std::uint64_t offset = absolute_offset + total;
            OVERLAPPED operation{};
            operation.Offset     = static_cast<DWORD>(offset & 0xffffffffULL);
            operation.OffsetHigh = static_cast<DWORD>(offset >> 32U);

            DWORD bytes = 0;
            const BOOL started = ::ReadFile(direct_file_, destination.data() + total, amount,
                                            &bytes, &operation);
            if (!started) {
                const auto error = ::GetLastError();
                if (error == ERROR_HANDLE_EOF) { break; }
                if (error != ERROR_IO_PENDING ||
                    !::GetOverlappedResult(direct_file_, &operation, &bytes, TRUE)) {
                    const auto final_error = error == ERROR_IO_PENDING ? ::GetLastError() : error;
                    throw std::system_error(static_cast<int>(final_error), std::system_category(),
                                            "direct artifact read");
                }
            }
            total += bytes;
            if (bytes != amount) { break; }
        }
        return total;
#else
        if (absolute_offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
            destination.size() > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())) {
            throw ArtifactError("direct artifact read exceeds platform I/O limits");
        }

        ssize_t bytes = -1;
        do {
            bytes = ::pread(fd_, destination.data(), destination.size(),
                            static_cast<off_t>(absolute_offset));
        } while (bytes < 0 && errno == EINTR);
        if (bytes < 0) {
            throw std::system_error(errno, std::generic_category(), "direct artifact read");
        }
        return static_cast<std::size_t>(bytes);
#endif
    }

private:
#ifdef _WIN32
    HANDLE mapping_file_       = INVALID_HANDLE_VALUE;
    HANDLE direct_file_        = INVALID_HANDLE_VALUE;
    HANDLE mapping_            = nullptr;
#else
    int fd_                = -1;
#endif
    const std::byte* data_ = nullptr;
    std::size_t size_      = 0;
};

} // namespace

std::string_view object_name(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) -> std::string_view { return descriptor.name; },
                      object);
}

std::uint64_t object_offset(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) { return descriptor.offset; }, object);
}

std::uint64_t object_bytes(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) { return descriptor.bytes; }, object);
}

std::uint64_t object_alignment(const ObjectDescriptor& object) {
    return std::visit(
        [](const auto& descriptor) {
            using Descriptor = std::decay_t<decltype(descriptor)>;
            if constexpr (std::is_same_v<Descriptor, TensorDescriptor>) {
                return tensor_alignment(descriptor.layout);
            } else {
                return resource_alignment(descriptor.encoding);
            }
        },
        object);
}

// Renames a parsed descriptor's identifying string without touching its geometry; used to
// expose a v3 physical object (id "weight/000042") under the logical parameter name that
// a "whole object" binding gives it (e.g. "text/final_norm"), so the rest of the reader
// (and every Binder::require_tensor call site) never has to know the artifact is v3.
ObjectDescriptor renamed(const ObjectDescriptor& object, std::string new_name) {
    return std::visit(
        [&](const auto& descriptor) -> ObjectDescriptor {
            auto copy = descriptor;
            copy.name = std::move(new_name);
            return copy;
        },
        object);
}

struct Reader::Impl {
    explicit Impl(const std::filesystem::path& path) : file(path) {
        if (file.size() < 8) { throw ArtifactError("artifact is shorter than the magic prefix"); }
        if (std::equal(kMagicV2.begin(), kMagicV2.end(), file.data())) {
            parse_v2();
        } else if (std::equal(kMagicV3.begin(), kMagicV3.end(), file.data())) {
            parse_v3();
        } else {
            throw ArtifactError("artifact magic is not NInfer v2");
        }
    }

    void parse_v2() {
        if (file.size() < kPrefixBytesV2) {
            throw ArtifactError("artifact is shorter than the v2 prefix");
        }
        const auto json_bytes = read_u64_le(file.data() + 8);
        if (json_bytes == 0) { throw ArtifactError("json_bytes must be positive"); }
        const auto metadata_end = checked_add(kPrefixBytesV2, json_bytes, "JSON range");
        payload_start           = align_up(metadata_end, kPayloadAlignment, "payload offset");
        if (metadata_end > file.size() || payload_start > file.size()) {
            throw ArtifactError("declared JSON or payload start extends beyond the file");
        }

        Json directory;
        try {
            const auto* begin = reinterpret_cast<const char*>(file.data() + kPrefixBytesV2);
            directory         = Json::parse(begin, begin + json_bytes);
        } catch (const Json::exception& error) {
            throw ArtifactError(std::string("invalid JSON directory: ") + error.what());
        }

        static constexpr std::array root_members = {"identity", "objects"};
        require_members(directory, root_members, "directory root");
        const auto& raw_identity                     = directory.at("identity");
        static constexpr std::array identity_members = {"model_id", "weights_id"};
        require_members(raw_identity, identity_members, "artifact identity");
        identity.model_id   = require_string(raw_identity.at("model_id"), "model_id");
        identity.weights_id = require_string(raw_identity.at("weights_id"), "weights_id");

        const auto& raw_objects = directory.at("objects");
        if (!raw_objects.is_array() || raw_objects.empty()) {
            throw ArtifactError("objects must be a nonempty array");
        }
        entries.reserve(raw_objects.size());
        index.reserve(raw_objects.size());

        const auto payload_bytes = static_cast<std::uint64_t>(file.size()) - payload_start;
        std::uint64_t cursor     = 0;
        for (const auto& raw_object : raw_objects) {
            auto object   = parse_object(raw_object);
            const auto name    = std::string(object_name(object));
            const auto offset  = object_offset(object);
            const auto bytes   = object_bytes(object);
            const auto alignment = object_alignment(object);

            if (offset < cursor) {
                throw ArtifactError("object " + name + " overlaps or is out of order");
            }
            if (offset % alignment != 0) {
                throw ArtifactError("object " + name + " is not " + std::to_string(alignment) +
                                    "-byte aligned");
            }
            const auto end = checked_add(offset, bytes, "object payload range");
            if (end > payload_bytes) {
                throw ArtifactError("object " + name + " extends beyond the file");
            }
            const auto object_index = entries.size();
            auto [_, inserted]      = index.emplace(name, object_index);
            if (!inserted) { throw ArtifactError("duplicate object name: " + name); }
            entries.push_back(std::move(object));
            cursor = end;
        }
    }

    // v3 entry header is 32 bytes: magic(8) | json_bytes(8) | artifact_id(16).
    // See docs/artifact-v3-port-notes.md for the format this parses and the design
    // rationale for how logical parameter bindings are resolved back onto the flat
    // name -> object index that the rest of this reader (and Binder) already expects.
    void parse_v3() {
        if (file.size() < kEntryHeaderBytesV3) {
            throw ArtifactError("artifact is shorter than the v3 entry header");
        }
        const auto json_bytes = read_u64_le(file.data() + 8);
        if (json_bytes == 0) { throw ArtifactError("json_bytes must be positive"); }
        const auto metadata_end = checked_add(kEntryHeaderBytesV3, json_bytes, "JSON range");
        payload_start           = align_up(metadata_end, kPayloadAlignment, "payload offset");
        if (metadata_end > file.size() || payload_start > file.size()) {
            throw ArtifactError("declared JSON or payload start extends beyond the file");
        }

        Json directory;
        try {
            const auto* begin = reinterpret_cast<const char*>(file.data() + kEntryHeaderBytesV3);
            directory         = Json::parse(begin, begin + json_bytes);
        } catch (const Json::exception& error) {
            throw ArtifactError(std::string("invalid v3 JSON directory: ") + error.what());
        }
        if (!directory.is_object() || !directory.contains("components") ||
            !directory.contains("objects") || !directory.contains("bindings") ||
            !directory.contains("files")) {
            throw ArtifactError("v3 directory root is missing components/objects/bindings/files");
        }

        parse_v3_files(directory.at("files"));

        const auto& raw_objects = directory.at("objects");
        if (!raw_objects.is_array() || raw_objects.empty()) {
            throw ArtifactError("v3 objects must be a nonempty array");
        }
        std::vector<ObjectDescriptor> raw;
        std::unordered_map<std::string, std::size_t, TransparentStringHash, std::equal_to<>>
            index_by_id;
        raw.reserve(raw_objects.size());
        const auto payload_bytes = static_cast<std::uint64_t>(file.size()) - payload_start;
        std::uint64_t cursor     = 0;
        for (const auto& raw_object : raw_objects) {
            auto object           = parse_object_v3(raw_object);
            const auto id         = std::string(object_name(object));
            const auto offset     = object_offset(object);
            const auto bytes      = object_bytes(object);
            const auto alignment  = object_alignment(object);
            if (offset < cursor) { throw ArtifactError("v3 object " + id + " overlaps or is out of order"); }
            if (offset % alignment != 0) {
                throw ArtifactError("v3 object " + id + " is not " + std::to_string(alignment) +
                                    "-byte aligned");
            }
            const auto end = checked_add(offset, bytes, "object payload range");
            if (end > payload_bytes) { throw ArtifactError("v3 object " + id + " extends beyond the file"); }
            const auto object_index = raw.size();
            auto [_, inserted]      = index_by_id.emplace(id, object_index);
            if (!inserted) { throw ArtifactError("duplicate v3 object id: " + id); }
            raw.push_back(std::move(object));
            cursor = end;
        }

        // "identity" has no v3 equivalent: v3 describes a model instance through
        // components[*].config (architectures/model_type/...), not a flat model/weights id
        // pair. This is enough for the reader itself; matching it to a WeightsProfile is a
        // model-target concern (Target::resolve_weights) that is out of scope for this pass
        // and still needs its own v3-aware update. Placeholder values below make that
        // failure explicit and easy to grep for instead of silently misidentifying a model.
        identity.model_id   = "v3-artifact";
        identity.weights_id = "unresolved";
        if (directory.contains("metadata") && directory.at("metadata").contains("name")) {
            identity.model_id = require_string(directory.at("metadata").at("name"), "metadata.name");
        }

        const auto& raw_bindings = directory.at("bindings");
        if (!raw_bindings.is_object()) { throw ArtifactError("v3 bindings must be a JSON object"); }

        std::unordered_map<std::string, V3BindingRef, TransparentStringHash, std::equal_to<>>
            binding_refs;
        // All binding names (whole or fragment) that reference a given physical object,
        // used to validate a find_fused() request covers a group exactly, with nothing
        // missing and nothing extra.
        std::unordered_map<std::size_t, std::vector<std::string>> names_by_object;

        for (const auto& [name, binding] : raw_bindings.items()) {
            if (!binding.is_object()) {
                throw ArtifactError("v3 binding " + name + " must be a JSON object");
            }
            if (binding.contains("object")) {
                const auto object_id = require_string(binding.at("object"), "binding object");
                const auto it        = index_by_id.find(object_id);
                if (it == index_by_id.end()) {
                    throw ArtifactError("v3 binding " + name + " references unknown object " +
                                        object_id);
                }
                binding_refs.emplace(name, V3BindingRef{it->second, 0, 0, true});
                names_by_object[it->second].push_back(name);
            } else if (binding.contains("parts")) {
                const auto& parts = binding.at("parts");
                if (!parts.is_array() || parts.size() != 1) {
                    throw ArtifactError(
                        "v3 binding " + name +
                        ": multi-part bindings are not supported by this reader (found " +
                        std::to_string(parts.is_array() ? parts.size() : 0) + " parts)");
                }
                const auto& part      = parts.at(0);
                const auto object_id  = require_string(part.at("object"), "part object");
                const auto it         = index_by_id.find(object_id);
                if (it == index_by_id.end()) {
                    throw ArtifactError("v3 binding " + name + " references unknown object " +
                                        object_id);
                }
                const auto& range = part.at("range");
                if (!range.is_array() || range.size() != 2) {
                    throw ArtifactError("v3 binding " + name + " has a malformed range");
                }
                const auto begin = require_unsigned(range.at(0), "range begin", false);
                const auto end   = require_unsigned(range.at(1), "range end", false);
                if (end <= begin) { throw ArtifactError("v3 binding " + name + " has an empty range"); }

                // `range` is in logical elements (container spec section 7.1). Planar
                // quantized layouts (row_split_k128_v1 and friends) store all rows' "low"
                // plane together, then all rows' "high" plane, etc., so this reader can only
                // reconstruct a shared parent from ranges that align to whole rows -- hence
                // the conversion to row units below using the object's own last dimension.
                // A plain contiguous_le_v1 tensor (e.g. a bias vector) has no such plane
                // structure, so any element range is inherently fine there; treat it as
                // "1 row = 1 element" (k=1) rather than requiring the whole vector at once.
                const auto* tensor_shape = std::get_if<TensorDescriptor>(&raw[it->second]);
                if (tensor_shape == nullptr || tensor_shape->shape.empty()) {
                    throw ArtifactError("v3 binding " + name +
                                        ": partial bindings are only supported for tensors "
                                        "with at least one dimension");
                }
                const auto k = tensor_shape->layout == StorageLayout::ContiguousLeV1
                                  ? std::uint64_t{1}
                                  : tensor_shape->shape.back();
                if (k == 0 || begin % k != 0 || (end - begin) % k != 0) {
                    throw ArtifactError("v3 binding " + name +
                                        ": range does not align to a whole number of rows "
                                        "(unsupported: this reader cannot split mid-row)");
                }
                binding_refs.emplace(name, V3BindingRef{it->second, begin / k, (end - begin) / k, false});
                names_by_object[it->second].push_back(name);
            } else {
                throw ArtifactError("v3 binding " + name + " has neither 'object' nor 'parts'");
            }
        }

        // Expose every whole-object binding directly under its logical name, exactly like a
        // v2 artifact's flat object list. Fragment bindings are intentionally NOT exposed
        // here (see find_fused()): this fork's model-loading code always needs the complete
        // fused parent tensor, never a lone row-range slice of it.
        entries.reserve(binding_refs.size());
        index.reserve(binding_refs.size());
        for (const auto& [name, ref] : binding_refs) {
            if (!ref.whole) { continue; }
            const auto object_index = entries.size();
            auto [_, inserted]      = index.emplace(name, object_index);
            if (!inserted) { throw ArtifactError("duplicate binding name: " + name); }
            entries.push_back(renamed(raw[ref.object_index], name));
        }

        // "components[*].resources" maps upstream frontend file roles (tokenizer.json, ...)
        // straight to an object id, bypassing "bindings" entirely (see container spec
        // section 9.1). This fork's frontend loader asks for them under the v2 convention
        // ("frontend/<role>"), so alias them the same way whole-object bindings are exposed.
        if (directory.at("components").contains("text")) {
            const auto& text = directory.at("components").at("text");
            if (text.contains("resources")) {
                for (const auto& [role, object_id_json] : text.at("resources").items()) {
                    const auto object_id = require_string(object_id_json, "resource object id");
                    const auto it        = index_by_id.find(object_id);
                    if (it == index_by_id.end()) {
                        throw ArtifactError("text resource " + role + " references unknown object " +
                                            object_id);
                    }
                    const auto alias         = "frontend/" + role;
                    const auto object_index  = entries.size();
                    auto [_, inserted]       = index.emplace(alias, object_index);
                    if (!inserted) { throw ArtifactError("duplicate binding name: " + alias); }
                    entries.push_back(renamed(raw[it->second], alias));
                }
            }
        }

        v3_raw_objects   = std::move(raw);
        v3_binding_refs  = std::move(binding_refs);
        v3_names_by_object = std::move(names_by_object);
    }

    void parse_v3_files(const Json& files) {
        if (!files.is_array() || files.empty()) {
            throw ArtifactError("v3 files must be a nonempty array");
        }
        if (files.size() != 1 || !files.at(0).at("path").is_null()) {
            throw ArtifactError(
                "multi-file (sharded) v3 artifacts are not supported by this reader; "
                "re-export with a larger single-file size limit");
        }
    }

    // Resolves a group of logical binding names that together tile one physical v3 object
    // (see reader.h). Returns nullptr only when this is neither a v2 artifact nor a v3
    // artifact with binding metadata (i.e. names.size()==1 and a plain find() should be
    // used instead); every other failure throws, since a mismatch here means a corrupt
    // artifact or a model-loading bug, not a "missing optional feature".
    const ObjectDescriptor* find_fused(std::span<const std::string_view> names) {
        if (names.empty()) { throw ArtifactError("find_fused requires at least one name"); }
        if (v3_binding_refs.empty()) { return nullptr; }

        const auto first = v3_binding_refs.find(names.front());
        if (first == v3_binding_refs.end()) { return nullptr; }
        const auto object_index = first->second.object_index;

        if (const auto cached = fused_cache.find(object_index); cached != fused_cache.end()) {
            return &entries[cached->second];
        }

        auto expected = v3_names_by_object.at(object_index);
        std::sort(expected.begin(), expected.end());
        std::vector<std::string> requested(names.begin(), names.end());
        std::sort(requested.begin(), requested.end());
        if (requested != expected) {
            throw ArtifactError("fused binding group for " + std::string(names.front()) +
                                " does not match the artifact's actual split: requested " +
                                std::to_string(requested.size()) + " name(s), object has " +
                                std::to_string(expected.size()));
        }

        std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
        ranges.reserve(names.size());
        for (const auto& name : names) {
            const auto& ref = v3_binding_refs.find(name)->second;
            if (ref.whole) {
                throw ArtifactError("fused binding group for " + std::string(name) +
                                    " includes a whole-object binding; use find() instead");
            }
            ranges.emplace_back(ref.row_begin, ref.row_count);
        }
        std::sort(ranges.begin(), ranges.end());
        const auto* tensor = std::get_if<TensorDescriptor>(&v3_raw_objects[object_index]);
        if (tensor == nullptr) {
            throw ArtifactError("fused binding group targets a resource, not a tensor");
        }
        const std::uint64_t k     = tensor->shape.empty() ? 1 : tensor->shape.back();
        const std::uint64_t rows  = k == 0 ? 0 : tensor_elements(tensor->shape) / k;
        std::uint64_t expected_row = 0;
        for (const auto& [begin, count] : ranges) {
            if (begin != expected_row) {
                throw ArtifactError("fused binding group for " + std::string(names.front()) +
                                    " leaves a gap or overlap at row " +
                                    std::to_string(expected_row));
            }
            expected_row += count;
        }
        if (expected_row != rows) {
            throw ArtifactError("fused binding group for " + std::string(names.front()) +
                                " does not fully tile object rows (" +
                                std::to_string(expected_row) + " of " + std::to_string(rows) + ")");
        }

        const auto entry_index = entries.size();
        entries.push_back(renamed(v3_raw_objects[object_index],
                                  "fused:" + std::string(names.front())));
        fused_cache.emplace(object_index, entry_index);
        return &entries[entry_index];
    }

    MappedFile file;
    ArtifactIdentity identity;
    std::vector<ObjectDescriptor> entries;
    std::unordered_map<std::string, std::size_t, TransparentStringHash, std::equal_to<>> index;
    std::uint64_t payload_start = 0;

    // v3-only state (empty for v2 artifacts); see parse_v3()/find_fused().
    std::vector<ObjectDescriptor> v3_raw_objects;
    std::unordered_map<std::string, V3BindingRef, TransparentStringHash, std::equal_to<>>
        v3_binding_refs;
    std::unordered_map<std::size_t, std::vector<std::string>> v3_names_by_object;
    std::unordered_map<std::size_t, std::size_t> fused_cache;
};

Reader::Reader(const std::filesystem::path& path) : impl_(std::make_unique<Impl>(path)) {}

Reader::~Reader()                            = default;
Reader::Reader(Reader&&) noexcept            = default;
Reader& Reader::operator=(Reader&&) noexcept = default;

const ArtifactIdentity& Reader::identity() const noexcept { return impl_->identity; }

const std::vector<ObjectDescriptor>& Reader::objects() const noexcept { return impl_->entries; }

const ObjectDescriptor* Reader::find_fused(std::span<const std::string_view> names) const {
    if (names.empty()) { throw ArtifactError("find_fused requires at least one name"); }
    if (names.size() == 1) { return find(names.front()); }
    const auto* found = impl_->find_fused(names);
    if (found == nullptr) {
        throw ArtifactError("fused binding group requested on a v2 artifact: " +
                            std::string(names.front()));
    }
    return found;
}

const ObjectDescriptor* Reader::find(std::string_view name) const noexcept {
    const auto it = impl_->index.find(name);
    return it == impl_->index.end() ? nullptr : &impl_->entries[it->second];
}

std::uint64_t Reader::file_bytes() const noexcept { return impl_->file.size(); }

std::uint64_t Reader::payload_offset() const noexcept { return impl_->payload_start; }

PayloadSpan Reader::payload(const ObjectDescriptor& object) const {
    const auto absolute =
        checked_add(impl_->payload_start, object_offset(object), "absolute payload offset");
    const auto end = checked_add(absolute, object_bytes(object), "absolute payload range");
    if (end > impl_->file.size()) { throw ArtifactError("object payload extends beyond the file"); }
    return {
        absolute,
        std::span<const std::byte>(impl_->file.data() + absolute,
                                   static_cast<std::size_t>(object_bytes(object))),
    };
}

PayloadSpan Reader::payload(std::string_view name) const {
    const auto* object = find(name);
    if (object == nullptr) { throw ArtifactError("unknown artifact object: " + std::string(name)); }
    return payload(*object);
}

std::size_t Reader::read_direct(std::uint64_t absolute_offset,
                                std::span<std::byte> destination) const {
    return impl_->file.read_direct(absolute_offset, destination);
}

} // namespace ninfer::artifact
