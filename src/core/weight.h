#pragma once

#include "core/dtype.h"

#include <cstdint>

namespace ninfer {

enum class QType : std::uint16_t {
    Q4_G64_FP16         = 0,
    Q5_G64_FP16         = 1,
    Q6_G64_FP16         = 2,
    Q8_G32_FP16         = 3,
    BF16                = 4,
    FP32                = 5,
    INT32               = 6,
    NVFP4               = 7,
    FP8_E4M3FN_ROW_BF16 = 8,
    T2_G128_FP16        = 9,
    T5_G128_FP16        = 10,
};

// Prism ternary weights (codes {0, 1, 2} = c - 1, one FP16 scale per 128 columns): T2 packs
// four 2-bit codes per byte, T5 scaled base-3 13-byte units of 64 columns (TernaryRowK128).
[[nodiscard]] constexpr bool ternary_qtype(QType qtype) noexcept {
    return qtype == QType::T2_G128_FP16 || qtype == QType::T5_G128_FP16;
}

enum class QuantLayout : std::uint16_t {
    RowSplit            = 0,
    Contiguous          = 1,
    BlockScaleK16M128x4 = 2,
    RowScale            = 3,
    TernaryRowK128      = 4,
};

struct Weight {
    const void* payload            = nullptr;
    std::uint64_t payload_bytes    = 0;
    std::uint64_t high_plane_bytes = 0;
    QType qtype                    = QType::Q4_G64_FP16;
    std::uint32_t group_size       = 0;
    std::int32_t shape[4]          = {1, 1, 1, 1};
    std::int32_t padded_shape[4]   = {1, 1, 1, 1};
    std::uint32_t ndim             = 0;

    const void* qdata          = nullptr;
    const void* qhigh          = nullptr;
    const void* scales         = nullptr;
    std::int32_t n             = 0;
    std::int32_t k             = 0;
    std::int32_t group         = 0;
    QuantLayout layout         = QuantLayout::RowSplit;
    DType scale_dtype          = DType::FP32;
    std::int32_t scale_ne[4]   = {1, 1, 1, 1};
    std::int64_t scale_nb[4]   = {0, 0, 0, 0};
    float weight_scale_divisor = 0.0F;
    float input_scale_divisor  = 0.0F;
    // Ternary (T2/T5) only: BF16 +-1 [k] Prism sign vector of a weight stored in the rotated basis.
    // The logical weight is W' H S: a projection multiplies W' by the 1024-block normalized
    // Walsh-Hadamard rotation (1/32) H (signs * x) of its input (ninfer/ops/hadamard.h); an
    // embedding table's logical row is signs * (1/32) H z' of its stored row z'.
    const void* input_signs = nullptr;
};

} // namespace ninfer
