#pragma once

#include "core/weight.h"
#include "core/tensor.h"
#include "ops/a8_g64_reference.h"

#include <cstdint>
#include <span>
#include <string_view>

namespace ninfer::test::linear_swiglu {

enum class ActivationCompute : std::uint8_t {
    A16,
    A8,
    A4,
    // AllowA8 on a Q4/Q5 RowSplit weight: from kA8G64MinTokens on, the documented per-token
    // 64-group int8 activation (tests/ops/a8_g64_reference.h), which the oracle models; below
    // it, A16.
    A8G64,
};

using test::kA8G64MinTokens;

struct Profile {
    QType qtype;
    std::int32_t gate_up_rows;
    std::int32_t input_rows;
    std::int32_t output_rows;
    std::uint32_t seed;
    ActivationCompute activation_compute;
};

int run_profile(std::string_view label, const Profile& profile,
                std::span<const std::int32_t> token_cases,
                std::span<const std::int32_t> graph_cases = {});

} // namespace ninfer::test::linear_swiglu
