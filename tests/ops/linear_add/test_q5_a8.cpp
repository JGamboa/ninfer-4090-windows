#include "ops/linear_add/linear_add_test_common.h"

#include <array>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <utility>

namespace {

using ninfer::test::linear_add::ShapeCase;
using ninfer::test::linear_add::WeightFormat;

int q5_a8_conformance() {
    // AllowA8 on both registered Q5 shapes: A16 below 129 columns (the A16 route starts are
    // probed as in the A16 suite), the documented A8 quantization from 129 on: its boundary, one
    // and several 64-token tiles, partial last tiles and graph replays with a zero activation.
    constexpr std::array<std::int32_t, 8> kInteriors{1, 8, 64, 128, 192, 256, 300, 1024};
    constexpr std::array<std::int32_t, 3> kRouteStarts{17, 49, 129};
    constexpr std::array<std::int32_t, 3> kGraphTokens{128, 129, 300};
    int failures = 0;
    for (const auto& [k, seed] : {std::pair{6144, 411U}, std::pair{17408, 419U}}) {
        failures += ninfer::test::linear_add::run_shape(
            "Q5_A8 LinearAdd", WeightFormat::Q5G64F16S,
            ShapeCase{5120, k, seed, kRouteStarts, kInteriors, kGraphTokens, false, 0, true});
    }
    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear_add::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        const int failures = q5_a8_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Q5_A8 LinearAdd\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q5_A8 LinearAdd: " << error.what() << '\n';
        return 1;
    }
}
