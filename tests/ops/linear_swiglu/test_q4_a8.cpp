#include "core/weight.h"
#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    try {
        // AllowA8 on the Q4 weight: A16 below kA8G64MinTokens (decode and verification widths),
        // the documented A8 quantization from it on, across its boundary, one and several
        // 128-token tiles and a partial last tile.
        constexpr std::array<std::int32_t, 9> kTokenCases{1, 64, 128, 129, 130, 255, 256, 257, 300};
        const int failures =
            run_profile("LinearSwiGLU Q4_A8",
                        {QType::Q4_G64_FP16, 34816, 5120, 17408, 1403U, ActivationCompute::A8G64},
                        kTokenCases, std::array<std::int32_t, 3>{128, 129, 300});
        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU Q4_A8 correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU Q4_A8 test failed: " << error.what() << '\n';
        return 1;
    }
}
