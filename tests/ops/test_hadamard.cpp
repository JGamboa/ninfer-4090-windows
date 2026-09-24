// hadamard_1024_launch against an FP64 Sylvester transform written from the definition
// (docs/maintainer/bonsai-ternary-design.md 1.5): y = H_blk(signs * x) / 32, H[r][c] =
// (-1)^popcount(r & c) over each consecutive 1024-column block.
#include "core/device.h"
#include "ops/hadamard/hadamard.h"
#include "ops/op_tester.h"

#include <bit>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr PointwiseCriterion hadamard_bf16_criterion() {
    // BF16 output rounding (2^-9 relative) plus FP32 accumulation of 1024 terms.
    return {/*absolute*/ 1.0e-4, /*relative*/ 4.0e-3};
}

std::vector<double> oracle(const std::vector<float>& x, const std::vector<float>& signs,
                           const std::vector<std::int32_t>& perm, int width, int tokens) {
    std::vector<double> y(x.size());
    for (int t = 0; t < tokens; ++t) {
        for (int block = 0; block < width / 1024; ++block) {
            for (int r = 0; r < 1024; ++r) {
                double sum = 0;
                for (int c = 0; c < 1024; ++c) {
                    const int column = block * 1024 + c;
                    const int source = perm.empty() ? column : perm[column];
                    const double h   = std::popcount(unsigned(r & c)) % 2 ? -1.0 : 1.0;
                    sum += h * signs[column] * x[std::size_t(t) * width + source];
                }
                y[std::size_t(t) * width + block * 1024 + r] = sum / 32.0;
            }
        }
    }
    return y;
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

// Grouped/tiled GDN value-head order: 16 key heads x 3 repeats of 128 columns.
std::vector<std::int32_t> head_permutation(int width) {
    std::vector<std::int32_t> perm(width);
    for (int column = 0; column < width; ++column) {
        const int head = column / 128, j = column % 128;
        perm[column]   = ((head % 3) * 16 + head / 3) * 128 + j;
    }
    return perm;
}

int run_case(const char* label, int width, int tokens, std::uint32_t seed, bool alias,
             bool gather, bool graph = false) {
    const std::size_t count = std::size_t(width) * tokens;
    std::vector<float> x(count), signs(width);
    fill_uniform(x, seed, -8.0f, 8.0f);
    round_to_bf16(x);
    std::mt19937 rng(seed + 1);
    for (auto& s : signs) s = (rng() & 1) ? 1.0f : -1.0f;
    const auto perm     = gather ? head_permutation(width) : std::vector<std::int32_t>{};
    const auto expected = oracle(x, signs, perm, width, tokens);

    const auto x_bits = bf16_bits(x);
    const auto s_bits = bf16_bits(signs);
    GuardedDeviceBuffer device_x(count * 2), device_signs(width * 2), device_y(count * 2);
    GuardedDeviceBuffer device_perm(gather ? width * 4 : 4);
    device_x.copy_from_host(x_bits.data(), device_x.bytes());
    device_signs.copy_from_host(s_bits.data(), device_signs.bytes());
    if (gather) device_perm.copy_from_host(perm.data(), device_perm.bytes());
    Tensor tx(device_x.data(), DType::BF16, {width, tokens});
    Tensor ts(device_signs.data(), DType::BF16, {width});
    Tensor ty(alias ? device_x.data() : device_y.data(), DType::BF16, {width, tokens});
    const auto* p = gather ? static_cast<const std::int32_t*>(device_perm.data()) : nullptr;

    if (graph) {
        cudaStream_t stream;
        cudaGraph_t captured;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        ops::detail::hadamard_1024_launch(tx, ts, p, ty, stream);
        CUDA_CHECK(cudaStreamEndCapture(stream, &captured));
        CUDA_CHECK(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
        for (int replay = 0; replay < 2; ++replay) {
            CUDA_CHECK(cudaMemcpyAsync(device_x.data(), x_bits.data(), count * 2,
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaGraphLaunch(executable, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(captured));
        CUDA_CHECK(cudaStreamDestroy(stream));
    } else {
        ops::detail::hadamard_1024_launch(tx, ts, p, ty, nullptr);
        cuda_synchronize();
    }

    int failures = verify_pointwise(label, from_device_bf16(ty.data, count), expected,
                                    hadamard_bf16_criterion());
    if (!alias) {
        failures += verify_exact("hadamard input unchanged",
                                 from_device<std::uint16_t>(device_x.data(), count), x_bits);
    }
    failures += device_x.verify_guards("hadamard x") + device_y.verify_guards("hadamard y") +
                device_signs.verify_guards("hadamard signs");
    return failures;
}

// A one-hot input selects one column of H: every output is exactly +-sign/32.
int run_exact_basis() {
    const int width = 1024, hot = 613;
    std::vector<float> x(width, 0.0f), signs(width, 1.0f);
    x[hot]     = 1.0f;
    signs[hot] = -1.0f;
    const auto x_bits = bf16_bits(x), s_bits = bf16_bits(signs);
    GuardedDeviceBuffer device_x(width * 2), device_signs(width * 2);
    device_x.copy_from_host(x_bits.data(), device_x.bytes());
    device_signs.copy_from_host(s_bits.data(), device_signs.bytes());
    Tensor tx(device_x.data(), DType::BF16, {width, 1});
    Tensor ts(device_signs.data(), DType::BF16, {width});
    ops::detail::hadamard_1024_launch(tx, ts, nullptr, tx, nullptr);
    cuda_synchronize();
    std::vector<std::uint16_t> expected(width);
    for (int r = 0; r < width; ++r) {
        expected[r] = f32_to_bf16(std::popcount(unsigned(r & hot)) % 2 ? 0x1p-5f : -0x1p-5f);
    }
    return verify_exact("hadamard one-hot column",
                        from_device<std::uint16_t>(device_x.data(), width), expected);
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = run_exact_basis();
    // The three Bonsai input widths at decode, MTP-verify and prefill-chunk token counts.
    failures += run_case("hadamard [5120,1]", 5120, 1, 11u, false, false);
    failures += run_case("hadamard [5120,8] in place", 5120, 8, 12u, true, false);
    failures += run_case("hadamard [6144,3] graph", 6144, 3, 13u, false, false, true);
    failures += run_case("hadamard [6144,5] head gather", 6144, 5, 14u, false, true);
    failures += run_case("hadamard [17408,2]", 17408, 2, 15u, false, false);
    failures += run_case("hadamard [5120,64] in place graph", 5120, 64, 16u, true, false, true);
    std::cout << (failures ? "FAIL" : "OK") << " hadamard_1024\n";
    return failures ? 1 : 0;
}
