// Cold-weight timing (median of repeated runs after a clock warm-up) of the Prism ternary
// projection and the Hadamard rotation at the Bonsai 2 27B shapes. Weight copies rotate so their
// total exceeds L2; the reported bandwidth is the T5 weight bytes (codes + scales) read per call
// divided by the call time. Each shape is timed under A8 (quantization included) with an
// unrotated weight and with the Prism rotation fused into the quantization (rot), as production.
#include "core/arena.h"
#include "core/device.h"
#include "core/weight_view.h"
#include "ninfer/ops/hadamard.h"
#include "ops/linear/t5/t5_project.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::size_t kColdBytes = 512ull << 20; // well above the 72 MiB L2 of an RTX 4090
constexpr int kIterations        = 40;
constexpr int kRepeats           = 7; // report the median repeat

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

// Run a timed body kRepeats times and return the median microseconds per iteration.
template <class Body>
double median_us(cudaEvent_t start, cudaEvent_t stop, Body&& body) {
    std::vector<double> samples;
    for (int repeat = 0; repeat < kRepeats; ++repeat) {
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < kIterations; ++i) body(i);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        samples.push_back(1000.0 * ms / kIterations);
    }
    return median(samples);
}

struct Shape {
    const char* role;
    int n, k;
};

} // namespace

int main(int argc, char** argv) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::puts("SKIP: no CUDA device");
        return 77;
    }
    std::vector<int> tokens{1, 2, 3, 4, 8, 16, 64, 512};
    if (argc > 1) {
        tokens.clear();
        for (int i = 1; i < argc; ++i) tokens.push_back(std::atoi(argv[i]));
    }
    const std::array<Shape, 5> shapes{{{"gdn in_proj", 16384, 5120},
                                       {"attn qkvg", 14336, 5120},
                                       {"mlp gate+up", 34816, 5120},
                                       {"o_proj/out", 5120, 6144},
                                       {"mlp down", 5120, 17408}}};
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    {
        // Bring the GPU to steady clocks before the first measurement.
        void* scratch = nullptr;
        CUDA_CHECK(cudaMalloc(&scratch, kColdBytes));
        for (int i = 0; i < 200; ++i) CUDA_CHECK(cudaMemsetAsync(scratch, i, kColdBytes));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(scratch));
    }
    std::printf("%-12s %6s %6s %4s %10s %10s %10s %10s\n", "role", "N", "K", "T", "A8 us",
                "A8 GB/s", "rot us", "rot GB/s");
    for (const auto& shape : shapes) {
        const std::array<std::uint64_t, 2> dims{std::uint64_t(shape.n), std::uint64_t(shape.k)};
        const auto geometry =
            weight_geometry(QType::T5_G128_FP16, QuantLayout::TernaryRowK128, dims);
        const int copies = int((kColdBytes + geometry.bytes - 1) / geometry.bytes);
        std::vector<void*> buffers(copies);
        std::vector<WeightParent> parents(copies);
        std::vector<Weight> weights(copies);
        for (int c = 0; c < copies; ++c) {
            CUDA_CHECK(cudaMalloc(&buffers[c], geometry.bytes));
            CUDA_CHECK(cudaMemset(buffers[c], 0x55 + c, geometry.code_bytes)); // codes
            CUDA_CHECK(cudaMemset(static_cast<char*>(buffers[c]) + geometry.scale_offset, 0x1c,
                                  geometry.scale_bytes)); // small positive FP16 scales
            parents[c] = {geometry, static_cast<const std::byte*>(buffers[c])};
            weights[c] = native_weight(
                WeightView{{dims[0], dims[1]}, {{&parents[c], 0, dims[0] * dims[1]}}});
        }
        void* signs = nullptr;
        CUDA_CHECK(cudaMalloc(&signs, std::size_t(shape.k) * 2));
        CUDA_CHECK(cudaMemset(signs, 0x3f, std::size_t(shape.k) * 2)); // ~+0.75, timing only
        for (const int t : tokens) {
            void *x_data = nullptr, *y_data = nullptr;
            CUDA_CHECK(cudaMalloc(&x_data, std::size_t(shape.k) * t * 2));
            CUDA_CHECK(cudaMalloc(&y_data, std::size_t(shape.n) * t * 2));
            CUDA_CHECK(cudaMemset(x_data, 0x3c, std::size_t(shape.k) * t * 2));
            Tensor x(x_data, DType::BF16, {shape.k, t});
            Tensor y(y_data, DType::BF16, {shape.n, t});
            Tensor* outputs[] = {&y};
            DeviceArena workspace(ops::detail::t5_workspace_capacity_bytes(
                ops::LinearPolicy::AllowA8, shape.k, t));
            double us[2];
            for (int mode = 0; mode < 2; ++mode) {
                for (auto& weight : weights) weight.input_signs = mode ? signs : nullptr;
                for (int c = 0; c < copies; ++c) {
                    ops::detail::t5_project(x, weights[c], outputs, false,
                                            ops::LinearPolicy::AllowA8, &workspace, nullptr);
                }
                us[mode] = median_us(start, stop, [&](int i) {
                    ops::detail::t5_project(x, weights[i % copies], outputs, false,
                                            ops::LinearPolicy::AllowA8, &workspace, nullptr);
                });
            }
            std::printf("%-12s %6d %6d %4d %10.1f %10.1f %10.1f %10.1f\n", shape.role, shape.n,
                        shape.k, t, us[0], double(geometry.bytes) / (us[0] * 1e3), us[1],
                        double(geometry.bytes) / (us[1] * 1e3));
            CUDA_CHECK(cudaFree(x_data));
            CUDA_CHECK(cudaFree(y_data));
        }
        CUDA_CHECK(cudaFree(signs));
        for (void* buffer : buffers) CUDA_CHECK(cudaFree(buffer));
    }

    std::printf("\n%-12s %6s %4s %10s\n", "hadamard", "K", "T", "us/call");
    for (const int k : {5120, 6144, 17408}) {
        void *x_data = nullptr, *s_data = nullptr;
        CUDA_CHECK(cudaMalloc(&s_data, std::size_t(k) * 2));
        CUDA_CHECK(cudaMemset(s_data, 0x3f, std::size_t(k) * 2));
        Tensor signs(s_data, DType::BF16, {k});
        for (const int t : tokens) {
            CUDA_CHECK(cudaMalloc(&x_data, std::size_t(k) * t * 2));
            CUDA_CHECK(cudaMemset(x_data, 0x3c, std::size_t(k) * t * 2));
            Tensor x(x_data, DType::BF16, {k, t});
            ops::hadamard_1024(x, signs, x, nullptr);
            const double us =
                median_us(start, stop, [&](int) { ops::hadamard_1024(x, signs, x, nullptr); });
            std::printf("%-12s %6d %4d %10.1f\n", "", k, t, us);
            CUDA_CHECK(cudaFree(x_data));
        }
        CUDA_CHECK(cudaFree(s_data));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}
