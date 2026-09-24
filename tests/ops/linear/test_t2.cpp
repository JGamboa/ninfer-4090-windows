// T2_G128_FP16 projections (A16 and A8 activation paths) against an FP64 oracle from the format
// (docs/maintainer/bonsai-ternary-design.md 2): weight (n, k) = (code - 1) * scale[n][k / 128],
// code = (codes[n][k / 4] >> 2 * (k % 4)) & 3. The weight is built through the production
// geometry and native Weight view, including a row view of a fused parent.
#include "core/device.h"
#include "core/weight_view.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ops/op_tester.h"

#include <cuda_fp16.h>

#include <array>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// One BF16 unit roundoff in relative L2; gross error covers final BF16 storage (A16 criterion).
constexpr ReductionCriterion kA16{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};
// The shared Linear A8 criterion: activation quantization allowance plus BF16 storage.
constexpr ReductionCriterion kA8{0.04, 1.0 / 256.0, 0.06};

ReductionCriterion criterion(ops::LinearPolicy policy) {
    return ops::allows_a8(policy) ? kA8 : kA16;
}

const char* policy_name(ops::LinearPolicy policy) {
    return ops::allows_a8(policy) ? " A8" : " A16";
}

struct Ternary {
    std::int32_t n = 0, k = 0;
    std::vector<std::uint8_t> code;  // [n][k] in {0, 1, 2}
    std::vector<float> scale;        // [n][k / 128], FP16-representable
    std::vector<std::byte> payload;  // ternary_row_k128_v1
    WeightGeometry geometry;
    DeviceBuffer device;
    WeightParent parent;

    Ternary(std::int32_t rows, std::int32_t columns, std::uint32_t seed) : n(rows), k(columns) {
        std::mt19937 rng(seed);
        code.resize(std::size_t(n) * k);
        scale.resize(std::size_t(n) * (k / 128));
        for (auto& c : code) c = static_cast<std::uint8_t>(rng() % 3);
        std::uniform_real_distribution<float> magnitude(0.002f, 0.05f);
        for (auto& s : scale) s = __half2float(__float2half(magnitude(rng)));
        const std::array<std::uint64_t, 2> shape{std::uint64_t(n), std::uint64_t(k)};
        geometry = weight_geometry(QType::T2_G128_FP16, QuantLayout::TernaryRowK128, shape);
        payload.assign(geometry.bytes, std::byte{0});
        for (std::int32_t row = 0; row < n; ++row) {
            for (std::int32_t column = 0; column < k; ++column) {
                auto& byte = payload[std::size_t(row) * (k / 4) + column / 4];
                byte |= std::byte(code[std::size_t(row) * k + column] << (2 * (column % 4)));
            }
            for (std::int32_t group = 0; group < k / 128; ++group) {
                const __half h = __float2half(scale[std::size_t(row) * (k / 128) + group]);
                std::memcpy(&payload[geometry.scale_offset + (std::size_t(row) * (k / 128) + group) * 2],
                            &h, 2);
            }
        }
        device = DeviceBuffer(payload.size());
        device.copy_from_host(payload.data(), payload.size());
        parent = {geometry, static_cast<const std::byte*>(device.p)};
    }

    Weight rows(std::int32_t first, std::int32_t count) const {
        return native_weight(WeightView{{std::uint64_t(count), std::uint64_t(k)},
                                        {{&parent, std::uint64_t(first) * k,
                                          std::uint64_t(first + count) * k}}});
    }

    double weight(std::int32_t row, std::int32_t column) const {
        return (double(code[std::size_t(row) * k + column]) - 1.0) *
               scale[std::size_t(row) * (k / 128) + column / 128];
    }
};

std::vector<float> activation(std::int32_t k, std::int32_t t, std::uint32_t seed) {
    std::vector<float> x(std::size_t(k) * t);
    fill_uniform(x, seed, -4.0f, 4.0f);
    round_to_bf16(x);
    return x;
}

// out[t][r] = sum_c W[first + r][c] x[t][c], FP64.
std::vector<double> oracle(const Ternary& w, std::int32_t first, std::int32_t rows,
                           const std::vector<float>& x, std::int32_t t) {
    std::vector<double> out(std::size_t(rows) * t);
    for (std::int32_t token = 0; token < t; ++token) {
        for (std::int32_t r = 0; r < rows; ++r) {
            double sum = 0;
            for (std::int32_t c = 0; c < w.k; ++c) sum += w.weight(first + r, c) * x[std::size_t(token) * w.k + c];
            out[std::size_t(token) * rows + r] = sum;
        }
    }
    return out;
}

int linear_case(const Ternary& w, std::int32_t first, std::int32_t rows, std::int32_t t,
                bool graph, ops::LinearPolicy policy = ops::LinearPolicy::A16Only) {
    const auto x = activation(w.k, t, 17u * t + rows);
    auto device_x = to_device_bf16(x);
    GuardedDeviceBuffer out(std::size_t(rows) * t * 2);
    Tensor tx(device_x.p, DType::BF16, {w.k, t});
    Tensor to(out.data(), DType::BF16, {rows, t});
    const Weight view = w.rows(first, rows);
    DeviceArena workspace(
        ops::linear_workspace_capacity_bytes(QType::T2_G128_FP16, rows, w.k, policy, 1, t) + 256);
    if (graph) {
        cudaStream_t stream;
        cudaGraph_t captured;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        ops::linear(tx, view, to, policy, workspace, stream);
        CUDA_CHECK(cudaStreamEndCapture(stream, &captured));
        CUDA_CHECK(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
        for (int replay = 0; replay < 2; ++replay) CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(captured));
        CUDA_CHECK(cudaStreamDestroy(stream));
    } else {
        ops::linear(tx, view, to, policy, workspace, nullptr);
        cuda_synchronize();
    }
    const std::string label = "t2 linear [" + std::to_string(rows) + "," + std::to_string(w.k) +
                              "] rows+" + std::to_string(first) + " T=" + std::to_string(t) +
                              policy_name(policy);
    return verify_reduction(label, from_device_bf16(out.data(), std::size_t(rows) * t),
                            oracle(w, first, rows, x, t), criterion(policy)) +
           out.verify_guards(label.c_str());
}

int linear_add_case(const Ternary& w, std::int32_t t,
                    ops::LinearPolicy policy = ops::LinearPolicy::A16Only) {
    const auto x        = activation(w.k, t, 91u + t);
    auto residual_value = activation(w.n, t, 92u + t);
    auto expected       = oracle(w, 0, w.n, x, t);
    for (std::size_t i = 0; i < expected.size(); ++i) expected[i] += residual_value[i];
    auto device_x        = to_device_bf16(x);
    auto device_residual = to_device_bf16(residual_value);
    Tensor tx(device_x.p, DType::BF16, {w.k, t});
    Tensor tr(device_residual.p, DType::BF16, {w.n, t});
    DeviceArena workspace(
        ops::linear_add_workspace_capacity_bytes(QType::T2_G128_FP16, w.n, w.k, policy, 1, t) +
        256);
    ops::linear_add(tx, w.rows(0, w.n), tr, policy, workspace, nullptr);
    cuda_synchronize();
    return verify_reduction("t2 linear_add [" + std::to_string(w.n) + "," + std::to_string(w.k) +
                                "] T=" + std::to_string(t) + policy_name(policy),
                            from_device_bf16(device_residual.p, expected.size()), expected,
                            criterion(policy));
}

// The fused attention parent is stored query, key, gate, value.
int attention_case(const Ternary& w, std::int32_t t,
                   ops::LinearPolicy policy = ops::LinearPolicy::A16Only) {
    const auto x  = activation(w.k, t, 71u + t);
    auto device_x = to_device_bf16(x);
    const std::array<std::int32_t, 4> rows{6144, 1024, 6144, 1024};
    std::array<DeviceBuffer, 4> outputs;
    std::array<Tensor, 4> tensors;
    for (int i = 0; i < 4; ++i) {
        outputs[i] = DeviceBuffer(std::size_t(rows[i]) * t * 2);
        tensors[i] = Tensor(outputs[i].p, DType::BF16, {rows[i], t});
    }
    Tensor tx(device_x.p, DType::BF16, {w.k, t});
    DeviceArena workspace(
        ops::attn_input_proj_workspace_capacity_bytes(QType::T2_G128_FP16, w.n, w.k, policy, 1, t) +
        256);
    ops::attn_input_proj(tx, w.rows(0, w.n), tensors[0], tensors[2], tensors[1], tensors[3],
                         policy, workspace, nullptr);
    cuda_synchronize();
    int failures = 0, first = 0;
    const std::array<const char*, 4> names{"query", "key", "gate", "value"};
    for (int i = 0; i < 4; ++i) {
        failures += verify_reduction(std::string("t2 attn_input_proj ") + names[i] +
                                         policy_name(policy),
                                     from_device_bf16(outputs[i].p, std::size_t(rows[i]) * t),
                                     oracle(w, first, rows[i], x, t), criterion(policy));
        first += rows[i];
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    {
        // A partial last row block, both token tiles and a row view of a fused parent.
        const Ternary small(300, 3072, 5u);
        for (std::int32_t t : {1, 3, 8, 9, 17, 33}) failures += linear_case(small, 0, 300, t, t == 9);
        failures += linear_case(small, 44, 200, 5, false);
    }
    {
        // Rows a multiple of 16 take the tensor-core route for T >= 5: several token tiles, a
        // partial last tile and a row view that starts inside the parent.
        const Ternary tiled(320, 3072, 6u);
        for (std::int32_t t : {4, 5, 8, 9, 17, 40}) failures += linear_case(tiled, 0, 320, t, t == 17);
        failures += linear_case(tiled, 32, 256, 7, false);
    }
    {
        // Rows a multiple of 64 take the prefill GEMM for T >= 17: one and several 64-token
        // tiles, partial tiles, a row view starting at a 64-row boundary, and the residual add.
        const Ternary gemm(448, 2048, 8u);
        for (std::int32_t t : {17, 64, 65, 130}) failures += linear_case(gemm, 0, 448, t, t == 65);
        failures += linear_case(gemm, 64, 320, 40, false);
        failures += linear_add_case(gemm, 70);
    }
    // Bonsai decode shapes (N, K) at decode and MTP-verify widths.
    for (const auto [n, k] : std::array<std::pair<int, int>, 3>{{{5120, 6144}, {5120, 17408}, {16384, 5120}}}) {
        const Ternary w(n, k, 1000u + n + k);
        for (std::int32_t t : {1, 4, 8}) failures += linear_case(w, 0, n, t, t == 4);
        if (n == 5120) failures += linear_add_case(w, 3);
    }
    {
        const Ternary attention(14336, 5120, 77u);
        failures += attention_case(attention, 2);
        failures += attention_case(attention, 6); // tensor-core route, four outputs
        failures += attention_case(attention, 3, ops::LinearPolicy::AllowA8);
    }
    {
        // A8: dp4a GEMV at every token template (with a partial row block and a row view), the
        // int8 MMA GEMM with partial 64-token tiles, graph replay, and the residual add.
        constexpr auto a8 = ops::LinearPolicy::AllowA8;
        const Ternary small(300, 3072, 15u);
        for (std::int32_t t : {1, 2, 3, 4, 5, 8, 9}) {
            failures += linear_case(small, 0, 300, t, t == 3, a8);
        }
        failures += linear_case(small, 44, 200, 6, false, a8);
        const Ternary gemm(448, 2048, 18u);
        for (std::int32_t t : {9, 64, 65, 130}) failures += linear_case(gemm, 0, 448, t, t == 65, a8);
        failures += linear_case(gemm, 64, 320, 40, false, ops::LinearPolicy::AllowA4);
        failures += linear_add_case(gemm, 70, a8);
        failures += linear_add_case(gemm, 3, a8);
        for (const auto [n, k] : std::array<std::pair<int, int>, 2>{{{5120, 17408}, {16384, 5120}}}) {
            const Ternary w(n, k, 2000u + n + k);
            for (std::int32_t t : {1, 3, 8, 32}) failures += linear_case(w, 0, n, t, false, a8);
        }
    }
    std::cout << (failures ? "FAIL" : "OK") << " t2 A16/A8\n";
    return failures ? 1 : 0;
}
