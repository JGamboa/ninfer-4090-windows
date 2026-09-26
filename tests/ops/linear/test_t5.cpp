// T5_G128_FP16 (base-3) projections and embedding against an FP64 oracle from the format
// (docs/maintainer/bonsai-ternary-design.md 2): weight (n, k) = (code - 1) * scale[n][k / 128],
// code = (codes[n][k / 4] >> 2 * (k % 4)) & 3. The weight is built through the production
// geometry and native Weight view, including a row view of a fused parent.
#include "core/device.h"
#include "core/weight_view.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/rmsnorm_swiglu_mlp.h"
#include "ops/op_tester.h"

#include <cuda_fp16.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <stdexcept>
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
    std::vector<std::byte> payload;  // ternary_row_k128_v1, base-3 codes
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
        geometry = weight_geometry(QType::T5_G128_FP16, QuantLayout::TernaryRowK128, shape);
        payload.assign(geometry.bytes, std::byte{0});
        for (std::int32_t row = 0; row < n; ++row) {
            // Base-3 units (design doc 9.1): byte i < 12 of unit u, g = i / 4, j = i % 4, holds
            // t_m = c[64 u + 20 g + 4 m + j]; byte 12 holds c[64 u + 60 + m] (m < 4) and t_4 = 0.
            // q = ceil(256 v / 243), v = sum_m t_m 3^(4 - m).
            const std::uint8_t* c = &code[std::size_t(row) * k];
            for (std::int32_t unit = 0; unit < k / 64; ++unit) {
                for (int i = 0; i < 13; ++i) {
                    int v = 0;
                    for (int m = 0; m < 5; ++m) {
                        const int column = i < 12 ? 20 * (i / 4) + 4 * m + i % 4 : 60 + m;
                        v = 3 * v + (i < 12 || m < 4 ? c[64 * unit + column] : 0);
                    }
                    payload[std::size_t(row) * (k / 64 * 13) + unit * 13 + i] =
                        std::byte((256 * v + 242) / 243);
                }
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

    // Prism rotation: the weight multiplies (1/32) H (signs * x) per 1024-column block.
    std::vector<float> signs;
    DeviceBuffer device_signs;

    void rotate(std::uint32_t seed) {
        std::mt19937 rng(seed);
        signs.resize(std::size_t(k));
        for (auto& sign : signs) sign = rng() & 1 ? 1.0f : -1.0f;
        device_signs = to_device_bf16(signs);
    }

    Weight rows(std::int32_t first, std::int32_t count) const {
        Weight view = native_weight(WeightView{{std::uint64_t(count), std::uint64_t(k)},
                                               {{&parent, std::uint64_t(first) * k,
                                                 std::uint64_t(first + count) * k}}});
        view.input_signs = signs.empty() ? nullptr : device_signs.p;
        return view;
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

// The weight's input: the logical input itself, or its Prism rotation (1/32) H (signs * input)
// with the Sylvester Walsh-Hadamard matrix H[r][c] = (-1)^popcount(r & c) of every 1024-column
// block, FP64.
std::vector<double> weight_input(const Ternary& w, std::vector<double> input, std::int32_t t) {
    if (w.signs.empty()) return input;
    std::vector<double> rotated(input.size());
    for (std::int32_t token = 0; token < t; ++token) {
        for (std::int32_t block = 0; block < w.k; block += 1024) {
            const std::size_t base = std::size_t(token) * w.k + block;
            for (int r = 0; r < 1024; ++r) {
                double sum = 0;
                for (int c = 0; c < 1024; ++c) {
                    const double term = input[base + c] * w.signs[block + c];
                    sum += std::popcount(unsigned(r & c)) & 1 ? -term : term;
                }
                rotated[base + r] = sum / 32.0;
            }
        }
    }
    return rotated;
}

// out[t][r] = sum_c W[first + r][c] input[t][c] for the weight input [T][K], FP64.
std::vector<double> project(const Ternary& w, std::int32_t first, std::int32_t rows,
                            const std::vector<double>& input, std::int32_t t) {
    std::vector<double> out(std::size_t(rows) * t), row(std::size_t(w.k));
    for (std::int32_t r = 0; r < rows; ++r) {
        for (std::int32_t c = 0; c < w.k; ++c) row[c] = w.weight(first + r, c);
        for (std::int32_t token = 0; token < t; ++token) {
            const double* x = &input[std::size_t(token) * w.k];
            double sum      = 0;
            for (std::int32_t c = 0; c < w.k; ++c) sum += row[c] * x[c];
            out[std::size_t(token) * rows + r] = sum;
        }
    }
    return out;
}

std::vector<double> oracle(const Ternary& w, std::int32_t first, std::int32_t rows,
                           const std::vector<float>& x, std::int32_t t) {
    return project(w, first, rows, weight_input(w, std::vector<double>(x.begin(), x.end()), t), t);
}

// ops::rmsnorm of the rows x [T][D] in FP64: x / sqrt(mean(x^2) + eps) * gain,
// gain = 1 + weight with unit_offset.
std::vector<double> rmsnorm_oracle(const std::vector<float>& x, const std::vector<float>& weight,
                                   double eps, bool unit_offset, std::int32_t t) {
    const std::size_t d = weight.size();
    std::vector<double> out(x.size());
    for (std::int32_t token = 0; token < t; ++token) {
        const float* row = &x[std::size_t(token) * d];
        double squares   = 0;
        for (std::size_t i = 0; i < d; ++i) squares += double(row[i]) * row[i];
        const double inverse = 1.0 / std::sqrt(squares / double(d) + eps);
        for (std::size_t i = 0; i < d; ++i) {
            const double gain               = unit_offset ? 1.0 + weight[i] : double(weight[i]);
            out[std::size_t(token) * d + i] = row[i] * inverse * gain;
        }
    }
    return out;
}

std::vector<float> norm_weight(std::int32_t d, std::uint32_t seed) {
    std::vector<float> weight(static_cast<std::size_t>(d));
    fill_uniform(weight, seed, -0.5f, 0.5f);
    round_to_bf16(weight);
    return weight;
}

int linear_case(const Ternary& w, std::int32_t first, std::int32_t rows, std::int32_t t,
                bool graph, ops::LinearPolicy policy = ops::LinearPolicy::AllowA8) {
    const auto x = activation(w.k, t, 17u * t + rows);
    auto device_x = to_device_bf16(x);
    GuardedDeviceBuffer out(std::size_t(rows) * t * 2);
    Tensor tx(device_x.p, DType::BF16, {w.k, t});
    Tensor to(out.data(), DType::BF16, {rows, t});
    const Weight view = w.rows(first, rows);
    DeviceArena workspace(
        ops::linear_workspace_capacity_bytes(QType::T5_G128_FP16, rows, w.k, policy, 1, t) + 256);
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
    const std::string label = "t5 linear [" + std::to_string(rows) + "," + std::to_string(w.k) +
                              "] rows+" + std::to_string(first) + " T=" + std::to_string(t) +
                              policy_name(policy);
    return verify_reduction(label, from_device_bf16(out.data(), std::size_t(rows) * t),
                            oracle(w, first, rows, x, t), criterion(policy)) +
           out.verify_guards(label.c_str());
}

int linear_add_case(const Ternary& w, std::int32_t t,
                    ops::LinearPolicy policy = ops::LinearPolicy::AllowA8) {
    const auto x        = activation(w.k, t, 91u + t);
    auto residual_value = activation(w.n, t, 92u + t);
    auto expected       = oracle(w, 0, w.n, x, t);
    for (std::size_t i = 0; i < expected.size(); ++i) expected[i] += residual_value[i];
    auto device_x        = to_device_bf16(x);
    auto device_residual = to_device_bf16(residual_value);
    Tensor tx(device_x.p, DType::BF16, {w.k, t});
    Tensor tr(device_residual.p, DType::BF16, {w.n, t});
    DeviceArena workspace(
        ops::linear_add_workspace_capacity_bytes(QType::T5_G128_FP16, w.n, w.k, policy, 1, t) +
        256);
    ops::linear_add(tx, w.rows(0, w.n), tr, policy, workspace, nullptr);
    cuda_synchronize();
    return verify_reduction("t5 linear_add [" + std::to_string(w.n) + "," + std::to_string(w.k) +
                                "] T=" + std::to_string(t) + policy_name(policy),
                            from_device_bf16(device_residual.p, expected.size()), expected,
                            criterion(policy));
}

// The fused attention parent is stored query, key, gate, value.
int attention_case(const Ternary& w, std::int32_t t,
                   ops::LinearPolicy policy = ops::LinearPolicy::AllowA8) {
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
        ops::attn_input_proj_workspace_capacity_bytes(QType::T5_G128_FP16, w.n, w.k, policy, 1, t) +
        256);
    ops::attn_input_proj(tx, w.rows(0, w.n), tensors[0], tensors[2], tensors[1], tensors[3],
                         policy, workspace, nullptr);
    cuda_synchronize();
    int failures = 0, first = 0;
    const std::array<const char*, 4> names{"query", "key", "gate", "value"};
    for (int i = 0; i < 4; ++i) {
        failures += verify_reduction(std::string("t5 attn_input_proj ") + names[i] + " T=" +
                                         std::to_string(t) +
                                         policy_name(policy),
                                     from_device_bf16(outputs[i].p, std::size_t(rows[i]) * t),
                                     oracle(w, first, rows[i], x, t), criterion(policy));
        first += rows[i];
    }
    return failures;
}

// RMSNorm-input form: the projections of rmsnorm(x) for the raw rows x, normalized, rotated and
// quantized in one kernel. The FP64 oracle normalizes the represented x, then rotates and projects.
int attention_rmsnorm_case(const Ternary& w, std::int32_t t, bool unit_offset,
                           ops::LinearPolicy policy = ops::LinearPolicy::AllowA8) {
    constexpr float kEps = 1e-6f;
    const auto x         = activation(w.k, t, 81u + t);
    const auto gain      = norm_weight(w.k, 82u + t);
    auto device_x        = to_device_bf16(x);
    auto device_gain     = to_device_bf16(gain);
    const auto input     = weight_input(w, rmsnorm_oracle(x, gain, kEps, unit_offset, t), t);
    const std::array<std::int32_t, 4> rows{6144, 1024, 6144, 1024};
    std::vector<GuardedDeviceBuffer> outputs;
    outputs.reserve(4);
    std::array<Tensor, 4> tensors;
    for (int i = 0; i < 4; ++i) {
        outputs.emplace_back(std::size_t(rows[i]) * t * 2);
        tensors[i] = Tensor(outputs[i].data(), DType::BF16, {rows[i], t});
    }
    Tensor tx(device_x.p, DType::BF16, {w.k, t});
    const ops::RmsNormPrologue norm{Tensor(device_gain.p, DType::BF16, {w.k}), kEps, unit_offset};
    DeviceArena workspace(
        ops::attn_input_proj_workspace_capacity_bytes(QType::T5_G128_FP16, w.n, w.k, policy, 1, t) +
        256);
    ops::attn_input_proj(tx, norm, w.rows(0, w.n), tensors[0], tensors[2], tensors[1], tensors[3],
                         policy, workspace, nullptr);
    cuda_synchronize();
    int failures = 0, first = 0;
    const std::array<const char*, 4> names{"query", "key", "gate", "value"};
    for (int i = 0; i < 4; ++i) {
        const std::string label = std::string("t5 attn_input_proj rmsnorm") +
                                  (w.signs.empty() ? "" : " rotated") +
                                  (unit_offset ? " 1+w " : " w ") + names[i] +
                                  " T=" + std::to_string(t) + policy_name(policy);
        failures +=
            verify_reduction(label, from_device_bf16(outputs[i].data(), std::size_t(rows[i]) * t),
                             project(w, first, rows[i], input, t), criterion(policy)) +
            outputs[i].verify_guards(label.c_str());
        first += rows[i];
    }
    return failures;
}

// residual += down(silu(g) * u), [g; u] = gate_up(rmsnorm(residual)): the down projection
// quantizes the SwiGLU of the BF16-staged gate and up rows directly. FP64 oracle throughout.
int mlp_case(const Ternary& gate_up, const Ternary& down, std::int32_t t, bool unit_offset,
             ops::LinearPolicy policy = ops::LinearPolicy::AllowA8) {
    constexpr float kEps = 1e-6f;
    const std::int32_t d = gate_up.k, m = down.k;
    const auto residual = activation(d, t, 101u + t);
    const auto gain     = norm_weight(d, 102u + t);
    const auto input =
        weight_input(gate_up, rmsnorm_oracle(residual, gain, kEps, unit_offset, t), t);
    const auto g = project(gate_up, 0, m, input, t);
    const auto u = project(gate_up, m, m, input, t);
    std::vector<double> activated(g.size());
    for (std::size_t i = 0; i < g.size(); ++i) {
        activated[i] = g[i] / (1.0 + std::exp(-g[i])) * u[i];
    }
    auto expected = project(down, 0, d, weight_input(down, activated, t), t);
    for (std::size_t i = 0; i < expected.size(); ++i) expected[i] += residual[i];

    auto device_residual = to_device_bf16(residual);
    auto device_gain     = to_device_bf16(gain);
    Tensor tr(device_residual.p, DType::BF16, {d, t});
    const ops::RmsNormPrologue norm{Tensor(device_gain.p, DType::BF16, {d}), kEps, unit_offset};
    DeviceArena workspace(ops::rmsnorm_swiglu_mlp_workspace_capacity_bytes(
                              QType::T5_G128_FP16, gate_up.n, d, policy, policy, 1, t) +
                          256);
    ops::rmsnorm_swiglu_mlp(norm, gate_up.rows(0, gate_up.n), policy, down.rows(0, d), policy, tr,
                            workspace, nullptr);
    cuda_synchronize();
    return verify_reduction(
        "t5 rmsnorm_swiglu_mlp [" + std::to_string(gate_up.n) + "," + std::to_string(d) + "]x[" +
            std::to_string(d) + "," + std::to_string(m) + "]" +
            (gate_up.signs.empty() ? "" : " rotated") + (unit_offset ? " 1+w" : " w") +
            " T=" + std::to_string(t) + policy_name(policy),
        from_device_bf16(device_residual.p, expected.size()), expected, criterion(policy));
}

// Embedding gather: the logical row ids[t] of the table, i.e. the decoded stored row or, for a
// rotated table, signs * H(z') / 32 per 1024-column block (FP64 Sylvester oracle).
int embedding_case(const Ternary& w, const std::vector<std::int32_t>& ids) {
    // A row of the logical table W' H S is (z' H) S: the signs follow the butterfly.
    const auto t = static_cast<std::int32_t>(ids.size());
    std::vector<double> expected(std::size_t(w.k) * t);
    for (std::int32_t token = 0; token < t; ++token) {
        for (std::int32_t block = 0; block < w.k; block += 1024) {
            for (int r = 0; r < 1024; ++r) {
                double value = w.weight(ids[token], block + r);
                if (!w.signs.empty()) {
                    double sum = 0;
                    for (int c = 0; c < 1024; ++c) {
                        const double term = w.weight(ids[token], block + c);
                        sum += std::popcount(unsigned(r & c)) & 1 ? -term : term;
                    }
                    value = sum / 32.0 * w.signs[block + r];
                }
                expected[std::size_t(token) * w.k + block + r] = value;
            }
        }
    }
    DeviceBuffer device_ids(ids.size() * 4);
    device_ids.copy_from_host(ids.data(), ids.size() * 4);
    GuardedDeviceBuffer out(std::size_t(w.k) * t * 2);
    Tensor tids(device_ids.p, DType::I32, {t});
    Tensor tout(out.data(), DType::BF16, {w.k, t});
    ops::embedding(tids, w.rows(0, w.n), tout, nullptr);
    cuda_synchronize();
    const std::string label = std::string("t5 embedding") + (w.signs.empty() ? "" : " rotated") +
                              " T=" + std::to_string(t);
    return verify_reduction(label, from_device_bf16(out.data(), expected.size()), expected, kA16) +
           out.verify_guards(label.c_str());
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    {
        // GEMV templates T = 1..4; the small-T MMA route from T = 5 through its 32-token
        // bound, with one to four 8-token tiles and partial tiles, on a partial 16-row block
        // (odd N) and a row view of a fused parent; beyond T = 32 a weight whose rows are not a
        // multiple of 64 stays on the small-T route in 32-token tiles (T = 40, 72). Graph replay,
        // and the 64-token prefill GEMM with partial tiles (this short-K, few-row weight keeps
        // 64-token CTAs beyond T = 64 as well).
        const Ternary small(301, 3072, 15u);
        for (std::int32_t t : {1, 2, 3, 4, 5, 8, 9, 16, 17, 24, 25, 32, 40, 72}) {
            failures += linear_case(small, 0, 301, t, t == 3 || t == 9);
        }
        failures += linear_case(small, 44, 200, 6, false);
        failures += linear_case(small, 44, 200, 12, false);
        const Ternary gemm(448, 2048, 18u);
        for (std::int32_t t : {9, 17, 32, 33, 64, 65, 130}) {
            failures += linear_case(gemm, 0, 448, t, t == 65);
        }
        failures += linear_case(gemm, 64, 320, 40, false, ops::LinearPolicy::AllowA4);
        failures += linear_add_case(gemm, 70);
        failures += linear_add_case(gemm, 3);
        // Weights of at least 8192 rows take the 16-warp 128 x 128 tile: three token tiles, the
        // last partial, and a residual epilogue.
        const Ternary tall(8192, 2048, 19u);
        failures += linear_case(tall, 0, 8192, 300, false);
        failures += linear_add_case(tall, 130);
    }
    // Bonsai shapes (N, K) at decode and MTP-verify widths: one lane (T = 1, 3, 4), and B
    // concurrent lanes packed as B x (draft + 1) columns (draft 2: 6, 9, 12, ..., 24; draft 3:
    // 8, 12, 16, ..., 32) on the small-T route. The 5120-row shapes (the longest K per warp)
    // take every tile count and the residual epilogue.
    for (const auto [n, k] : std::array<std::pair<int, int>, 4>{
             {{5120, 6144}, {5120, 17408}, {16384, 5120}, {34816, 5120}}}) {
        const Ternary w(n, k, 3000u + n + k);
        for (std::int32_t t : {1, 3, 4, 8}) failures += linear_case(w, 0, n, t, t == 3);
        failures += linear_case(w, 0, n, 9, true);
        failures += linear_case(w, 0, n, 24, false);
        if (n == 5120) {
            for (std::int32_t t : {6, 12, 16, 32}) failures += linear_case(w, 0, n, t, false);
            failures += linear_add_case(w, 3);
            failures += linear_add_case(w, 9);
        }
    }
    {
        // 128-token prefill GEMM at the longest K (mlp down, 80 CTAs), accumulating into a
        // residual.
        const Ternary down(5120, 17408, 3100u);
        failures += linear_add_case(down, 72);
    }
    {
        const Ternary attention(14336, 5120, 78u);
        failures += attention_case(attention, 3);
        failures += attention_case(attention, 6);
        failures += attention_case(attention, 9);  // four outputs from the small-T route
        failures += attention_case(attention, 72); // four outputs from the 128-token GEMM
    }
    {
        // RMSNorm-input attention parent (all 16 Bonsai full-attention layers): GEMV and small-T
        // route at decode and MTP-verify widths, the 128-token GEMM, and an unrotated parent
        // without the unit offset.
        Ternary attention(14336, 5120, 79u);
        failures += attention_rmsnorm_case(attention, 3, false);
        attention.rotate(80u);
        for (std::int32_t t : {1, 3, 8, 12, 72}) {
            failures += attention_rmsnorm_case(attention, t, true);
        }
    }
    {
        // Normalized SwiGLU MLP: both fused prologues (RMSNorm into gate/up, SwiGLU into down),
        // unrotated and rotated, GEMV, small-T (down K = 1024: two groups per warp) and 64-token
        // GEMM widths.
        Ternary gate_up(2048, 1024, 111u), down(1024, 1024, 112u);
        failures += mlp_case(gate_up, down, 3, false);
        gate_up.rotate(113u);
        down.rotate(114u);
        for (std::int32_t t : {1, 3, 9, 65}) failures += mlp_case(gate_up, down, t, true);
    }
    {
        // Bonsai MLP: gate/up [34816,5120], down [5120,17408], rotated; GEMV (T = 1, 3) and
        // small-T route (T = 8, 16).
        Ternary gate_up(34816, 5120, 121u), down(5120, 17408, 122u);
        gate_up.rotate(123u);
        down.rotate(124u);
        for (std::int32_t t : {1, 3, 8, 16}) failures += mlp_case(gate_up, down, t, true);
    }
    {
        // The fused forms are registered only for t5 under AllowA8.
        const bool registered =
            ops::attn_input_proj_accepts_rmsnorm(QType::T5_G128_FP16, ops::LinearPolicy::AllowA8) &&
            ops::rmsnorm_swiglu_mlp_accepts(QType::T5_G128_FP16, ops::LinearPolicy::AllowA8,
                                            QType::T5_G128_FP16, ops::LinearPolicy::AllowA4);
        const bool refused =
            !ops::attn_input_proj_accepts_rmsnorm(QType::T5_G128_FP16,
                                                  ops::LinearPolicy::A16Only) &&
            !ops::attn_input_proj_accepts_rmsnorm(QType::Q8_G32_FP16, ops::LinearPolicy::AllowA8) &&
            !ops::rmsnorm_swiglu_mlp_accepts(QType::T5_G128_FP16, ops::LinearPolicy::AllowA8,
                                             QType::Q8_G32_FP16, ops::LinearPolicy::AllowA8);
        if (!registered || !refused) {
            std::cerr << "t5 fused prologue registration mismatch\n";
            ++failures;
        }
    }
    {
        // Rotated weights: the projection rotates its primal input inside the quantization.
        Ternary rotated(448, 2048, 23u);
        rotated.rotate(24u);
        for (std::int32_t t : {1, 3, 8, 12, 32, 65}) {
            failures += linear_case(rotated, 0, 448, t, t == 3);
        }
        failures += linear_add_case(rotated, 3);
        failures += linear_add_case(rotated, 18);
    }
    {
        // Embedding tables: first, last and repeated ids; plain and rotated.
        Ternary table(301, 3072, 33u);
        const std::vector<std::int32_t> ids{0, 300, 7, 7, 150};
        failures += embedding_case(table, ids);
        table.rotate(34u);
        failures += embedding_case(table, ids);
    }
    {
        // t5 has no A16 route: an A16Only projection is refused, not silently computed.
        const Ternary w(64, 1024, 35u);
        auto device_x = to_device_bf16(activation(1024, 2, 36u));
        DeviceBuffer out(64 * 2 * 2);
        Tensor tx(device_x.p, DType::BF16, {1024, 2});
        Tensor to(out.p, DType::BF16, {64, 2});
        DeviceArena workspace(1 << 20);
        bool refused = false;
        try {
            ops::linear(tx, w.rows(0, 64), to, ops::LinearPolicy::A16Only, workspace, nullptr);
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        if (!refused) {
            std::cerr << "t5 linear accepted A16Only\n";
            ++failures;
        }
    }
    std::cout << (failures ? "FAIL" : "OK") << " t5 A8\n";
    return failures ? 1 : 0;
}
