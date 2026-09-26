#include "core/weight.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"

#include "ops/linear_swiglu/q4/q4_linear_swiglu_gemm_mma.cuh"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/rowsplit_tall_a8_mma.cuh"
#include "ops/common/rowsplit_tall_mma.cuh"
#include "ops/common/token_slices.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using GateUpC40Cfg = GemmCfg<64, 40, 64, 64, 8, 2, 1, false, true, true>;
constexpr std::int32_t kTallTokens = 128;

template <class Cfg, bool Full>
void launch_folded(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr int PM = Cfg::BM / 2;
    const int t      = x.ne[1];
    const dim3 grid(static_cast<unsigned>(div_up(out.ne[0], PM)),
                    static_cast<unsigned>(div_up(t, Cfg::BN)));
    if constexpr (Full) {
        q4_linear_swiglu_mma_split_half_pair_kernel<Cfg, true><<<grid, Cfg::THREADS, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
            out.ne[0], x.ne[0], t, weight.padded_shape[1]);
    } else {
        q4_linear_swiglu_mma_split_half_pair_kernel<Cfg, false><<<grid, Cfg::THREADS, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
            out.ne[0], x.ne[0], t, weight.padded_shape[1]);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <class Cfg>
void launch_route(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const bool full = (x.ne[1] % Cfg::BN) == 0;
    for_each_token_slice(x.ne[1], Cfg::BN, [&](std::int32_t offset, std::int32_t count) {
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor out_slice     = out.slice(1, offset, count);
        if (full) {
            launch_folded<Cfg, true>(x_slice, weight, out_slice, stream);
        } else {
            launch_folded<Cfg, false>(x_slice, weight, out_slice, stream);
        }
    });
}

rowsplit_tall::SwiGluQ4Problem folded_problem(const Weight& weight, Tensor& out, std::int32_t k) {
    const std::int32_t intermediate = out.ne[0];
    if (intermediate % 64 != 0 || k % 64 != 0 || weight.padded_shape[1] != k) {
        throw std::invalid_argument("q4 linear_swiglu: pipelined GEMM shape is unsupported");
    }
    return {static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
            intermediate};
}

} // namespace

void q4_linear_swiglu_a8_mma_folded_pipelined_r64_c128_launch(const A8G64Activation& x,
                                                              const Weight& weight, Tensor& out,
                                                              cudaStream_t stream) {
    const std::int32_t k = x.q.ne[0];
    const auto problem   = folded_problem(weight, out, k);
    rowsplit_tall_a8::launch<kTallTokens>(problem, problem.intermediate / 64,
                                          static_cast<const std::int8_t*>(x.q.data),
                                          static_cast<const float*>(x.scale.data), k, x.q.ne[1],
                                          stream);
}

void q4_linear_swiglu_mma_folded_pipelined_r64_c128_launch(const Tensor& x, const Weight& weight,
                                                           Tensor& out, cudaStream_t stream) {
    const std::int32_t k = x.ne[0];
    const auto problem   = folded_problem(weight, out, k);
    rowsplit_tall::launch<kTallTokens>(problem, problem.intermediate / 64,
                                       static_cast<const __nv_bfloat16*>(x.data), k, x.ne[1],
                                       stream);
}

void q4_linear_swiglu_mma_folded_pipelined_r64_c128_tail_launch(const Tensor& x,
                                                                const Weight& weight, Tensor& out,
                                                                cudaStream_t stream) {
    // The pipelined kernel serves a wide extent in 128-column tiles. A remainder of at most
    // kNarrowTailCols columns measured cheaper on the narrow routes than a 128-wide tile that
    // would be almost entirely empty, so that remainder is launched on its own; any wider
    // remainder keeps the single wide launch.
    constexpr std::int32_t kBlockCols      = kTallTokens;
    constexpr std::int32_t kNarrowTailCols = 40;
    constexpr std::int32_t kExactTailCols  = 24;

    const std::int32_t tokens = x.ne[1];
    const std::int32_t blocks = tokens / kBlockCols;
    const std::int32_t tail   = tokens - blocks * kBlockCols;
    if (blocks == 0 || tail == 0 || tail > kNarrowTailCols) {
        q4_linear_swiglu_mma_folded_pipelined_r64_c128_launch(x, weight, out, stream);
        return;
    }

    const std::int32_t wide_cols = blocks * kBlockCols;
    const Tensor x_wide = x.slice(1, 0, wide_cols);
    Tensor out_wide     = out.slice(1, 0, wide_cols);
    q4_linear_swiglu_mma_folded_pipelined_r64_c128_launch(x_wide, weight, out_wide, stream);
    const Tensor x_tail = x.slice(1, wide_cols, tail);
    Tensor out_tail     = out.slice(1, wide_cols, tail);
    if (tail == 1) {
        // The exact small-T tiled route starts at two columns.
        q4_linear_swiglu_gemv_pair_launch(x_tail, weight, out_tail, stream);
    } else if (tail <= kExactTailCols) {
        q4_linear_swiglu_small_t_tiled_launch(x_tail, weight, out_tail, stream);
    } else {
        launch_route<GateUpC40Cfg>(x_tail, weight, out_tail, stream);
    }
}

} // namespace ninfer::ops::detail
