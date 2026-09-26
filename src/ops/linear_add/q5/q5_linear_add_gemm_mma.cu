#include "core/weight.h"
#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/rowsplit_tall_a8_mma.cuh"
#include "ops/common/rowsplit_tall_mma.cuh"
#include "ops/common/token_slices.h"
#include "ops/linear/q5/q5_rowsplit_gemm_mma.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using MmaR64C16Schedule =
    Q5RowSplitMmaGemmSchedule<64, 16, 64, 16, 8, 2, 3, Q5FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q5ScaleLoad::Pair32>;
using MmaR64C24Schedule =
    Q5RowSplitMmaGemmSchedule<64, 24, 64, 16, 8, 2, 2, Q5FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q5ScaleLoad::Pair32>;
using MmaR64C32S3Schedule =
    Q5RowSplitMmaGemmSchedule<64, 32, 64, 16, 16, 3, 2, Q5FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q5ScaleLoad::Pair32>;
using MmaR64C32S4Schedule =
    Q5RowSplitMmaGemmSchedule<64, 32, 64, 16, 16, 4, 2, Q5FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q5ScaleLoad::Pair32>;
// 64-token tiles: a 5120-row weight has 40 row blocks, so 128-token tiles leave a half-empty last
// wave at T = 512 and 1024 (measured 2-4 % faster than the old kernel, against 24-31 % here).
constexpr int kTallTokens = 64;

template <class Schedule, bool Full>
void launch_kernel(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    const auto* xp              = static_cast<const __nv_bfloat16*>(x.data);
    const auto* codes           = static_cast<const std::uint8_t*>(w.qdata);
    const auto* high            = static_cast<const std::uint8_t*>(w.qhigh);
    const auto* scales          = static_cast<const std::uint8_t*>(w.scales);
    auto* out                   = static_cast<__nv_bfloat16*>(residual_out.data);
    const std::int32_t rows     = residual_out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t cols     = x.ne[1];
    const std::int32_t padded_k = w.padded_shape[1];
    const dim3 grid(static_cast<unsigned>(div_up(rows, Schedule::kBlockRows)),
                    static_cast<unsigned>(div_up(cols, Schedule::kBlockCols)), 1u);

    q5_rowsplit_gemm_mma_kernel<Schedule, Full, Q5MmaEpilogue::CtaCollectiveResidual>
        <<<grid, Schedule::kThreads, 0, stream>>>(xp, codes, high, scales, out, out, rows, k, cols,
                                                  padded_k);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void launch_route(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    const bool full = (w.n % 64) == 0 && (x.ne[1] % Schedule::kBlockCols) == 0 &&
                      w.k == w.padded_shape[1] && (w.k % 64) == 0;
    for_each_token_slice(x.ne[1], Schedule::kBlockCols,
                         [&](std::int32_t offset, std::int32_t count) {
                             const Tensor x_slice  = x.slice(1, offset, count);
                             Tensor residual_slice = residual_out.slice(1, offset, count);
                             if (full) {
                                 launch_kernel<Schedule, true>(x_slice, w, residual_slice, stream);
                             } else {
                                 launch_kernel<Schedule, false>(x_slice, w, residual_slice, stream);
                             }
                         });
}

} // namespace

void q5_linear_add_mma_r64_c16_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    launch_route<MmaR64C16Schedule>(x, w, residual_out, stream);
}

void q5_linear_add_mma_r64_c24_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    launch_route<MmaR64C24Schedule>(x, w, residual_out, stream);
}

void q5_linear_add_mma_r64_c32_s3_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                         cudaStream_t stream) {
    launch_route<MmaR64C32S3Schedule>(x, w, residual_out, stream);
}

void q5_linear_add_mma_r64_c32_s4_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                         cudaStream_t stream) {
    launch_route<MmaR64C32S4Schedule>(x, w, residual_out, stream);
}

namespace {

rowsplit_tall::ResidualQ5Problem residual_problem(const Weight& w, Tensor& residual_out,
                                                  std::int32_t k) {
    const std::int32_t rows = residual_out.ne[0];
    if (w.qtype != QType::Q5_G64_FP16 || rows != w.n || rows % rowsplit_tall::kRows != 0 ||
        w.padded_shape[1] != w.k || w.k != k || w.k % rowsplit_tall::kStepK != 0) {
        throw std::invalid_argument("q5 linear_add: pipelined GEMM shape is unsupported");
    }
    return {static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.qhigh),
            static_cast<const std::uint8_t*>(w.scales),
            static_cast<__nv_bfloat16*>(residual_out.data), rows};
}

} // namespace

void q5_linear_add_mma_pipelined_r128_c64_launch(const Tensor& x, const Weight& w,
                                                 Tensor& residual_out, cudaStream_t stream) {
    const auto problem = residual_problem(w, residual_out, x.ne[0]);
    rowsplit_tall::launch<kTallTokens>(problem, problem.rows / rowsplit_tall::kRows,
                                       static_cast<const __nv_bfloat16*>(x.data), x.ne[0], x.ne[1],
                                       stream);
}

void q5_linear_add_a8_mma_pipelined_r128_c64_launch(const A8G64Activation& x, const Weight& w,
                                                    Tensor& residual_out, cudaStream_t stream) {
    const auto problem            = residual_problem(w, residual_out, x.q.ne[0]);
    const std::int32_t row_blocks = problem.rows / rowsplit_tall::kRows;
    const std::int32_t tokens     = x.q.ne[1];
    // The 5120-row weights have only 40 row blocks, so the token tile sets the number of
    // one-per-SM waves. A 128-token CTA costs ~1.55x a 64-token one (measured at k = 17408 and
    // 6144, T = 256..4096): take it when its waves cost less, e.g. from T = 640 on and at 256.
    const auto waves = [&](std::int32_t tile) {
        const std::int64_t ctas = static_cast<std::int64_t>(row_blocks) * div_up(tokens, tile);
        return (ctas + device_sm_count() - 1) / device_sm_count();
    };
    const auto* qx      = static_cast<const std::int8_t*>(x.q.data);
    const auto* x_scale = static_cast<const float*>(x.scale.data);
    if (static_cast<double>(waves(128)) * 1.55 < static_cast<double>(waves(64))) {
        rowsplit_tall_a8::launch<128>(problem, row_blocks, qx, x_scale, x.q.ne[0], tokens, stream);
    } else {
        rowsplit_tall_a8::launch<64>(problem, row_blocks, qx, x_scale, x.q.ne[0], tokens, stream);
    }
}

} // namespace ninfer::ops::detail
