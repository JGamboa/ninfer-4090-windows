#include "core/weight.h"
#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#include "core/device.h"
#include "ops/linear/q5/q5_ksplit_mma.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <int InputRows, int TileCols>
void launch_ksplit(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    constexpr int kOutputRows = 5120;
    const dim3 grid(static_cast<unsigned>(kOutputRows / Q5KSplitMmaSchedule::kRowsPerCta));
    const Q5KSplitOutput out{static_cast<__nv_bfloat16*>(residual_out.data), kOutputRows,
                             static_cast<int>(residual_out.nb[1] / sizeof(__nv_bfloat16))};
    q5_ksplit_mma_kernel<InputRows, TileCols, true><<<grid, Q5KSplitMmaSchedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales), out,
        x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <int TileCols>
void dispatch_shape(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    if (residual_out.ne[0] != 5120 || w.qhigh == nullptr) {
        throw std::invalid_argument("q5 linear_add ksplit: unsupported exact problem");
    }
    if (w.k == 6144) {
        launch_ksplit<6144, TileCols>(x, w, residual_out, stream);
    } else if (w.k == 17408) {
        launch_ksplit<17408, TileCols>(x, w, residual_out, stream);
    } else {
        throw std::invalid_argument("q5 linear_add ksplit: unsupported exact K");
    }
}

} // namespace

void q5_linear_add_ksplit_mma_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                     cudaStream_t stream) {
    const std::int32_t cols = x.ne[1];
    if (cols >= 1 && cols <= 8) {
        dispatch_shape<8>(x, w, residual_out, stream);
    } else if (cols <= 16) {
        dispatch_shape<16>(x, w, residual_out, stream);
    } else {
        throw std::invalid_argument("q5 linear_add ksplit: at most 16 columns");
    }
}

} // namespace ninfer::ops::detail
