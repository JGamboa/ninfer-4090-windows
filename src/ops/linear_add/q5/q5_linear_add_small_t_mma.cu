#include "core/weight.h"
#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#include "core/device.h"
#include "ops/linear/q5/q5_small_t_mma.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kOutputRows = 5120;

// residual_out[col, row] += W x, in the BF16 residual stream: the FP32 add and the single
// rounding of the SIMT residual epilogues. `ld` is the residual's column stride in elements.
struct Q5SmallTResidualEpilogue {
    __nv_bfloat16* out;
    int ld;
    int columns;

    __device__ __forceinline__ void add(int row, int col, float value) const {
        __nv_bfloat16& slot = out[static_cast<std::int64_t>(col) * ld + row];
        slot                = __float2bfloat16_rn(__bfloat162float(slot) + value);
    }

    __device__ __forceinline__ void store(int row, int col0, float4 v) const {
        if (col0 < columns) {
            add(row, col0, v.x);
            add(row + 8, col0, v.z);
        }
        if (col0 + 1 < columns) {
            add(row, col0 + 1, v.y);
            add(row + 8, col0 + 1, v.w);
        }
    }
};

template <int K, int XCols, int Stages>
void launch(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    const Q5SmallTResidualEpilogue epilogue{
        static_cast<__nv_bfloat16*>(residual_out.data),
        static_cast<int>(residual_out.nb[1] / sizeof(__nv_bfloat16)), x.ne[1]};
    q5_small_t_mma_launch<kOutputRows, K, XCols, Stages, Q5SmallTResidualEpilogue>(
        stream, static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.qhigh),
        static_cast<const std::uint8_t*>(w.scales), epilogue, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

// A 5120-row matrix is 320 sixteen-row tiles. Up to four columns stage a four-column activation
// tile (two-deep ring), five to eight an eight-column one, nine to sixteen a sixteen-column one;
// with one stage the wider tiles keep the most CTAs resident. One kernel, clean L2, RTX 4090, us
// (K = 6144 / 17408; the K-split MMA kernel this replaced took 36.9-41.0 / 93.2-105.5):
//
//   T1-4: 26.6 / 67.6    T5-8: 26.6-27.6 / 68.6    T9-16: 27.6-29.7 / 69.6-72.7
template <int K>
void launch_columns(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    const int columns = x.ne[1];
    if (columns <= 4) {
        launch<K, 4, 2>(x, w, residual_out, stream);
    } else if (columns <= 8) {
        launch<K, 8, 1>(x, w, residual_out, stream);
    } else {
        launch<K, 16, 1>(x, w, residual_out, stream);
    }
}

} // namespace

void q5_linear_add_small_t_mma_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    if (x.ne[1] < 1 || x.ne[1] > 16) {
        throw std::invalid_argument("q5 linear_add small-T MMA: T must be in [1,16]");
    }
    if (residual_out.ne[0] != kOutputRows || w.n != kOutputRows || w.qhigh == nullptr) {
        throw std::invalid_argument("q5 linear_add small-T MMA: unsupported exact problem");
    }
    if (w.k == 6144) {
        launch_columns<6144>(x, w, residual_out, stream);
    } else if (w.k == 17408) {
        launch_columns<17408>(x, w, residual_out, stream);
    } else {
        throw std::invalid_argument("q5 linear_add small-T MMA: unsupported exact K");
    }
}

} // namespace ninfer::ops::detail
