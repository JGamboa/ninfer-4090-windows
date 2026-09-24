#include "ops/hadamard/hadamard.h"

#include "ninfer/ops/hadamard.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock   = 1024;
constexpr int kThreads = 256;
constexpr int kPerLane = kBlock / kThreads;

// Shared-memory butterfly: ten stages over one 1024-column block of one token. The standalone
// kernel is bandwidth bound (2 bytes in, 2 out per element); fused prologues replace it in M4.
__global__ void __launch_bounds__(kThreads)
    hadamard_1024_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ signs,
                         const std::int32_t* __restrict__ perm, __nv_bfloat16* y,
                         std::int64_t width) {
    __shared__ float values[kBlock];
    const std::int64_t column0 = static_cast<std::int64_t>(blockIdx.x) * kBlock;
    const std::int64_t row     = static_cast<std::int64_t>(blockIdx.y) * width;
    const int lane             = static_cast<int>(threadIdx.x);
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
        const int index            = lane + i * kThreads;
        const std::int64_t column  = column0 + index;
        const std::int64_t source  = perm ? perm[column] : column;
        values[index] = __bfloat162float(x[row + source]) * __bfloat162float(signs[column]);
    }
    __syncthreads();
#pragma unroll
    for (int stride = 1; stride < kBlock; stride <<= 1) {
#pragma unroll
        for (int pair = lane; pair < kBlock / 2; pair += kThreads) {
            const int low     = (pair / stride) * 2 * stride + pair % stride;
            const float a     = values[low];
            const float b     = values[low + stride];
            values[low]          = __fadd_rn(a, b);
            values[low + stride] = __fsub_rn(a, b);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
        const int index = lane + i * kThreads;
        y[row + column0 + index] = __float2bfloat16_rn(__fmul_rn(values[index], 0x1p-5f));
    }
}

bool contiguous_2d(const Tensor& t) {
    return t.dtype == DType::BF16 && t.ne[2] == 1 && t.ne[3] == 1 && t.is_contiguous();
}

} // namespace

void hadamard_1024_launch(const Tensor& x, const Tensor& signs, const std::int32_t* perm,
                          Tensor& y, cudaStream_t stream) {
    if (!contiguous_2d(x) || !contiguous_2d(y) || x.ne[0] != y.ne[0] || x.ne[1] != y.ne[1]) {
        throw std::invalid_argument("hadamard_1024: x and y must be equal contiguous BF16 [K,T]");
    }
    if (x.ne[0] % kBlock) {
        throw std::invalid_argument("hadamard_1024: width must be a multiple of 1024");
    }
    if (signs.dtype != DType::BF16 || signs.numel() != x.ne[0] || !signs.is_contiguous()) {
        throw std::invalid_argument("hadamard_1024: signs must be contiguous BF16 [K]");
    }
    if (perm && x.data == y.data) {
        throw std::invalid_argument("hadamard_1024: a gathered input cannot alias the output");
    }
    const dim3 grid(static_cast<unsigned>(x.ne[0] / kBlock), static_cast<unsigned>(x.ne[1]));
    hadamard_1024_kernel<<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(signs.data),
        perm, static_cast<__nv_bfloat16*>(y.data), x.ne[0]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail

namespace ninfer::ops {

void hadamard_1024(const Tensor& x, const Tensor& signs, Tensor& y, cudaStream_t stream) {
    detail::hadamard_1024_launch(x, signs, nullptr, y, stream);
}

} // namespace ninfer::ops
