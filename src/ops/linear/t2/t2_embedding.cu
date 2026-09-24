#include "ops/linear/t2/t2_project.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock   = 1024;
constexpr int kThreads = 256; // one code byte (four columns) per thread

// One CTA per (1024-column block, token). Decodes the stored ternary row block, then, for a
// rotated table (Weight::input_signs), applies the logical W' H S row transform: the FP32
// shared-memory Walsh-Hadamard butterfly of ops/hadamard, the 1/32 normalization and the
// signs. One BF16 rounding of the result.
__global__ void __launch_bounds__(kThreads)
    t2_embedding_kernel(const std::int32_t* __restrict__ ids, const std::uint8_t* __restrict__ codes,
                        const __half* __restrict__ scales, std::int64_t scale_row_halves,
                        const __nv_bfloat16* __restrict__ signs, int k,
                        __nv_bfloat16* __restrict__ out) {
    __shared__ float values[kBlock];
    const int tid     = static_cast<int>(threadIdx.x);
    const int column0 = static_cast<int>(blockIdx.x) * kBlock;
    const int token   = static_cast<int>(blockIdx.y);
    const std::int64_t row = ids[token];
    const std::uint8_t byte = codes[row * (k / 4) + column0 / 4 + tid];
    const float scale = __half2float(scales[row * scale_row_halves + (column0 + 4 * tid) / 128]);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        // Code 3 is invalid in stored data; like the projections it decodes as +2.
        values[4 * tid + j] = static_cast<float>(static_cast<int>((byte >> (2 * j)) & 3u) - 1) * scale;
    }
    __nv_bfloat16* destination = out + std::int64_t(token) * k + column0;
    if (signs == nullptr) {
#pragma unroll
        for (int j = 0; j < 4; ++j) destination[4 * tid + j] = __float2bfloat16_rn(values[4 * tid + j]);
        return;
    }
    __syncthreads();
#pragma unroll
    for (int stride = 1; stride < kBlock; stride <<= 1) {
#pragma unroll
        for (int pair = tid; pair < kBlock / 2; pair += kThreads) {
            const int low        = (pair / stride) * 2 * stride + pair % stride;
            const float a        = values[low];
            const float b        = values[low + stride];
            values[low]          = a + b;
            values[low + stride] = a - b;
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < kBlock / kThreads; ++i) {
        const int index = tid + i * kThreads;
        destination[index] = __float2bfloat16_rn(values[index] * 0x1p-5f *
                                                 __bfloat162float(signs[column0 + index]));
    }
}

} // namespace

void t2_embedding(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t stream) {
    validate_t2_weight(table, "embedding");
    const int tokens = ids.ne[0];
    if (ids.dtype != DType::I32 || out.dtype != DType::BF16 || out.ne[0] != table.k ||
        out.ne[1] != tokens || !ids.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("embedding: t2 expects I32 ids [T] and BF16 out [K,T]");
    }
    if (tokens == 0) return;
    const dim3 grid(static_cast<unsigned>(table.k / kBlock), static_cast<unsigned>(tokens));
    t2_embedding_kernel<<<grid, kThreads, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), static_cast<const std::uint8_t*>(table.qdata),
        static_cast<const __half*>(table.scales), table.scale_nb[1] / 2,
        static_cast<const __nv_bfloat16*>(table.input_signs), table.k,
        static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
