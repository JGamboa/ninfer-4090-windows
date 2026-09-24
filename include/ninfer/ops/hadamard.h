#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Prism input rotation of a ternary (T5_G128_FP16) projection:
 *
 *   y[t][k] = (1/32) * sum_j H[k mod 1024][j] * signs[b + j] * x[t][b + j],  b = 1024 * (k / 1024),
 *
 * with H the unnormalized Sylvester Walsh-Hadamard matrix of order 1024 (H[r][c] =
 * (-1)^popcount(r & c)). x and y are contiguous BF16 [K,T] with K % 1024 == 0; signs is BF16 +-1
 * [K]; y may alias x. FP32 arithmetic, one BF16 rounding of y. No workspace; graph-capturable.
 */
void hadamard_1024(const Tensor& x, const Tensor& signs, Tensor& y, cudaStream_t stream);

} // namespace ninfer::ops
