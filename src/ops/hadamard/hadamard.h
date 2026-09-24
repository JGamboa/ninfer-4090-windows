#pragma once

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Prism input rotation of a ternary projection (docs/maintainer/bonsai-ternary-design.md 1.5):
// y[t][k] = (1/32) * H_1024(signs * x[t])[k], blockwise over consecutive 1024 columns, where H is
// the unnormalized Sylvester Walsh-Hadamard matrix of order 1024. x, y: contiguous BF16
// [width, tokens] (width % 1024 == 0); signs: BF16 +-1 [width]. y may alias x. When perm is a
// device int32 [width] array, the transform reads x[t][perm[k]] instead (y must not alias x).
// Graph-capturable: one launch, grid (width / 1024, tokens), no host synchronization.
void hadamard_1024_launch(const Tensor& x, const Tensor& signs, const std::int32_t* perm,
                          Tensor& y, cudaStream_t stream);

} // namespace ninfer::ops::detail
