#pragma once

#include "core/weight.h"
#include "core/tensor.h"
#include "ops/common/rowsplit_a8_quantize.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void q5_linear_add_split2_exact_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream);
// T = 1..16 small-T tensor-core route (q5_small_t_mma.cuh).
void q5_linear_add_small_t_mma_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream);
void q5_linear_add_mma_r64_c16_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream);
void q5_linear_add_mma_r64_c24_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream);
void q5_linear_add_mma_r64_c32_s3_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                         cudaStream_t stream);
void q5_linear_add_mma_r64_c32_s4_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                         cudaStream_t stream);
void q5_linear_add_mma_pipelined_r128_c64_launch(const Tensor& x, const Weight& w,
                                                 Tensor& residual_out, cudaStream_t stream);
// The pipelined tile over an A8 activation (quantized x of the same K and T).
void q5_linear_add_a8_mma_pipelined_r128_c64_launch(const A8G64Activation& x, const Weight& w,
                                                    Tensor& residual_out, cudaStream_t stream);

} // namespace ninfer::ops::detail
