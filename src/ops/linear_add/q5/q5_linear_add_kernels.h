#pragma once

#include "core/weight.h"
#include "core/tensor.h"
#include "ops/common/rowsplit_a8_quantize.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void q5_linear_add_split2_exact_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream);
// T = 1..16 tensor-core split-K route (TileCols - 8 < T for its 8- or 16-column tile).
void q5_linear_add_ksplit_mma_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
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
