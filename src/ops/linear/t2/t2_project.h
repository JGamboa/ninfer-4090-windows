#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <span>

namespace ninfer::ops::detail {

// out_s = W x for a T2_G128_FP16 TernaryRowK128 weight W [N,K] and BF16 x [K,T] (K % 1024 == 0,
// already Hadamard-rotated by the caller). The N parent rows are written in order to the
// contiguous BF16 outputs [rows_s, T] (sum rows_s == N, at most four), e.g. query/key/gate/value
// of one fused parent. With accumulate, out_s += W x in FP32 before one BF16 rounding (the
// residual epilogue of linear_add). FP32 accumulation, one FP16 scale per 128-column group.
// Graph-capturable: static grid per (N, T), no workspace, no host synchronization.
void t2_project(const Tensor& x, const Weight& w, std::span<Tensor* const> outputs,
                bool accumulate, cudaStream_t stream);

// Throws unless w is a resident T2_G128_FP16 TernaryRowK128 view with K % 1024 == 0.
void validate_t2_weight(const Weight& w, const char* op);

} // namespace ninfer::ops::detail
