#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::ops::detail {

// out_s = W x for a T2_G128_FP16 TernaryRowK128 weight W [N,K] and BF16 x [K,T] (K % 1024 == 0,
// see input_signs below). The N parent rows are written in order to the
// contiguous BF16 outputs [rows_s, T] (sum rows_s == N, at most four), e.g. query/key/gate/value
// of one fused parent. With accumulate, out_s += W x in FP32 before one BF16 rounding (the
// residual epilogue of linear_add). FP32 accumulation, one FP16 scale per 128-column group.
//
// With w.input_signs (a Prism weight stored in the rotated basis) x is the primal input and the
// projection applies the rotation itself: fused into the int8 quantization under A8, a rotated
// BF16 workspace copy under A16 (a workspace is then required).
//
// A16 (A16Only, or no workspace): x is consumed as BF16. A8 (AllowA8/AllowA4 with a workspace
// of t2_workspace_capacity_bytes): x is first quantized to int8 with one FP32 scale per token
// and 128-column group (t2_a8.cuh), then multiplied in integer arithmetic (dp4a GEMV through
// T = 8, int8 MMA GEMM beyond). Graph-capturable: static grid per (N, T), no host sync.
void t2_project(const Tensor& x, const Weight& w, std::span<Tensor* const> outputs,
                bool accumulate, LinearPolicy policy, WorkspaceArena* workspace,
                cudaStream_t stream);

// Transient bytes of t2_project for any T <= max_tokens under `policy`, rotated or not.
std::size_t t2_workspace_capacity_bytes(LinearPolicy policy, std::int32_t input_rows,
                                        std::int32_t max_tokens);

// Throws unless w is a resident T2_G128_FP16 TernaryRowK128 view with K % 1024 == 0.
void validate_t2_weight(const Weight& w, const char* op);

} // namespace ninfer::ops::detail
