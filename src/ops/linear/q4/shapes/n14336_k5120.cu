#include "ops/linear/q4/q4_shapes.h"
#include "ops/linear/q4/q4_ksplit_launch.cuh"
#include "ops/linear/q4/q4_mma_launch.cuh"

namespace ninfer::ops::detail {
namespace {

// Field-for-field the instantiations other Q4 shapes already compile (n7168_k5120.cu,
// n34816_k5120.cu); only the K-split capacities are new instances of this row count.
using MmaR32C32 = Q4RowSplitMmaGemmSchedule<32, 32, 64, 16, 16, 3, 2, Q4FragmentPipeline::Serial,
                                            Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
using MmaR32C64 = Q4RowSplitMmaGemmSchedule<32, 64, 64, 16, 32, 2, 2, Q4FragmentPipeline::Serial,
                                            Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;

} // namespace

// The packed Q/K/gate/V parent of a Q4 MTP layer. The K-split grid is Rows/16 = 896 CTAs, a
// full device already at capacity 4, so the tiers follow the wide n34816 table; the bounds are
// not retuned for this geometry.
Q4Launch select_q4_n14336_k5120(std::int32_t tokens) {
    if (tokens == 1) return launch_q4_gemv_r1_q8_direct;
    if (tokens <= 4) return launch_q4_ksplit<14336, 5120, 4>;
    if (tokens <= 8) return launch_q4_ksplit<14336, 5120, 8>;
    if (tokens <= 16) return launch_q4_ksplit<14336, 5120, 16>;
    if (tokens <= 24) return launch_q4_ksplit<14336, 5120, 24>;
    if (tokens <= 32) return launch_q4_mma<MmaR32C32>;
    if (tokens <= 64) return launch_q4_mma<MmaR32C64>;
    return launch_q4_mma_r64_c128;
}

} // namespace ninfer::ops::detail
