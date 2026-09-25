#include "ops/linear/q5/q5_shapes.h"
#include "ops/linear/q5/q5_ksplit_launch.cuh"

namespace ninfer::ops::detail {

// The packed gate/up parent of a Q5 MTP layer. Every route takes its row count at run time,
// so this is the n7168 table over the wider bank (the same compiled instances); its bounds
// are not retuned for this geometry.
Q5Launch select_q5_n34816_k5120(std::int32_t tokens) {
    if (tokens == 1) return launch_q5_split4_c1_k5120;
    if (tokens <= 2) return launch_q5_ksplit<5120, 2, 4>;
    if (tokens <= 3) return launch_q5_ksplit<5120, 3, 4>;
    if (tokens <= 4) return launch_q5_ksplit<5120, 4, 4>;
    if (tokens <= 5) return launch_q5_ksplit<5120, 5, 4>;
    if (tokens <= 6) return launch_q5_ksplit<5120, 6, 4>;
    if (tokens <= 11) return launch_q5_simt_r8_c4;
    if (tokens <= 112) return launch_q5_mma_r64_c32_s3;
    return launch_q5_mma_r64_c128;
}

} // namespace ninfer::ops::detail
