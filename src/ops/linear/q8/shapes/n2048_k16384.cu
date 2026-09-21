#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_ksplit_launch.cuh"
#include "ops/linear/q8/q8_ksplit_grouped_mma.cuh"

#include <algorithm>

namespace ninfer::ops::detail {
namespace {
using Geometry = Q8N2048K16384;
using Access   = Q8KSplitScaleAccess;
using Stage    = Q8KSplitActivationStage;
#if defined(NINFER_SM86)
// sm_86/sm_89 occupancy differs enough from the sm_120a target these brackets were tuned on that
// the fork's original exact-T route forced every bracket onto a flat 4-warp/2-block K-split
// rather than trust the wider (16/8-warp, 1-block) schedules below; this keeps that same
// conservative choice under upstream's now capacity-based dispatch. Not independently re-tuned.
using C4  = Q8KSplitSchedule<4, 8, 2, Access::Direct, Cache::cg, Cache::cg, Stage::RuntimeActive>;
using C8  = Q8KSplitSchedule<4, 8, 2, Access::Shared, Cache::cg, Cache::cg, Stage::RuntimeActive>;
using C16 = Q8KSplitSchedule<4, 16, 2, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
using C24 = Q8KSplitSchedule<4, 24, 2, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
using C32 = Q8KSplitSchedule<4, 32, 2, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
using C40 = Q8KSplitSchedule<4, 40, 2, Access::Shared, Cache::cg, Cache::cg, Stage::RuntimeActive>;
using C48 = Q8KSplitSchedule<4, 48, 2, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
using C56 = Q8KSplitSchedule<4, 56, 2, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
using C64 = Q8KSplitSchedule<4, 64, 2, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
#else
using C4  = Q8KSplitSchedule<16, 8, 1, Access::Direct, Cache::cg, Cache::cg, Stage::RuntimeActive>;
using C8  = Q8KSplitSchedule<16, 8, 1, Access::Shared, Cache::cg, Cache::cg, Stage::RuntimeActive>;
using C16 = Q8KSplitSchedule<16, 16, 1, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
using C24 = Q8KSplitSchedule<16, 24, 1, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
using C32 = Q8KSplitSchedule<16, 32, 1, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
using C40 = Q8KSplitSchedule<16, 40, 1, Access::Shared, Cache::cg, Cache::cg, Stage::RuntimeActive>;
using C48 = Q8KSplitSchedule<8, 48, 2, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
using C56 = Q8KSplitSchedule<8, 56, 2, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
using C64 = Q8KSplitSchedule<8, 64, 2, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
#endif

template <int Capacity, int KWarps, int TokenGroups>
void launch_grouped(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (weight.padded_shape[1] != Geometry::kInputRows) {
        throw std::invalid_argument(
            "q8 grouped K-split: padded K differs from registered geometry");
    }
    const Q8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
    q8_ksplit_grouped_mma_kernel<Geometry::kInputRows, Capacity, KWarps, TokenGroups, 1>
        <<<Geometry::kOutputRows / 16, KWarps * TokenGroups * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

#if defined(NINFER_SM86)
// The grouped K-split kernel above (launch_grouped) is the same kernel family the fork's medium-T
// DFlash/composite routes refused to launch on sm_86/89 for this geometry; rather than trust its
// 5090-tuned 65..128 brackets unverified, chain the row-per-CTA K-split kernel under its own
// 64-column capacity instead, matching the fork's original chunked fallback.
void launch_medium_sm86(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
#endif
} // namespace

Q8Launch select_q8_n2048_k16384(std::int32_t tokens) {
    if (tokens == 1) return launch_q8_gemv_n2048_k16384;
    if (tokens <= 4) return launch_q8_ksplit<Geometry, 4, C4>;
    if (tokens <= 8) return launch_q8_ksplit<Geometry, 8, C8>;
    if (tokens <= 16) return launch_q8_ksplit<Geometry, 16, C16>;
    if (tokens <= 24) return launch_q8_ksplit<Geometry, 24, C24>;
    if (tokens <= 32) return launch_q8_ksplit<Geometry, 32, C32>;
    if (tokens <= 40) return launch_q8_ksplit<Geometry, 40, C40>;
    if (tokens <= 48) return launch_q8_ksplit<Geometry, 48, C48>;
    if (tokens <= 56) return launch_q8_ksplit<Geometry, 56, C56>;
    if (tokens <= 64) return launch_q8_ksplit<Geometry, 64, C64>;
#if defined(NINFER_SM86)
    if (tokens <= 128) return launch_medium_sm86;
#else
    if (tokens <= 80) return launch_grouped<80, 8, 2>;
    if (tokens <= 96) return launch_grouped<96, 4, 6>;
    if (tokens <= 128) return launch_grouped<128, 4, 8>;
#endif

    // Broad throughput regions; each selected MMA handles its own complete and partial tiles.
    if (tokens <= 384) return launch_q8_mma_r32_c64;
    if (tokens <= 480) return launch_q8_mma_r32_c96;
    if (tokens <= 640) return launch_q8_mma_r32_c128;
    if (tokens <= 704) return launch_q8_mma_r48_c64;
    if (tokens <= 960) return launch_q8_mma_r64_c96;
    if (tokens <= 1344) return launch_q8_mma_r128_c64;
    if (tokens <= 1680) return launch_q8_mma_r128_c80;
    if (tokens <= 2016) return launch_q8_mma_r64_c96;
    if (tokens <= 2112) return launch_q8_mma_r96_c96;
    return launch_q8_mma_r64_c128;
}

#if defined(NINFER_SM86)
namespace {
void launch_medium_sm86(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    std::int32_t offset = 0;
    while (offset < x.ne[1]) {
        const std::int32_t count = std::min<std::int32_t>(64, x.ne[1] - offset);
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor out_slice     = out.slice(1, offset, count);
        select_q8_n2048_k16384(count)(x_slice, weight, out_slice, stream);
        offset += count;
    }
}
} // namespace
#endif

} // namespace ninfer::ops::detail
