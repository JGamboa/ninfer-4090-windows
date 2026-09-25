#pragma once

#include "core/device.h" // kTargetSmCount
#include "ops/softmax_attention/common/head_mapping.cuh"

namespace ninfer::ops {

// The small-T partial kernels run two CTAs per SM and their grid is (KVHeads, splits, batch), so
// one resident wave on the build's target device is 2 * SMs / KVHeads splits. Upstream's literal
// cap of 85 splits was one wave of the RTX 5090's 170 SMs (4 KV heads x 85 = 340 = 2 x 170); on
// the RTX 4090 it launched 340 CTAs into 256 slots, a second wave at one-third occupancy at every
// depth that reached the cap (design 9.1, 128K decode profile). Past one wave, split counts round
// up to whole waves, and the cap is two waves: one wave would exceed the 3968 keys a split can
// stage near the 262144-key limit. The host split policy (small_t.cu) and the device mirror
// (small_t.cuh) both use these definitions.
inline constexpr int kCausalSmallTCtasPerSm       = 2;
inline constexpr int kCausalSmallTMaxKeysPerSplit = 3968;
inline constexpr int kCausalSmallTMaxKeys         = 262144;

template <int QHeadsValue, int KVHeadsValue, int SmallTSplitScaleValue>
struct CausalAttentionGeometry : AttentionHeadMapping<QHeadsValue, KVHeadsValue> {
    static_assert(SmallTSplitScaleValue > 0);

    static constexpr int SmallTSplitScale    = SmallTSplitScaleValue;
    static constexpr int SmallTWaveSplits    = kCausalSmallTCtasPerSm * kTargetSmCount / KVHeadsValue;
    static constexpr int SmallTMaximumSplits = 2 * SmallTWaveSplits;
    static_assert(SmallTWaveSplits % SmallTSplitScale == 0);
    static_assert(SmallTMaximumSplits * kCausalSmallTMaxKeysPerSplit >= kCausalSmallTMaxKeys);
};

// Rounds a split count past one wave up to whole waves, then applies the cap.
template <typename Geometry>
__host__ __device__ constexpr int causal_small_t_wave_splits(int splits) {
    constexpr int kWave = Geometry::SmallTWaveSplits;
    if (splits > kWave) { splits = (splits + kWave - 1) / kWave * kWave; }
    return splits < Geometry::SmallTMaximumSplits ? splits : Geometry::SmallTMaximumSplits;
}

using CausalD256H24Kv4 = CausalAttentionGeometry<24, 4, 1>;
using CausalD256H16Kv2 = CausalAttentionGeometry<16, 2, 2>;

} // namespace ninfer::ops
