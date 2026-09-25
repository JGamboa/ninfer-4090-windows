# N-gram speculation: remaining phases (temporary)

Status: phase 1 is complete and its contracts live in their references. This file keeps only the
optional phases and their decision gate; delete it when they are built or abandoned.

## Phase 1 (done, 2026-09-25)

`--ngram chain` extends each MTP round with host drafts in the style of llama.cpp `ngram-mod`, up
to a verify window of 15. Where it is documented:

- Behaviour and flags: [CLI](../cli.md#speculative-decoding), [HTTP serving](../serving.md) (flags,
  request-log schema 22, `/metrics`).
- Program ownership, round widths and graph families: [Engine architecture](engine-architecture.md)
  section 8.
- Record views per round width: [ReplaySSM GDN](replayssm-gdn.md) section 3.2.
- Pool: `src/models/qwen3_5/program/speculative/ngram_pool.h`; chain and width policy:
  `ngram_policy.h` next to it; simulator: `tools/spec_sim/`.
- Validation and measurements: [Bonsai ternary design](bonsai-ternary-design.md) section 9.1,
  items 16-20, and `WINDOWS_PORT.md` (Qwen3.8).

## Phase 1b (optional): `--spec ngram` without MTP

A backend whose round is target verify and accept only, for artifacts without MTP weights. It
reuses the Phase 1 ingress and chain policy with an empty MTP proposal. Effort: about 2 days.

## Phase 2: verification wider than 16

Build this only if the simulator shows that cap 32 or 64 beats cap 15 on the user's recorded
sessions by a margin that justifies the work: a large share of tokens committed by rounds that
accept more than 15 drafts (`>15 tok`).

### Every current width limit on the path

Product and Program:

- `validate_speculative_cli_options`: MTP `1..5`, DFlash/DFlash2 `1..15`
  (`src/product/speculative_options.h`).
- `kMtpVerifyMaximumDrafts = 15`, `kDFlashDecodeMaximumDrafts = 15`, their widths and the fixed
  ingress/egress arrays (`src/models/qwen3_5/program/round_buffers.h`);
  `kMaximumMtpVerifyDrafts = 15` (`src/models/qwen3_5/program/internal.h`).
- `TextContext::target_verify_batch_impl`: `width <= kDFlashDecodeMaximumWidth`;
  `mtp_forward_decode_batch`: `width <= 16` (`src/models/qwen3_5/execution/text.cpp`).
- ReplaySSM record width and KV page slack from `draft_window`
  (`src/models/qwen3_5/program/planning/startup.cpp`); graph profiles
  (`planning/graph_profiles.cpp`); `accepted_per_position` (`program/decode.cpp`).

Target-verify Ops:

- `causal_softmax_attention`: `kMaximumVerifyTokens = 16` for `B>1`; `B=1` wider rows already
  take the prompt route, and the workspace query only sizes widths up to 16
  (`src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp`).
- `gdn_input_proj_conv_record`: `2 <= T <= 16` for every batch (`require_record_input` in
  `src/ops/wrapper/gdn_input_proj.cpp`, also its workspace query). This blocks even `B=1`.
- `gdn_input_proj_conv_snapshot`: `T <= 16` when `B>1` (same file).
- `gated_delta_net_replay_record` and `GdnReplayFoldPlan`: `2 <= width <= 16`
  (`src/ops/linear_attention/gated_delta_net/replay.cpp`). Fold must stay a bitwise clone of the
  verify recurrence at the new width ([ReplaySSM GDN](replayssm-gdn.md) section 4).
- `causal_conv1d_silu`: `T <= 16` when `B>1` (`src/ops/wrapper/causal_conv1d_silu.cpp`).
- `bf16` fused GDN norm/control (`gdn_gating_proj`): `T <= 16` on its fused route
  (`src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.cu`).
- Sampling: `kSamplerMaxColumns = 16` (`src/ops/common/sampling_workspace.h`) bounds the
  multi-block scratch route of `speculative_accept_greedy_drafts` and its workspace query
  (`src/ops/wrapper/sampling.cpp`). Wider rounds fall back to the single-block kernel in
  `src/ops/kernel/speculative_round.cuh`, which is correct but scans the whole vocabulary with one
  block per row; the limit should be raised with the rest.
- `mtp_prepare_next_round`: `T <= 16` (`include/ninfer/ops/mtp_round.h`).
- Linear, attention-input, GDN-input and FFN GEMMs dispatch by aggregate column count and have
  prefill routes above 16, so they are functional. Their small-T routes stop at 16, and the first
  columns past a route boundary are slow (5120 x 17408 at T=17 measured 300 us against 118 us at
  T=16, `WINDOWS_PORT.md`), so T=17..65 needs route tuning to reach the cost model.

DFlash-only limits, not on this path: `speculative_accept_sparse_drafts` `K <= 15`,
`prepare_masked_block` `W <= 16`, `dynamic_grouped_conv` `2..16`, `rmsnorm_pack_tail` `2..16`,
`context_softmax_attention` and `sliding_window_attention` `T <= 16`.

Memory at the 27B geometry: ReplaySSM records 1.71 MiB per column per lane (111 MiB at `T=65`,
one lane; 887 MiB at eight lanes); target logits BF16 `[248320, T, B]` 30.8 MiB at `T=65`, one lane.

### Option A: widen the record/fold route to 64, one lane first

Raise the record-side Ops (`gdn_input_proj_conv_record`, `gated_delta_net_replay_record`, fold)
to `T <= 65` at `B=1`, with oracle qualification at the new widths. Attention and snapshot-side
Ops already accept `B=1` beyond 16. Wide rounds run only when one lane is active; with more
lanes the round falls back to the Phase 1 width. The transaction and rollback semantics are
unchanged. Cost per round is `C(T)` whatever is accepted. Effort: 6-10 days, including route
tuning for T=17..65 and the graph widths.

### Option B: prefill-route wide verify with state snapshot and replay

Run a wide round as a prefill chunk that reads the committed state slot and writes a scratch
destination slot (the existing source/destination slot pair), with logits for every column. On
full acceptance, publish the destination. On rejection at `A`, drop the destination and prefill
the `A+1` committed tokens from the untouched source slot. Attention KV beyond the committed
frontier is simply invalid. No record Op changes, but a partially accepted round costs
`C(T) + C(A+1)`, and n-gram rounds are partially accepted at the end of every copied span. The
simulator models this with `--wide-verify replay`. Effort: 4-6 days.

Prefer A unless the simulator shows B within a few percent of A on the user's sessions.

### Option C: several lanes wide

Raise the `B>1` limits (attention small-T, conv, snapshot) as well. Only if recorded multi-agent
sessions show a gain.

## Decision gate

1. Collect sessions (see `tools/spec_sim/README.md`; `--request-log-jsonl` has no token ids or
   text) and run the simulator with `--ngram-n 8,12,24 --caps 15,32,64` for `mtp`,
   `chain:ngram-mod` and `select:ngram-mod`, both `--wide-verify` models.
2. Phase 1 went ahead on the simulator's +28 % projection and is built; its measured gains are
   in the design notes.
3. Phase 2 goes ahead if cap 32/64 beat cap 15 for the same policy clearly, with a substantial
   `>15 tok` share.
4. Rerun the simulator with measured Phase 1 round costs (`--cost-points`) before starting
   Phase 2.
