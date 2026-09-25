# N-gram speculation: integration plan (temporary)

Status: active plan. Delete this file when the work is done or abandoned; stable contracts then move
to [Engine architecture](engine-architecture.md), [Qwen3.5 model](qwen3_5-model.md),
[ReplaySSM GDN](replayssm-gdn.md), `docs/cli.md` and `docs/serving.md`.

Goal: draft-free n-gram drafts in the style of llama.cpp `ngram-mod` for coding-agent workloads
(file rewrites, tool-call JSON, paths), chained with MTP, and a data-driven decision on verifying
more than 16 tokens per round.

## What exists (groundwork)

- `tools/spec_sim/`: offline greedy replay of recorded sequences under `mtp`, `ngram-simple`,
  `ngram-mod`, `select:*` and `chain:*` policies, with caps 15/32/64 and a stated round-cost
  model. It is the decision tool for every phase below; see its README.
- `src/models/qwen3_5/program/speculative/ngram_pool.h`: `NgramDraftPool`, a host-only,
  allocation-bounded hash pool (one table allocated at construction, 4 bytes per entry; `observe`
  and `propose` never allocate). Test: `tests/models/qwen3_5/test_ngram_pool.cpp`
  (`ninfer_qwen3_5_ngram_pool_test`). It is not wired into the runtime.

The pool maps `h(last n tokens)` to the latest continuation. Entries carry a 14-bit tag from the
mixed hash (token ids use 18 bits, which covers the 248,077-id Qwen domain), so a slot collision is
usually a miss instead of a wrong draft. The simulator implements the identical hash, slot, tag and
encoding, and both tests pin the same vector, so simulated collisions are runtime collisions.

## Which verify round takes an external draft

| Backend | Draft source | Can take host drafts |
|---|---|---|
| MTP | host: `MtpDecodeIngress::current_drafts[C*V]` and `current_extents[C]` are filled from `SequenceState::mtp_drafts` by `ProgramImpl::decode_mtp_batch` (`program/decode.cpp`); the previous round's `MtpDecodeEgress::next_drafts` land there in `ProgramImpl::resolve_pending_raw` (`program/prefill.cpp`) | yes, unchanged data path |
| DFlash, DFlash2 | device: the drafter writes `draft_tokens` inside the same graph (`execution/draft.cpp`); `DFlashDecodeIngress` has no draft field | no |

The MTP round therefore already verifies externally supplied drafts. Its device body
(`mtp_decode_batch_body` in `program/speculative/mtp.cpp`) is:

1. `speculative_prepare_verify_inputs` builds `[W,B]` ids (round width `W <= V+1`) and positions from anchors,
   `current_drafts` and `current_extents`;
2. `target_verify_accept` (`program/speculative/target_verification.cpp`) runs
   `TextContext::target_verify_batch` with GDN `RecordForReplay`, then
   `speculative_accept_greedy_drafts` and selects the continuation hidden;
3. `mtp_prepare_next_round` + `mtp_forward_decode_batch` align the MTP head on the accepted
   columns and `mtp_propose_batch` produces the next round's k drafts into egress.

Nothing in steps 1-3 depends on where the drafts came from. Step 3 conditions on the verified
prefix, so MTP keeps proposing correctly after an n-gram round. Acceptance is lossless for any
deterministic proposal: greedy rows accept the longest prefix equal to the target argmax, and
stochastic rows accept draft `i` with probability `p_i(draft_i)`, which is exact rejection sampling
for a one-hot proposal. N-gram drafts are one-hot.

Before Phase 1 stage 1, one startup `draft_window` K set both the verify width `K+1` and the MTP
autoregressive depth. Stage 1 (below) split them; the MTP depth limit stays 5.

## Phase 1: pool drafts through the MTP round, verify window up to 15

Progress: stage 1 (split `V` from `k`, tests) is done and recorded in
[Bonsai ternary design](bonsai-ternary-design.md) section 9.1 item 16; the Program still runs
`V = k`. Stage 2 is the pool, the draft policy and the second graph width; stage 3 is flags, logs
and measurement.

### Model/Program changes

- Split the MTP window into verify window `V` (drafts per round, `1..15`) and MTP depth `k`
  (`1..5`, the existing `--draft-tokens`). `RoundStateSpec::verify_window`, the MTP decode layout,
  `MtpDecodeIngress::current_drafts`/`target_rope_positions` and
  `MtpDecodeEgress::licensed_tokens` use `V`; `next_drafts` and the MTP prefill tensors keep `k`.
  A round of width `W <= V+1` uses exact `[W,B]` views over the `V`-sized buffers. (Done.)
- `mtp_prepare_next_round` takes the verify width `T = V+1` and the proposal depth `k`
  (`1 <= k <= 5`, `k+1 <= T <= 16`) for `next_extents` and the AR position rows. Its oracle test
  covers every `(V,k)`. (Done.)
- `mtp_forward_decode_batch` admits alignment width up to 16. It uses the same attention/FFN Ops
  as target verify, which already support `T<=16`, `B<=8`. (Done.)
- ReplaySSM records (`planning/startup.cpp`, `GdnReplayRecordSpec::width`) are sized for `V+1`:
  27.3 MiB per lane at `T=16` for the 27B geometry ([ReplaySSM GDN](replayssm-gdn.md) section 6),
  218 MiB at eight lanes. (Done.)
- Target KV for a round is mapped through `frontier + extent + 1`, which covers any `extent <= V`.
  The MTP KV end (`frontier + extent + k`) and its page slack depend on `k`, not `V`, because the MTP
  head covers the verified columns plus its `k-1` autoregressive steps.
- `SpeculativeStats::accepted_per_position` has `V` positions; add drafted/accepted counters per
  source (MTP, pool) so the product can report where accepted tokens came from.

### Host policy, per lane and shared pool

- `ProgramImpl` owns one `NgramDraftPool` (default 16 MiB) shared by all lanes of the Program.
  Only the Engine worker mutates a Program, so the pool needs no synchronization. Programs share no
  mutable state, so separate model instances would have separate pools.
- Per-lane state is one cursor, `SequenceState::ngram_observed`: the ledger size already reported
  to the pool. It resets to 0 where the lane resets (`storage/context.cpp`, session restore).
  Before building ingress, `decode_mtp_batch` catches up with
  `pool.observe(sequence.ledger, sequence.ngram_observed)`. One catch-up point covers every ledger
  append path: prompt, forced tokens (`transactions/commit.cpp`), speculative commits and session
  restore. The first call of a request hashes the whole prompt (about 1 ms for 100k tokens on the
  host). If that shows up in TTFT, move the prompt observation between prefill submit and
  synchronize, where the host is idle.
- Draft choice per row (a pure host function with its own unit test), with `M` =
  `sequence.mtp_drafts[0..mtp_draft_count)`:
  - `chain`: `M`, then `pool.propose(ledger + M)` for up to `V - |M|` more tokens;
  - `select`: `pool.propose(ledger)` when it is longer than `M` and at least `--ngram-min`,
    otherwise `M`.
  An extension shorter than `--ngram-min` is dropped. The extent is bounded by the budget and
  context rules the round already applies (`max_by_budget`, `capacity - frontier - 1`).
  `NgramDraftPool::propose` takes one contiguous context, so `chain` proposes from a small
  per-Program scratch holding the last `n` ledger tokens followed by `M`. The same holds for any
  prefix: only the last `n` tokens enter the hash.
- Proposing runs after the previous round's synchronize and commit, where the host already
  builds ingress, so it adds no synchronization. Cost is `O(n + V)` per row.

### CUDA Graphs and ingress

- Ingress needs no new transfer: `current_drafts` and `current_extents` already travel in the
  single fixed-size `MtpDecodeIngress` copy at the top of the round body.
- The graph width is fixed per captured executable. A round's width is the largest row extent
  plus one, rounded up to a captured width. Capture two widths, `k+1` for MTP-only rounds and
  `V+1` for rounds that carry a pool draft, so ordinary rounds keep today's cost. This doubles the
  MTP graph set (`mtp_graph_profiles` in `planning/graph_profiles.cpp`, keyed by batch size and
  frontier range) and the graph reservation (86 MiB for one-lane MTP 3 on the RTX 4090). The
  simulator's width buckets measure whether more widths pay for their memory.
- Causal-attention envelopes (`mtp_causal_attention_envelopes`) and the workspace plan
  (`workspace_plan.mtp_round`) are computed for `V`.

### Product, CLI and serving

- `SpeculativeOptions` (`include/ninfer/types.h`) gains
  `NgramOptions {mode: off|chain|select, max_drafts V, match_tokens n, min_drafts, pool_bytes}`.
- CLI and `ninfer-serve` flags, validated in `product::validate_speculative_cli_options`
  (`src/product/speculative_options.h`), parsed in `apps/cli/options.cpp` and
  `src/serve/serve_options.cpp`:
  `--ngram chain|select --ngram-max V --ngram-n N --ngram-min N --ngram-pool-mib M`.
  Phase 1 requires `--spec mtp`, `V` in `[draft_tokens, 15]`.
- Request log `server_start` (`src/serve/request_log.cpp`, schema version bump) records the
  options; request records and `/metrics` add pool drafted/accepted totals. Update `docs/cli.md`,
  `docs/serving.md` and the serve-options tests together.

### Verification

- Host: policy unit test; the existing pool test.
- Ops: `mtp_prepare_next_round` oracle test at the new `(V,k)` domain.
- Engine (real artifact, GPU box): greedy text with `--ngram chain` equals the text without it
  (lossless check); stochastic runs keep their accept statistics; accepted-per-source counters add
  up to the round totals.
- Performance: MTP 3 single-lane `ms per round` for MTP-only rounds is unchanged; end-to-end
  tok/s on recorded agent sessions against `--spec mtp --draft-tokens 3`.

Effort: 4-6 days.

## Phase 1b (optional): `--spec ngram` without MTP

A backend whose round is steps 1 and 2 only, for artifacts without MTP weights. It reuses the
Phase 1 ingress and host policy with `M` empty. Effort: about 2 days after Phase 1.

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
2. Phase 1 goes ahead if `chain`/`select` at cap 15 beat `mtp` clearly.
3. Phase 2 goes ahead if cap 32/64 beat cap 15 for the same policy clearly, with a substantial
   `>15 tok` share.
4. Rerun the simulator with measured Phase 1 round costs (`--cost-points`) before starting
   Phase 2.
