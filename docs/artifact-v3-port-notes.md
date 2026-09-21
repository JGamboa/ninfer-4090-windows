# v3 artifact reader — design notes (WIP, `main` branch)

This documents the port that lets this fork's reader open v3 `.ninfer` artifacts
(magic `NINFER\0\3`), alongside its existing v2 support. See
`WINDOWS_PORT.md` for the v2/v3 background and why v2 support exists at all.

## Status

**Done**: `src/artifact/reader.{h,cpp}` and `src/artifact/binder.{h,cpp}` parse v3's
container framing and JSON directory (`components`/`objects`/`bindings`/`uses`/`files`) and
expose it through the *same* flat name-keyed API v2 already had (`Reader::find`,
`Binder::require_tensor`), plus one new primitive (`find_fused` /
`Binder::require_tensor_fused`) for the cases v2 didn't need. Validated against the real
`Qwen3.8-27B` v3 artifact with `apps/artifact_inspect/main.cpp`
(`ninfer-artifact-inspect.exe <file.ninfer>`), which resolves the actual fusion groups the
model needs and prints their shapes.

**Not done**: `src/targets/qwen3_6_27b/impl/load/bindings.cpp` still only asks for v2-style
names (e.g. `"gdn/query_key"` as one name) — it needs to switch its ~7 fused call sites to
`require_tensor_fused` with the v3 leaf names (see below). `Reader::identity()` for v3 is a
placeholder (`model_id="v3-artifact"`, `weights_id="unresolved"`); matching it to a
`WeightsProfile` (`registry.cpp` / `Target::resolve_weights`) needs its own v3-aware pass.
No engine/GPU integration test has been run yet.

## Why v3 isn't just a header/magic bump

v3 replaces v2's flat `{name, format, layout, offset, bytes}` object list with a real
indirection layer: physical `objects` (opaque ids like `weight/000042`) are mapped to
logical parameter names through `bindings` (whole-object, or a byte/element range within
one shared object), and `uses` records per-use-site activation-quantization policy. See
the upstream spec at `Neroued/ninfer`'s `docs/maintainer/artifact-container.md` (in
Chinese) for the full format; this fork only implements the subset that the real
Qwen3.8-27B artifact actually uses (verified by extracting and inspecting its JSON
directory directly — see below).

## What the real artifact actually needs (verified, not assumed)

Pulled the JSON directory out of the real `qwen3_8_27b.ninfer` (v3) and its `qwen3_8_27b_v2.ninfer`
counterpart and diffed them directly, rather than implementing the full abstract spec:

- **Single file.** 20.4 GB payload, well under the 32 GB default shard limit → the `files`
  array always has exactly one entry with `path: null`. Multi-file continuation
  (`NINPRT` part headers) is **not implemented** — `parse_v3_files()` throws a clear error
  if it ever sees more than one file, instead of silently mishandling it.
- **Formats/layouts already existed.** Every format (`q4/q5/q6/q8_g*_fp16`, `bf16`, `fp32`,
  `int32`) and layout (`row_split_k128_v1`, `contiguous_le_v1`) the real artifact uses maps
  directly onto this fork's existing `NumericFormat`/`StorageLayout` enums and geometry
  code (`tensor_encoded_size`, `RowSplitGeometry`, ...) — v3 just spells the names
  differently (snake_case vs. this fork's kebab-case/enum-style names for v2).
  `NVFP4`/`FP8_E4M3FN_ROW_BF16S`/`RowScaleV1` are wired into the format/layout parsers for
  completeness but untested against real v3 data (the real artifact doesn't use them).
- **Bindings are simple.** Of 1513 bindings, 926 are whole-object (trivial rename) and 587
  are single-part fragments (never more than 1 part per binding — no need to support
  concatenating multiple parts for one logical parameter). Every fragment into a quantized
  (`row_split_k128_v1`) object is exactly row-aligned; the only misaligned-by-naive-check
  fragments are `contiguous_le_v1` bias vectors (vision component, unused by this fork's
  serving path), which don't need row alignment at all since they have no per-row plane
  structure — see the `k = layout == ContiguousLeV1 ? 1 : shape.back()` special case in
  `parse_v3()`.
- **`uses` is uniform.** Every one of the 844 `uses` entries has `activation_policy:
  "A16Only"` — no mixed-precision activation plumbing is needed for this artifact, so
  `uses` is parsed for structural validation but not otherwise consumed.
- **Fusion groups map onto existing "Split" bindings.cpp code paths.** Grouping the 587
  fragments by target object and by name pattern gives exactly (for the `text` and `mtp`
  components this fork actually serves):

  | Pattern | Count | Matches existing bindings.cpp call |
  |---|---:|---|
  | `attention/query + attention/key` | 16 (full-attention layers) | `SplitAttentionProjectionPlan.query_key` |
  | `attention/gate + attention/value` | 16 | `SplitAttentionProjectionPlan.gate_value` |
  | `gdn/query + gdn/key` | 48 (linear-attention layers) | `SplitGdnInputProjectionPlan.query_key` |
  | `gdn/value + gdn/z` | 48 | `SplitGdnInputProjectionPlan.value_z` |
  | `mlp/gate + mlp/up` | 64 (+ 1 mtp layer) | `load_mlp`'s `plan.gate_up` |
  | `attention/query+key+gate+value` | 1 (the single mtp layer) | `bind_mtp("mtp/layer/attention/query_key_gate_value", ...)` |

  `vision` (27 layers, `query_bias+key_bias+value_bias` and `query+key+value`) and
  `dflash2` (5 layers, 5-way fusion incl. `context_key`/`context_value`) have their own
  patterns but are out of scope: this fork's serving config never enables `--vision` or
  `--spec dflash2`, so their bindings are simply never looked up.
  `gdn/a_projection`/`gdn/b_projection` are whole-object bindings in v3 (not fused at all),
  so they need no special handling.

## Why the fix is "reconstruct the whole fused object", not "expose row-range sub-tensors"

`row_split_k128_v1` is **planar**: all rows' low-bits plane, then all rows' high-bits
plane, then all rows' scale plane — not row-interleaved. A row range therefore isn't one
contiguous byte range within the object; naively synthesizing a "sub-tensor" descriptor
with an adjusted `.offset`/`.shape` would require either multi-plane descriptors (not
supported by today's simple `{offset, bytes}` `TensorDescriptor`) or a physical
repack-into-owned-buffer step for every fragment.

Neither is necessary: `src/targets/qwen3_6_27b/impl/load/bindings.cpp` never actually
wants a row-range sub-tensor. Its "Split" binding plans (`SplitAttentionProjectionPlan`,
`SplitGdnInputProjectionPlan`, ...) already bind the **whole** fused tensor as one object
(e.g. `"attention/query_key"` at `{7168, 5120}` in v2) and only slice rows out of it
*after* materialization, via the already-existing `row_view()` helper in bindings.cpp
(operates on device pointers with the same per-plane row-stride math, post-upload). So the
v3 adapter's job is exactly: given the leaf binding names bindings.cpp still asks for
(`"attention/query"` + `"attention/key"`), verify they jointly and exactly tile one shared
physical object, and hand back *that whole object* — reusing every existing kernel/loading
code path unchanged. This is what `Reader::find_fused()` /
`Binder::require_tensor_fused()` do.

## Reader/Binder API added

- `Reader::find_fused(std::span<const std::string_view> names) const` — for v2, or a v3
  artifact where `names.size() == 1` and it's a whole-object binding, behaves like
  `find(names[0])`. Otherwise resolves the physical object all the given names bind into,
  throwing unless they *exactly* cover it (no gaps, no overlap, no missing/extra name) —
  see `Reader::Impl::find_fused()` in `reader.cpp`. Newly-synthesized fused entries are
  appended to the same `entries` vector v2 uses and cached by physical-object index, so
  repeated lookups (there won't be any in practice — each physical object has exactly one
  binder call site) don't re-validate or re-allocate.
- `Binder::require_tensor_fused(names, format, layout, shape)` — same
  contract/error-handling as `require_tensor`, but resolves through `find_fused`. Note:
  `Binder`'s `consumed_`/`planned_` bookkeeping vectors are sized off
  `reader.objects().size()` *at construction time*; since `find_fused` can append new
  entries afterward, `require_tensor_fused` grows those vectors lazily the first time an
  index falls outside them (see the comment at its start in `binder.cpp`) — a real bug that
  showed up in review before it could crash Binder::finish()'s bounds-free iteration.

## Remaining work (next session)

1. In `bind_groupwise_text_layers()` (the only weights profile with a real v3 artifact —
   `Qwen38GroupwiseInt`/`Qwen36GroupwiseInt`), change the ~7 fused `bind_weight(binder,
   prefix + "attention/query_key", ...)`-style calls to pass both the v2 whole name and the
   v3 leaf name list, e.g. something like `bind_weight_either(binder, prefix +
   "attention/query_key", {prefix + "attention/query", prefix + "attention/key"}, ...)`
   trying `require_tensor` first (v2) and falling back to `require_tensor_fused` (v3). The
   MTP call site uses `"mtp/layer/..."` (singular, no layer index) in v2 but the real v3
   artifact uses `"mtp/layers/0/..."` (plural, indexed) — get the leaf name path prefix
   right, it's not a straight substring substitution.
2. Give v3 artifacts a real `identity()` (or bypass `Target::resolve_weights` entirely for
   v3 and select the profile from `components.text.config` + the object format set
   directly) so `registry.cpp` can pick the right `WeightsProfile` instead of hitting the
   `"unresolved"` placeholder.
3. Build the full engine (not just `ninfer_artifact`/`ninfer-artifact-inspect`) and run
   `ninfer-serve.exe` against the real v3 file end-to-end, the same way `WINDOWS_PORT.md`'s
   v2 numbers were validated.

## How to re-verify the reader in isolation

No GPU or full engine build needed — `ninfer_artifact` is a standalone static library:

```bat
cmake --build build --target ninfer-artifact-inspect -j
build\apps\ninfer-artifact-inspect.exe path\to\v2.ninfer path\to\v3.ninfer
```

It prints `[OK]`/`[FAIL]` for a hard-coded set of names representative of what
`bind_groupwise_text_layers()` needs (see `apps/artifact_inspect/main.cpp`) — extend that
list as more of `bindings.cpp` gets ported.
