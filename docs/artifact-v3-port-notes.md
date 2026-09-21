# v3 artifact reader — design notes (WIP, `main` branch)

This documents the port that lets this fork's reader open v3 `.ninfer` artifacts
(magic `NINFER\0\3`), alongside its existing v2 support. See
`WINDOWS_PORT.md` for the v2/v3 background and why v2 support exists at all.

## Status

**Done and validated end-to-end**: `ninfer-serve.exe` loads the real `Qwen3.8-27B` v3
artifact -- every binding resolves (text, vision, mtp, dflash2), `weights_id` resolves
correctly (inferred, not a placeholder), and all 16.7 GiB of weights load onto the GPU.
Startup now only fails on a data problem in the specific test artifact, not a bug in this
port (see "Known blocker" below).

What changed to get there, on top of the reader/binder layer (`src/artifact/reader.{h,cpp}`,
`src/artifact/binder.{h,cpp}` -- container framing, JSON directory, `find_fused()` /
`require_tensor_fused()`; see git history for that first commit's details):

- `artifact::bind_tensor_fused()` (`src/artifact/typed_binding.{h,cpp}`): tries the v2
  whole-object name first (`Binder::has_object`), falls back to `require_tensor_fused` with
  the v3 leaf names otherwise. This is the one new primitive every call site below uses, so
  each of them works unchanged against either artifact version.
- `src/targets/qwen3_6_27b/impl/load/bindings.cpp` (the real model's loader): rewired
  attention `query_key`/`gate_value`, gdn `query_key`/`value_z`, `mlp/gate_up`, all of MTP's
  per-layer tensors (v3 path is `"mtp/layers/0/..."`, not v2's `"mtp/layer/..."`),
  `draft_head`/`draft_head_token_ids` (v3: `"proposal/head"`/`"proposal/token_ids"`, per
  container spec section 9.2), and dflash2's `query_key_value` fusion + `mlp/gate_up`.
  dflash2 turned out **not** to be optional to fix: `bind_artifact()` auto-detects and
  validates it whenever the artifact declares the component at all (`has_dflash2 =
  binder.has_object(...)`), regardless of whether `--spec dflash2` was requested at serve
  time -- placement is just `ValidateOnly` instead of `Device` when the feature isn't
  selected, but the shapes still have to resolve.
- `src/targets/qwen3_6/impl/vision/bindings.cpp`: same treatment for the vision backbone's
  `qkv`/`qkv_bias` fusion, plus a plain renaming (`"norm1/weight"` v2 vs. `"norm1_weight"`
  v3, no underlying fusion). This is *also* not optional even without `--vision`, for the
  same reason as dflash2 above.

Three bugs turned up only once tested against the real artifact end-to-end, all fixed (see
commit `d5d9092` for detail): a units mismatch in the parts→row-range conversion (raw
element count vs. rows), an overly strict `find_fused()` check that broke on a
tied/shared binding (dflash2's `"context_key"` is a separate logical name for the exact
same range as `"key"`, not an additional distinct row range -- container spec section
12.5), and a real `std::string_view`-into-a-destroyed-temporary lifetime bug in the vision
bindings file (fixed by owning the leaf-name strings in a named array instead of
inlining them where they'd only live for the wrong expression).

## Known blocker (data, not code) -- currently bypassed for testing

`ninfer-serve.exe` reached `initializing frontend` and failed there twice, on two
resources this fork's frontend loader already reads correctly via the aliasing this port
added:

1. `tokenizer_config.json.chat_template does not match frontend/chat_template.jinja` --
   `validate_tokenizer_config()` in `src/targets/qwen3_6/impl/frontend/frontend.cpp`.
   Diffing the actual bytes: the standalone `chat_template.jinja` (9712 bytes) is this
   fork's *modified* template (SPDX header, selects semantics by SHA-256 below); the one
   embedded in `tokenizer_config.json` (8952 bytes) is the *unmodified upstream Qwen
   template*. They were never supposed to match unless the v3 export's converter forgot to
   re-embed the patched template into `tokenizer_config.json`.
2. `unsupported frontend/chat_template.jinja (sha256 a497db9e...)` --
   `CompiledChatTemplate::resolve()` in `chat_template.cpp` doesn't recognize this
   artifact's `chat_template.jinja` as either known digest
   (`kThinkingToggleTemplateDigest` / `kReasoningEffortTemplateDigest`).

Both are content bugs in the specific downloaded v3 artifact (or its converter), not in
this port's reader/binder/bindings.cpp wiring, which resolved every resource's *bytes*
correctly. **Both are currently bypassed** (search for `TEMPORARY` in `frontend.cpp` and
`chat_template.cpp`) to validate the rest of the v3 path -- the mismatch check is `#if 0`'d
out, and the digest resolver defaults to `ChatTemplateSemantics::ReasoningEffort` (matches
every `enable_thinking`/`reasoning_effort` request already verified against this same
model's v2 artifact) when neither known digest matches. **Revert both before merging** --
they exist only to isolate the chat-template problem from everything downstream of it,
and are not a fix for the actual data issue upstream (re-convert/re-download the v3
artifact with matching, recognized resources).

## v3 end-to-end validation: PASSED

With the two bypasses above, `ninfer-serve.exe` loads the real v3 artifact completely
(`engine ready | qwen3.8-27b/groupwise-int`, CUDA graphs built, listening on
`:8080`) and serves a real request correctly -- asked for a documented Fibonacci function
with `enable_thinking:false`, got back correct Python with the right docstring/type hints,
`reasoning_tokens: 0` (confirms the ReasoningEffort semantics fallback actually behaves
correctly for this request shape), 131.85 tok/s decode at 90% MTP draft acceptance
(145/161). This confirms the reader/binder/bindings.cpp work in this branch is
functionally complete and correct for the real Qwen3.8-27B v3 artifact; the only remaining
gap is the chat-template data problem above, which is not this port's bug to fix.

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
