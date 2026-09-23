# Bonsai-2-27B conversion: M0 findings and M1 procedure (agent A)

Status: M0 complete (design doc section 7). M1 converter code complete and tested on
synthetic inputs (section "M1 conversion" below); the real conversion run, the mapping
check and the `inspect` output are still to be recorded on the Windows machine. This records the numeric/order conventions
resolved by comparing `E:\LLM\Ternary-Bonsai-2-27B-PTQ1_0.gguf` against
`E:\LLM\qwen3_8_27b.ninfer` (same base architecture, per
`docs/maintainer/bonsai-ternary-design.md` section 1.1). Reproduce with:

```
E:\LLM\ninfer-4090-bonsai\.venv\Scripts\python.exe -m tools.convert.bonsai_m0_check ^
  --gguf E:\LLM\Ternary-Bonsai-2-27B-PTQ1_0.gguf --ninfer E:\LLM\qwen3_8_27b.ninfer
```

and `python -m pytest tests/convert/test_bonsai_m0.py -q` (skips automatically if the two
local files are absent).

## (a) Ternary reconstruction

Reconstructed `W_orig[:, blk] ≈ (W'[:, blk] @ H) * s[blk]` per 1024-column block
(`tools/convert/quantization/hadamard.unrotate_embedding_rows`) for the three PTQ1_0
tensors named in the design doc, and compared per row (cosine similarity) against the
matching logical parameter dequantized from `qwen3_8_27b.ninfer`:

| GGUF tensor | NInfer parameter | K | N | median cosine | min | max |
|---|---|---|---|---|---|---|
| `blk.0.attn_gate.weight` | `text/layers/0/gdn/z` | 5120 | 6144 | 0.8834 | 0.5362 | 0.8939 |
| `blk.0.ssm_out.weight` | `text/layers/0/gdn/output` | 6144 | 5120 | 0.8850 | 0.8763 | 0.8938 |
| `blk.0.ffn_down.weight` | `text/layers/0/mlp/down` | 17408 | 5120 | 0.8780 | 0.8731 | 0.8894 |

The `(W'@H)*s` formula (signs applied AFTER the Hadamard butterfly) is confirmed correct,
not merely plausible: every alternative tried collapses to a median cosine within noise of
0 (or its negation), sharply separated from the ~0.88 achieved by the right one:

| variant | median cosine (`ssm_out`) |
|---|---|
| `(W' @ H) * s` (design doc formula) | **0.8850** |
| `H(s ⊙ W')` (signs before H instead of after) | 0.0006 |
| `(W' @ H) * (-s)` (sign vector negated) | -0.8850 |
| `W' @ H` (no sign at all) | -0.0112 |
| `W'` (no transform at all) | 0.0004 |

**Open point**: 0.88 is comfortably above "not garbage" but short of the design doc's
"well above 0.9" acceptance language (section 7's M0 row). Given how sharply every wrong
convention collapses to ~0 with nothing in between, this is very likely the genuine
reconstruction noise floor of 1.76-bit PTQ1_0 quantization for this weight distribution
rather than a remaining convention bug, but it has not been independently confirmed against
a second oracle (e.g. the original bf16 HF checkpoint, which was not available locally for
this run) and is flagged for the M1/M3 owner.

### `gdn/z` needs the same head permutation as `ssm_out`'s columns

The design doc says `gdn_v_grouped` "only affects `ssm_out.weight`" (section 1.5). That is
incomplete for static weight-storage order: without a 48-head permutation, `attn_gate.weight`
(-> `gdn/z`) reconstructs at median cosine ~0.004 (garbage), matching only at head indices 0
and 47 -- the two fixed points of the permutation below. Applying it brings `gdn/z` in line
with the other two tensors (0.8834 median, table above).

## (b) Scalar/vector parameter conventions

All of `blk.0`'s GDN scalar/vector parameters use the same 48-head index permutation
(GDN has `linear_num_key_heads = 16`, `linear_num_value_heads = 48`, so 3 value heads share
each key head):

```
GGUF tiled index   i = rep_idx * 16 + k_head      (rep_idx outer, k_head inner)
NInfer grouped idx  j = 3 * k_head + rep_idx        (k_head outer, rep_idx inner; HF
                                                      repeat_interleave order)
perm[i] = j = 3*(i % 16) + (i // 16)
```

`tools/convert/bonsai_m0_check.tiled_to_grouped_head_permutation()` implements this and is
cross-checked against an independent rank-matching of `ssm_dt.bias`'s 48 values (which
agrees at 46/48 positions; the other two are near-tied dt_bias values that rank-matching
alone cannot disambiguate, resolved instead by the closed-form permutation).

**Direction matters and is easy to get backwards.** `perm` above is the direction used
directly for per-head SCALAR arrays: `dt_bias_ninfer[perm[i]] == dt_bias_gguf[i]`. Reordering
a ROW axis or a reshaped HEAD axis by NumPy/PyTorch fancy indexing is a *gather*, and needs
the *inverse* permutation (`argsort(perm)`) as the index array; using `perm` directly there
silently produces a different-looking but equally wrong permutation (this was hit once
during development of the M0 check and is called out in
`tools/convert/bonsai_m0_check.py`'s docstring and inline comments).

| check | result |
|---|---|
| `dt_bias` (GGUF) vs `gdn/dt_bias` (NInfer), sorted (order-invariant) | max diff 0.0039 |
| `dt_bias`, with the closed-form permutation applied | max diff 0.0039 |
| `ssm_a` convention | `ssm_a = -exp(a_log)`, i.e. `a_log = log(-ssm_a)` (standard Mamba2/GDN parameterization) |
| `ssm_a`, with the convention + permutation | max diff 0.0313 |
| `ssm_conv1d.weight` query/key columns (0:4096), direct | cosine 0.9964, max diff 0.043 |
| `ssm_conv1d.weight` value columns (4096:10240), with the permutation | cosine 0.9917 |
| `input_norm` vs `attn_norm.weight - 1` | cosine 0.9865, max diff 0.071 |
| `post_attention_norm` vs `post_attention_norm.weight - 1` | cosine 0.99991, max diff 0.037 |
| `gdn/norm` vs `ssm_norm.weight`, direct (no offset, no permutation) | cosine 0.9996 |

Residuals of a few hundredths are consistent with reduced-precision (bf16-level) rounding
somewhere in either checkpoint's export pipeline, not a remaining conversion bug: e.g.
`dt_bias`'s max diff (0.0039) is close to fp16 ULP at that magnitude, and `ssm_a`'s (0.0313)
and the norms' (0.04-0.07) are close to bf16 ULP at theirs.

### Norm convention: "residual" weight for the two outer RMSNorms only

`input_norm` and `post_attention_norm` (the two per-layer RMSNorms NInfer applies around
GDN/attention and the MLP) are stored as `scale - 1` (mean ~0, "Gemma/Qwen-style residual"
parameterization), while `gdn/norm` (the GDN block's own internal gated-RMSNorm weight) is
stored directly as `scale` (mean ~0.87, matching GGUF's `ssm_norm.weight` with no offset).
The converter must apply the `-1` offset only to the two outer norms.

### `ssm_conv1d.weight`'s GGUF memory layout is transposed relative to its own `ne`

GGUF records `ssm_conv1d.weight`'s `ne` as `[4, 10240]`; since `ne[0]` is the fastest-varying
axis, the true row-major layout is `(10240, 4)` (10240 channels, each with 4 contiguous
kernel-tap values), the *opposite* of naively reading `ne` as a NumPy shape and reshaping
`(4, 10240)` directly (which silently produces a transposed, uncorrelated array with cosine
~0 against the reference). Read `(10240, 4)` and transpose to `(4, 10240)` to match NInfer's
declared (tap-major) shape.

## Summary for design doc section 9

- (A) `ssm_a` convention: `ssm_a = -exp(a_log)` (`a_log = log(-ssm_a)`), combined with the
  48-head tiled->grouped permutation (`perm[i] = 3*(i%16) + i//16`) also needed for
  `dt_bias`, `ssm_conv1d`'s value columns, and `gdn/z`'s output rows (not just `ssm_out`'s
  input columns as section 1.5 states).
- (A) Python environment: Microsoft Store Python 3.12.10, venv at
  `E:\LLM\ninfer-4090-bonsai\.venv`; `numpy`, `safetensors`, `pytest` from PyPI; `torch`
  2.11.0+cu128 from the cu128 index (GPU wheel installed successfully, no CPU fallback
  needed; `torch.cuda.is_available()` was not itself exercised by this run's checks, which
  ran on CPU).

## Open questions for agents B/C

1. Matrix-reconstruction cosine plateaus at ~0.88, not "well above 0.9" -- see "(a)" above.
   Needs a second oracle (original bf16 checkpoint) or a wider tensor sample to confirm this
   is quantization noise rather than a residual convention gap.
2. This M0 check only proves that GGUF's and the existing `.ninfer`'s STATIC weight/parameter
   storage orders now agree once tiled->grouped is applied. It does NOT prove NInfer's
   runtime GDN kernel actually PRODUCES its output activation (`gdn/output`'s K=6144 input,
   `on` in `execution/text.cpp`) in that same grouped order at inference time -- that is
   section 5.3's `perm == nullptr` question for agent B, and can only be settled by reading
   the kernel or by M3's token-level comparison, per design doc section 8 risk #2.

## M1 conversion (Brief A steps 2 and 3)

Code: `t2_g128_fp16` / `ternary_row_k128_v1` (`tools/artifact/formats.py`, `layouts.py`,
`codecs/ternary.py`, `tensor_output.py`), `import_encoded` for every encoded format
(`tools/convert/methods.py`), `PrismCheckpoint` (`tools/convert/sources/prism_checkpoint.py`,
the GGUF as an HF-named checkpoint), `NInferArtifactStore.parameter_source` (exact Q8 rows
for the MTP copy), recipe `bonsai2_27b` (`tools/convert/official_recipes.py`) and
`tools/convert/bonsai_base.py`.

What the recipe writes:

| parameters | format | source |
|---|---|---|
| layer attention q/k/gate/v, o; GDN q/k/v/z, out; MLP gate/up, down | `t2_g128_fp16`, one parent per fused group | rotated GGUF words, rows in NInfer order |
| `text/token_embedding`, `text/output_head` | `q8_g32_fp16` (`grouped_absmax`) | primal-basis values `(W' @ H) * s` |
| GDN `a_projection`, `b_projection` | bf16, separate parents | GGUF BF16, grouped head order |
| norms, `a_log`, `dt_bias`, `convolution`, q/k norms | as the Qwen3.8 recipe | GGUF F32 with the M0 conventions |
| `text/hadamard/signs_{5120,6144,17408}` | bf16 | `prism.hadamard.sign_values` |
| `mtp/*` | as stored in the reference (Q8 words copied exactly) | `qwen3_8_27b.ninfer` |
| `dflash2/*` | as the official recipes | DFlash2 HF companion |

Run on Windows (venv of design doc section 6.5):

```
.venv\Scripts\python.exe -m tools.convert.bonsai_base --gguf E:\LLM\Ternary-Bonsai-2-27B-PTQ1_0.gguf ^
  --reference E:\LLM\qwen3_8_27b.ninfer --out E:\LLM\bonsai2-27b
.venv\Scripts\python.exe -m tools.convert.bonsai_mapping_check --gguf E:\LLM\Ternary-Bonsai-2-27B-PTQ1_0.gguf ^
  --reference E:\LLM\qwen3_8_27b.ninfer --base E:\LLM\bonsai2-27b
.venv\Scripts\python.exe -m tools.convert --model E:\LLM\bonsai2-27b --recipe bonsai2_27b ^
  --components text,mtp,dflash2 --source gguf=E:\LLM\Ternary-Bonsai-2-27B-PTQ1_0.gguf ^
  --source mtp=E:\LLM\qwen3_8_27b.ninfer --source dflash2=E:\LLM\dflash2-src ^
  --proposal --name bonsai2-27b --out E:\LLM\bonsai2_27b_t2.ninfer --device cuda
.venv\Scripts\python.exe -m tools.artifact.inspect E:\LLM\bonsai2_27b_t2.ninfer
```

`bonsai_base` prints whether the reference chat template matches the GGUF's embedded one
and saves the latter as `gguf_chat_template.jinja` (design doc risk 5).
`bonsai_mapping_check` compares every logical parameter of layers 0 (GDN) and 3 (attention)
plus the globals against the reference and exits non-zero if one falls below its bar
(ternary median row cosine 0.80, direct 0.98). It covers the conventions M0 did not measure:
`in_proj_qkv` value rows, `in_proj_a`/`in_proj_b` rows, `q_norm`/`k_norm`/final norm offsets,
and the attention q/gate split. Run it before the full conversion; a NO row means that
tensor's mapping in `prism_checkpoint.py` is wrong.

Synthetic coverage (`tests/convert/test_bonsai_recipe.py`): a Bonsai-shaped PQ2_0 GGUF (4
layers, nontrivial 2x2 head permutation) and a reference artifact with a Q8 MTP head go
through `bonsai_base` and the CLI; the test rebuilds the expected t2 parents, primal Q8
head/embedding, vectors and MTP words independently.

Results of the real run: _pending (Windows machine)_.
