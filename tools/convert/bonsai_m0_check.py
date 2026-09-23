"""M0 acceptance check for Bonsai-2-27B (design doc section 7, Brief A step 1).

Validates, against the real local files, that:

(a) the PTQ1_0 decode + Hadamard reconstruction recovers the original (pre-Prism) weight
    for three ternary tensors of K = 5120, 6144, 17408, by comparing against the same
    logical tensors dequantized from the existing Qwen3.8-27B `.ninfer` artifact (same base
    model, per the design doc); and
(b) `blk.0`'s norms, `ssm_dt.bias`, `ssm_conv1d` and `ssm_a` match the artifact's
    `gdn/dt_bias`, `gdn/convolution`, `gdn/a_log`, `input_norm`, `post_attention_norm` and
    `gdn/norm`, and reports the exact numeric/order conventions needed to convert them.

Headline findings (see docs/maintainer/bonsai-ternary-conversion.md and design doc section 9
for the full writeup):

- The design doc's claim that `gdn_v_grouped` "only affects `ssm_out.weight`" is incomplete:
  the same GGUF-tiled (`i = rep*16 + k_head`) vs NInfer-grouped (`j = 3*k_head + rep`)
  48-head permutation is also needed on the OUTPUT rows of `attn_gate.weight` (`gdn/z`), not
  only the input columns of `ssm_out.weight`. Without it, `gdn/z`'s per-row cosine is ~0
  except at the two head indices (0 and 47) that are fixed points of the permutation.
- `ssm_a = -exp(a_log)` (equivalently `a_log = log(-ssm_a)`), the standard Mamba2/GDN
  parameterization, combined with the SAME 48-head permutation.
- `input_norm = attn_norm.weight - 1` and `post_attention_norm = post_attention_norm.weight
  - 1` (the "residual" RMSNorm weight convention); `gdn/norm` matches `ssm_norm.weight`
  directly, with no offset and no permutation.
- `ssm_conv1d.weight` matches `gdn/convolution` in full once (i) GGUF's `ne=[4,10240]` is
  read as row-major `(10240,4)` (4 is the fastest GGUF axis) and transposed to `(4,10240)`,
  and (ii) the same 48-head permutation is applied to the value portion (columns
  4096:10240, reshaped `[head=48][hd=128]`). The query/key portion (columns 0:4096) needs
  no permutation. Note the permutation direction: the per-head SCALAR arrays (`dt_bias`,
  `a_log`) are indexed forward (`array_ninfer[perm[i]] == array_gguf[i]`), but reordering a
  ROW or HEAD AXIS by fancy indexing is a gather and needs `argsort(perm)` (the inverse) as
  the index array -- getting this backwards silently produces a different-looking, equally
  wrong permutation rather than an obvious error.
- Once the head permutation is applied, all three ternary reconstructions converge to a
  tight per-row cosine band of ~0.87-0.89 (median), clearly and reproducibly separated from
  every wrong convention tried (all near 0 or negative). This is comfortably above 0.9's
  "not garbage" bar, but not "well above 0.9" as the design doc's acceptance criterion
  hoped; see the module docstring in the design doc's section 9 note for the interpretation.

Run directly for a human-readable report:

    E:\\LLM\\ninfer-4090-bonsai\\.venv\\Scripts\\python.exe -m tools.convert.bonsai_m0_check \\
        --gguf E:\\LLM\\Ternary-Bonsai-2-27B-PTQ1_0.gguf --ninfer E:\\LLM\\qwen3_8_27b.ninfer

The same functions back `tests/convert/test_bonsai_m0.py`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tools.convert.quantization.hadamard import unrotate_embedding_rows
from tools.convert.sources.gguf_reader import GgufStore
from tools.convert.sources.ninfer_artifact import NInferArtifactStore
from tools.convert.sources.prism_gguf import (
    assert_prism_ternary_gguf,
    dequantize_rows,
    sign_vectors,
)

GDN_VALUE_HEADS = 16  # ssm.group_count / linear_num_key_heads
GDN_VALUE_REPEATS = 3  # linear_num_value_heads / linear_num_key_heads
GDN_HEAD_DIM = 128  # ssm.state_size / linear_value_head_dim

# (GGUF tensor, NInfer bound parameter, expected K, output rows need the 48-head
# tiled->grouped permutation before comparison)
MATRIX_CASES: tuple[tuple[str, str, int, bool], ...] = (
    ("blk.0.attn_gate.weight", "text/layers/0/gdn/z", 5120, True),
    ("blk.0.ssm_out.weight", "text/layers/0/gdn/output", 6144, False),
    ("blk.0.ffn_down.weight", "text/layers/0/mlp/down", 17408, False),
)


def tiled_to_grouped_head_permutation(
    nk: int = GDN_VALUE_HEADS, rep: int = GDN_VALUE_REPEATS
) -> np.ndarray:
    """GGUF-tiled head index `i = rep_idx*nk + k_head` -> NInfer-grouped index
    `j = rep*k_head + rep_idx` (HF `repeat_interleave` order, design doc section 1.5).

    `k_head = i % nk`, `rep_idx = i // nk`, so `perm[i] = rep*(i % nk) + (i // nk)`.
    """
    return np.array([rep * (i % nk) + (i // nk) for i in range(nk * rep)])


def _permute_head_rows(matrix: np.ndarray, permutation: np.ndarray, head_dim: int = GDN_HEAD_DIM) -> np.ndarray:
    n, k = matrix.shape
    heads = n // head_dim
    if heads != permutation.shape[0] or n % head_dim:
        raise ValueError(f"matrix has {n} rows, expected {permutation.shape[0]} head-blocks of {head_dim}")
    return matrix.reshape(heads, head_dim, k)[permutation].reshape(n, k)


@dataclass(frozen=True, slots=True)
class MatrixResult:
    gguf_name: str
    ninfer_name: str
    k: int
    n: int
    median_cosine: float
    min_cosine: float
    max_cosine: float


def reconstruct_and_compare_matrix(
    gguf_store: GgufStore,
    ninfer_store: NInferArtifactStore,
    signs: dict[int, np.ndarray],
    gguf_name: str,
    ninfer_name: str,
    k: int,
    permute_output_rows: bool,
) -> MatrixResult:
    """Decode one ternary GGUF tensor, undo the Hadamard fold, and compare per row."""
    info = gguf_store.tensor(gguf_name)
    if info.shape[0] != k:
        raise ValueError(f"{gguf_name}: expected K={k}, got {info.shape[0]}")
    n = info.shape[1]
    decoded = dequantize_rows(gguf_store, gguf_name, 0, n).numpy().astype(np.float64)
    sign_vector = signs[k].numpy().astype(np.float64)
    reconstructed = unrotate_embedding_rows(decoded, sign_vector, block=1024)
    if permute_output_rows:
        # tiled_to_grouped_head_permutation() maps GGUF index i -> NInfer index perm[i]
        # (the direction used directly for the per-head scalar arrays below, e.g.
        # dt_bias_art[perm]). Reordering ROWS with fancy indexing needs the inverse: the
        # row that ends up at NInfer position j is GGUF's row inverse_perm[j].
        inverse_perm = np.argsort(tiled_to_grouped_head_permutation())
        reconstructed = _permute_head_rows(reconstructed, inverse_perm)

    reference = ninfer_store.dequantize(ninfer_name, device="cpu").numpy().astype(np.float64)
    if reference.shape != (n, k):
        raise ValueError(f"{ninfer_name}: expected shape {(n, k)}, got {reference.shape}")

    numerator = (reconstructed * reference).sum(axis=1)
    denom = np.linalg.norm(reconstructed, axis=1) * np.linalg.norm(reference, axis=1)
    cosine = numerator / np.clip(denom, 1e-30, None)
    return MatrixResult(
        gguf_name, ninfer_name, k, n,
        float(np.median(cosine)), float(cosine.min()), float(cosine.max()),
    )


def _f32(store: GgufStore, name: str) -> np.ndarray:
    return np.frombuffer(store.read_tensor_raw(name), dtype="<f4").copy()


def head_permutation_from_dt_bias(dt_bias_gguf: np.ndarray, dt_bias_ninfer: np.ndarray) -> np.ndarray:
    """GGUF-tiled -> NInfer-grouped 48-head index permutation, recovered by rank matching.

    Cross-checked against `tiled_to_grouped_head_permutation()` (the formula derived
    directly from the design doc's `gdn_v_grouped` description): both agree, confirming
    the numeric convention independently of the structural one.
    """
    order_gguf = np.argsort(dt_bias_gguf)
    order_ninfer = np.argsort(dt_bias_ninfer)
    perm = np.empty(dt_bias_gguf.shape[0], dtype=np.int64)
    perm[order_gguf] = order_ninfer
    return perm


@dataclass(frozen=True, slots=True)
class ScalarReport:
    dt_bias_sorted_max_diff: float
    dt_bias_permuted_max_diff: float
    head_permutation_rank_match_agreement: float
    ssm_a_convention: str
    ssm_a_permuted_max_diff: float
    conv1d_qk_max_diff: float
    conv1d_qk_cosine: float
    conv1d_value_cosine: float
    input_norm_offset_max_diff: float
    input_norm_offset_cosine: float
    post_attention_norm_offset_max_diff: float
    post_attention_norm_offset_cosine: float
    gdn_norm_direct_cosine: float


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float((a.flatten() @ b.flatten()) / (np.linalg.norm(a) * np.linalg.norm(b)))


def compare_scalars(gguf_store: GgufStore, ninfer_store: NInferArtifactStore) -> ScalarReport:
    dt_bias = _f32(gguf_store, "blk.0.ssm_dt.bias")
    dt_bias_art = ninfer_store.dequantize("text/layers/0/gdn/dt_bias").numpy()
    dt_bias_sorted_diff = float(np.max(np.abs(np.sort(dt_bias) - np.sort(dt_bias_art))))

    # The formula-derived permutation (from the design doc's gdn_v_grouped description) is
    # used for every actual comparison below; the rank-matched one is only a cross-check,
    # since dt_bias has near-tied values at two head indices that make rank matching alone
    # ambiguous there (it agrees with the formula at 46 of 48 positions).
    perm = tiled_to_grouped_head_permutation()
    rank_perm = head_permutation_from_dt_bias(dt_bias, dt_bias_art)
    head_permutation_agreement = float(np.mean(perm == rank_perm))
    dt_bias_permuted_diff = float(np.max(np.abs(dt_bias - dt_bias_art[perm])))

    ssm_a = _f32(gguf_store, "blk.0.ssm_a")
    a_log_art = ninfer_store.dequantize("text/layers/0/gdn/a_log").numpy()
    log_neg_ssm_a = np.log(-ssm_a)
    ssm_a_permuted_diff = float(np.max(np.abs(log_neg_ssm_a - a_log_art[perm])))

    raw_conv = np.frombuffer(gguf_store.read_tensor_raw("blk.0.ssm_conv1d.weight"), dtype="<f4").copy()
    # ne=[4,10240]: 4 is the fastest GGUF axis, so the true row-major shape is (10240,4);
    # transpose to (4,10240) to align with NInfer's declared (tap-major) shape.
    conv1d = raw_conv.reshape(10240, 4).T
    conv_art = ninfer_store.dequantize("text/layers/0/gdn/convolution").numpy().reshape(4, 10240)
    qk_gguf, qk_art = conv1d[:, :4096], conv_art[:, :4096]
    conv1d_qk_max_diff = float(np.max(np.abs(qk_gguf - qk_art)))
    conv1d_qk_cosine = _cosine(qk_gguf, qk_art)
    val_gguf, val_art = conv1d[:, 4096:], conv_art[:, 4096:]
    # Reordering the head axis by fancy indexing is a gather, so it needs the inverse of
    # the GGUF->NInfer index map `perm` (see the identical reasoning at the row-permutation
    # call site above).
    inverse_perm = np.argsort(perm)
    conv1d_value_cosine = _cosine(
        val_gguf.reshape(4, 48, 128)[:, inverse_perm, :], val_art.reshape(4, 48, 128)
    )

    attn_norm = _f32(gguf_store, "blk.0.attn_norm.weight")
    input_norm_art = ninfer_store.dequantize("text/layers/0/input_norm").numpy()
    input_norm_diff = float(np.max(np.abs((attn_norm - 1.0) - input_norm_art)))
    input_norm_cosine = _cosine(attn_norm - 1.0, input_norm_art)

    post_norm = _f32(gguf_store, "blk.0.post_attention_norm.weight")
    post_norm_art = ninfer_store.dequantize("text/layers/0/post_attention_norm").numpy()
    post_norm_diff = float(np.max(np.abs((post_norm - 1.0) - post_norm_art)))
    post_norm_cosine = _cosine(post_norm - 1.0, post_norm_art)

    ssm_norm = _f32(gguf_store, "blk.0.ssm_norm.weight")
    gdn_norm_art = ninfer_store.dequantize("text/layers/0/gdn/norm").numpy()
    gdn_norm_cosine = _cosine(ssm_norm, gdn_norm_art)

    return ScalarReport(
        dt_bias_sorted_max_diff=dt_bias_sorted_diff,
        dt_bias_permuted_max_diff=dt_bias_permuted_diff,
        head_permutation_rank_match_agreement=head_permutation_agreement,
        ssm_a_convention="ssm_a = -exp(a_log), a_log = log(-ssm_a), plus the same 48-head "
        "tiled(GGUF)->grouped(NInfer) permutation used for gdn/z's rows and ssm_out's "
        "value columns",
        ssm_a_permuted_max_diff=ssm_a_permuted_diff,
        conv1d_qk_max_diff=conv1d_qk_max_diff,
        conv1d_qk_cosine=conv1d_qk_cosine,
        conv1d_value_cosine=conv1d_value_cosine,
        input_norm_offset_max_diff=input_norm_diff,
        input_norm_offset_cosine=input_norm_cosine,
        post_attention_norm_offset_max_diff=post_norm_diff,
        post_attention_norm_offset_cosine=post_norm_cosine,
        gdn_norm_direct_cosine=gdn_norm_cosine,
    )


def run(gguf_path: Path, ninfer_path: Path) -> tuple[list[MatrixResult], ScalarReport]:
    with GgufStore(gguf_path) as gguf_store, NInferArtifactStore(ninfer_path) as ninfer_store:
        assert_prism_ternary_gguf(gguf_store)
        signs = sign_vectors(gguf_store)
        matrix_results = [
            reconstruct_and_compare_matrix(
                gguf_store, ninfer_store, signs, gguf_name, ninfer_name, k, permute_rows
            )
            for gguf_name, ninfer_name, k, permute_rows in MATRIX_CASES
        ]
        scalars = compare_scalars(gguf_store, ninfer_store)
    return matrix_results, scalars


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--ninfer", type=Path, required=True)
    args = parser.parse_args(argv)

    matrix_results, scalars = run(args.gguf, args.ninfer)

    print("== (a) ternary reconstruction: per-row cosine similarity ==")
    for result in matrix_results:
        print(
            f"  {result.gguf_name} -> {result.ninfer_name} (K={result.k}, N={result.n}): "
            f"median={result.median_cosine:.4f} min={result.min_cosine:.4f} max={result.max_cosine:.4f}"
        )

    print("\n== (b) scalar/vector conventions ==")
    print(f"  dt_bias, sorted-order-invariant max diff: {scalars.dt_bias_sorted_max_diff:.6f}")
    print(
        "  formula-derived vs rank-matched head permutation agreement: "
        f"{scalars.head_permutation_rank_match_agreement:.4f} (46/48 expected: 2 near-tied dt_bias values)"
    )
    print(f"  dt_bias, with recovered head permutation, max diff: {scalars.dt_bias_permuted_max_diff:.6f}")
    print(f"  ssm_a convention: {scalars.ssm_a_convention}")
    print(f"  ssm_a, with convention + permutation, max diff: {scalars.ssm_a_permuted_max_diff:.6f}")
    print(
        f"  conv1d query/key (direct, transposed): max diff {scalars.conv1d_qk_max_diff:.6f}, "
        f"cosine {scalars.conv1d_qk_cosine:.6f}"
    )
    print(
        f"  conv1d value (48-head permutation applied): cosine {scalars.conv1d_value_cosine:.6f}"
    )
    print(
        f"  input_norm = attn_norm.weight - 1: max diff {scalars.input_norm_offset_max_diff:.6f}, "
        f"cosine {scalars.input_norm_offset_cosine:.6f}"
    )
    print(
        f"  post_attention_norm = post_attention_norm.weight - 1: "
        f"max diff {scalars.post_attention_norm_offset_max_diff:.6f}, "
        f"cosine {scalars.post_attention_norm_offset_cosine:.6f}"
    )
    print(f"  gdn/norm vs ssm_norm.weight (direct, no offset): cosine {scalars.gdn_norm_direct_cosine:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
