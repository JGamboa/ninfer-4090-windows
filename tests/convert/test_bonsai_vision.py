"""`bonsai2_27b` Vision from a synthetic Prism Qwen3-VL mmproj GGUF.

The mmproj follows llama.cpp's `Qwen3VLVisionModel` layout (PrismML-Eng/llama.cpp,
`conversion/qwen3vl.py`): Q8_0 matrices, an F16 `ffn_down`, F32 vectors, the Conv3D patch
embedding split into two temporal halves, `mm.0`/`mm.2`/`v.post_ln` for the merger. The
oracle is the float values the GGUF itself represents (Q8_0 decoded here from its blocks),
assembled into the HF layout with literal indexing; each converted parameter must match
within its stored format's quantization error.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from tools.convert.__main__ import main as convert_main
from tools.convert.bonsai_base import write_base
from tools.convert.bonsai_vision_check import compare
from tools.convert.sources.ninfer_artifact import NInferArtifactStore

from .bonsai_fixtures import H, write_gguf, write_reference
from .gguf_fixtures import build_gguf

VH, VI, VHEADS, P, T, MERGE, POS, DEPTH = 128, 256, 4, 8, 2, 2, 16, 2
M = MERGE * MERGE * VH


def _q8_0(values):
    """Q8_0 blocks of a row-major float matrix and the values they represent."""
    blocks = values.reshape(-1, 32).astype(np.float32)
    d = (np.abs(blocks).max(axis=1) / 127).astype(np.float16)
    safe = np.where(d == 0, 1, d.astype(np.float32))
    q = np.clip(np.rint(blocks / safe[:, None]), -127, 127).astype(np.int8)
    raw = b"".join(d[i].tobytes() + q[i].tobytes() for i in range(len(blocks)))
    return raw, (q.astype(np.float32) * d.astype(np.float32)[:, None]).reshape(values.shape)


def write_mmproj(path):
    rng = np.random.default_rng(5)
    tensors, truth = [], {}

    def matrix_q8(name, n, k):
        raw, values = _q8_0(rng.standard_normal((n, k)).astype(np.float32))
        tensors.append((name, (k, n), 8, raw))
        truth[name] = values

    def matrix_f16(name, n, k):
        values = rng.standard_normal((n, k)).astype(np.float16)
        tensors.append((name, (k, n), 1, values.tobytes()))
        truth[name] = values.astype(np.float32)

    def f32(name, torch_shape, values=None):
        values = rng.standard_normal(torch_shape).astype(np.float32) if values is None else values
        tensors.append((name, tuple(reversed(torch_shape)), 0, values.tobytes()))
        truth[name] = values

    for i in range(DEPTH):
        b = f"v.blk.{i}."
        matrix_q8(b + "attn_qkv.weight", 3 * VH, VH)
        f32(b + "attn_qkv.bias", (3 * VH,))
        matrix_q8(b + "attn_out.weight", VH, VH)
        f32(b + "attn_out.bias", (VH,))
        matrix_q8(b + "ffn_up.weight", VI, VH)
        f32(b + "ffn_up.bias", (VI,))
        matrix_f16(b + "ffn_down.weight", VH, VI)
        f32(b + "ffn_down.bias", (VH,))
        for norm in ("ln1", "ln2"):
            f32(b + norm + ".weight", (VH,))
            f32(b + norm + ".bias", (VH,))
    matrix_q8("mm.0.weight", M, M)
    f32("mm.0.bias", (M,))
    matrix_q8("mm.2.weight", H, M)
    f32("mm.2.bias", (H,))
    f32("v.post_ln.weight", (VH,))
    f32("v.post_ln.bias", (VH,))
    # Distinct signs per temporal half expose a swapped Conv3D reassembly.
    f32("v.patch_embd.weight", (VH, 3, P, P), rng.random((VH, 3, P, P)).astype(np.float32) + 0.5)
    f32("v.patch_embd.weight.1", (VH, 3, P, P), -rng.random((VH, 3, P, P)).astype(np.float32) - 0.5)
    f32("v.patch_embd.bias", (VH,))
    f32("v.position_embd.weight", (POS, VH))
    metadata = {
        "general.architecture": (8, "clip"),
        "general.type": (8, "mmproj"),
        "clip.has_vision_encoder": (7, True),
        "clip.projector_type": (8, "qwen3vl_merger"),
        "clip.use_gelu": (7, True),
        "clip.vision.projection_dim": (4, H),
        "clip.vision.image_size": (4, 32),
        "clip.vision.patch_size": (4, P),
        "clip.vision.embedding_length": (4, VH),
        "clip.vision.feed_forward_length": (4, VI),
        "clip.vision.block_count": (4, DEPTH),
        "clip.vision.attention.head_count": (4, VHEADS),
        "clip.vision.spatial_merge_size": (4, MERGE),
        "clip.vision.is_deepstack_layers": (9, (7, [False] * DEPTH)),
    }
    path.write_bytes(build_gguf(metadata, tensors))
    return truth


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bonsai-vision")
    write_gguf(tmp / "bonsai.gguf")
    reference = write_reference(tmp)
    truth = write_mmproj(tmp / "mmproj.gguf")
    resources = tmp / "vision-resources"
    resources.mkdir()
    for role in ("preprocessor_config.json", "video_preprocessor_config.json"):
        (resources / role).write_text(
            json.dumps({"patch_size": P, "temporal_patch_size": T, "merge_size": MERGE})
        )
    write_base(tmp / "bonsai.gguf", reference, tmp / "base", tmp / "mmproj.gguf", resources)
    out = tmp / "bonsai.ninfer"
    convert_main(
        [
            "--model", str(tmp / "base"),
            "--recipe", "bonsai2_27b",
            "--components", "text,vision",
            "--source", f"gguf={tmp / 'bonsai.gguf'}",
            "--source", f"mmproj={tmp / 'mmproj.gguf'}",
            "--device", "cpu",
            "--out", str(out),
        ]
    )
    return truth, out, tmp / "mmproj.gguf"


def _relative(got, expected):
    got, expected = np.asarray(got, np.float64), np.asarray(expected, np.float64)
    return np.linalg.norm(got - expected) / np.linalg.norm(expected)


def test_vision_config_follows_the_mmproj(converted):
    _, out, _ = converted
    with NInferArtifactStore(out) as store:
        config = store.directory.components["vision"]["config"]
    assert (config["depth"], config["hidden_size"], config["intermediate_size"]) == (DEPTH, VH, VI)
    assert (config["num_heads"], config["patch_size"], config["temporal_patch_size"]) == (
        VHEADS, P, T
    )
    assert (config["spatial_merge_size"], config["num_position_embeddings"]) == (MERGE, POS)


def test_vision_parameters_match_the_mmproj_values(converted):
    truth, out, _ = converted
    qkv = truth["v.blk.1.attn_qkv.weight"]
    qkv_bias = truth["v.blk.1.attn_qkv.bias"]
    patch = np.stack((truth["v.patch_embd.weight"], truth["v.patch_embd.weight.1"]), axis=2)
    expected = {
        # name: (values, format tolerance in relative L2)
        "vision/patch_embedding": (patch.reshape(VH, -1), 0.04),
        "vision/patch_embedding_bias": (truth["v.patch_embd.bias"], 0.01),
        "vision/position_embedding": (truth["v.position_embd.weight"], 0.01),
        "vision/layers/1/norm1_weight": (truth["v.blk.1.ln1.weight"], 0.01),
        "vision/layers/1/norm2_bias": (truth["v.blk.1.ln2.bias"], 0.01),
        "vision/layers/1/attention/query": (qkv[:VH], 0.15),
        "vision/layers/1/attention/key": (qkv[VH : 2 * VH], 0.15),
        "vision/layers/1/attention/value": (qkv[2 * VH :], 0.15),
        "vision/layers/1/attention/value_bias": (qkv_bias[2 * VH :], 0.01),
        "vision/layers/1/attention/output": (truth["v.blk.1.attn_out.weight"], 0.08),
        "vision/layers/1/mlp/fc1": (truth["v.blk.1.ffn_up.weight"], 0.15),
        "vision/layers/1/mlp/fc2": (truth["v.blk.1.ffn_down.weight"], 0.08),
        "vision/layers/1/mlp/fc2_bias": (truth["v.blk.1.ffn_down.bias"], 0.01),
        "vision/merger/norm_weight": (truth["v.post_ln.weight"], 0.01),
        "vision/merger/fc1": (truth["mm.0.weight"], 0.02),
        "vision/merger/fc2": (truth["mm.2.weight"], 0.02),
        "vision/merger/fc2_bias": (truth["mm.2.bias"], 0.01),
    }
    with NInferArtifactStore(out) as store:
        for name, (values, tolerance) in expected.items():
            got = store.dequantize(name).numpy().reshape(values.shape)
            assert _relative(got, values) < tolerance, name
        # A swapped query/key or temporal half would be off by far more than any format.
        swapped = store.dequantize("vision/layers/1/attention/query").numpy()
        assert _relative(swapped, qkv[VH : 2 * VH]) > 1.0
        time_one = store.dequantize("vision/patch_embedding").numpy().reshape(VH, 3, T, P, P)
        assert (time_one[:, :, 0] > 0).mean() > 0.95 and (time_one[:, :, 1] < 0).mean() > 0.95


def test_vision_check_recognizes_the_same_tower(converted):
    _, out, mmproj = converted
    results = compare(mmproj, out)
    assert len(results) >= 10 and max(difference for _, difference in results) < 0.2
