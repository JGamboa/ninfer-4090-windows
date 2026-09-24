"""End-to-end `bonsai2_27b` conversion of a synthetic Prism GGUF plus a reference artifact.

The oracle is independent of the converter: the fixture keeps its stored ternary codes,
and the expected artifact contents are rebuilt here from the design-doc conventions
(sections 1.5, 2.1 and `bonsai-ternary-conversion.md`) with literal loops and a
Kronecker-built Hadamard matrix.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.artifact.codecs.ternary import decode_ternary_words, unpack_ternary_codes
from tools.artifact.reader import Artifact
from tools.convert.official_recipes import TERNARY_FORMAT
from tools.convert.sources.ninfer_artifact import NInferArtifactStore

from .bonsai_fixtures import (
    CONV, DV, H, HEAD_DIM, HEADS, INTER, KG, NK, NV, REP, VG, convert_bonsai,
)

def _grouped_to_tiled(j):
    k_head, r = divmod(j, REP)
    return r * NK + k_head


def _rows(codes, heads, width):
    return np.concatenate([codes[h * width : (h + 1) * width] for h in heads])


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    fixture, signs, reference, out, report = convert_bonsai(tmp_path_factory.mktemp("bonsai"))
    assert report["chat_template_matches_gguf"] is True
    return fixture, signs, reference, out


def _parent(artifact, name):
    binding = artifact.directory.bindings[name]
    object_id = binding["object"] if "object" in binding else binding["parts"][0]["object"]
    return artifact.object(object_id)


def test_gdn_input_projection_is_one_t2_parent_in_grouped_head_order(converted):
    fixture, _, _, out = converted
    with Artifact(out) as artifact:
        obj = _parent(artifact, "text/layers/0/gdn/query")
        assert (obj.format, obj.layout) == (TERNARY_FORMAT, "ternary_row_k128_v1")
        assert obj.shape == (2 * KG + 2 * VG, H)
        for role in ("key", "value", "z"):
            assert _parent(artifact, f"text/layers/0/gdn/{role}").id == obj.id
        codes, scales = decode_ternary_words(artifact.read_object(obj.id), obj.shape, obj.format)
    qkv, qkv_scales = fixture.ternary["blk.0.attn_qkv.weight"]
    gate, gate_scales = fixture.ternary["blk.0.attn_gate.weight"]
    heads = [_grouped_to_tiled(j) for j in range(NV)]
    expected = np.concatenate(
        (qkv[: 2 * KG], _rows(qkv[2 * KG :], heads, DV), _rows(gate, heads, DV))
    )
    expected_scales = np.concatenate(
        (
            qkv_scales[: 2 * KG],
            _rows(qkv_scales[2 * KG :], heads, DV),
            _rows(gate_scales, heads, DV),
        )
    )
    np.testing.assert_array_equal(unpack_ternary_codes(codes, obj.format).numpy(), expected)
    np.testing.assert_array_equal(scales.numpy(), expected_scales)


def test_attention_and_mlp_parents_keep_the_bf16_path_row_assembly(converted):
    fixture, _, _, out = converted
    with Artifact(out) as artifact:
        attention = _parent(artifact, "text/layers/3/attention/query")
        codes, _ = decode_ternary_words(artifact.read_object(attention.id), attention.shape, attention.format)
        mlp = _parent(artifact, "text/layers/1/mlp/gate")
        mlp_codes, _ = decode_ternary_words(artifact.read_object(mlp.id), mlp.shape, mlp.format)
        down = _parent(artifact, "text/layers/1/gdn/output")
        down_codes, _ = decode_ternary_words(artifact.read_object(down.id), down.shape, down.format)
    q = fixture.ternary["blk.3.attn_q.weight"][0]
    query = np.concatenate([q[2 * h * HEAD_DIM : (2 * h + 1) * HEAD_DIM] for h in range(HEADS)])
    gate = np.concatenate([q[(2 * h + 1) * HEAD_DIM : (2 * h + 2) * HEAD_DIM] for h in range(HEADS)])
    expected = np.concatenate(
        (query, fixture.ternary["blk.3.attn_k.weight"][0], gate, fixture.ternary["blk.3.attn_v.weight"][0])
    )
    np.testing.assert_array_equal(unpack_ternary_codes(codes, attention.format).numpy(), expected)
    np.testing.assert_array_equal(
        unpack_ternary_codes(mlp_codes, mlp.format).numpy(),
        np.concatenate((fixture.ternary["blk.1.ffn_gate.weight"][0], fixture.ternary["blk.1.ffn_up.weight"][0])),
    )
    # out_proj's input axis is already grouped: no column permutation.
    np.testing.assert_array_equal(
        unpack_ternary_codes(down_codes, down.format).numpy(), fixture.ternary["blk.1.ssm_out.weight"][0]
    )


def test_output_head_and_embedding_keep_the_rotated_t2_words(converted):
    fixture, _, _, out = converted
    with Artifact(out) as artifact:
        for name, gguf in (("text/output_head", "output.weight"), ("text/token_embedding", "token_embd.weight")):
            obj = _parent(artifact, name)
            assert obj.format == TERNARY_FORMAT, name
            codes, scales = decode_ternary_words(artifact.read_object(obj.id), obj.shape, obj.format)
            stored_codes, stored_scales = fixture.ternary[gguf]
            np.testing.assert_array_equal(unpack_ternary_codes(codes, obj.format).numpy(), stored_codes)
            np.testing.assert_array_equal(scales.numpy(), stored_scales)


def test_gdn_vectors_norms_and_hadamard_metadata(converted):
    fixture, signs, _, out = converted
    with NInferArtifactStore(out) as store:
        heads = [_grouped_to_tiled(j) for j in range(NV)]
        d = fixture.dense
        checks = {
            "text/layers/0/gdn/a_log": np.log(-d["blk.0.ssm_a"][heads]),
            "text/layers/0/gdn/dt_bias": d["blk.0.ssm_dt.bias"][heads],
            "text/layers/0/gdn/a_projection": d["blk.0.ssm_alpha.weight"][heads],
            "text/layers/0/gdn/b_projection": d["blk.0.ssm_beta.weight"][heads],
            "text/layers/0/gdn/norm": d["blk.0.ssm_norm.weight"],
            "text/layers/0/input_norm": d["blk.0.attn_norm.weight"] - 1,
            "text/layers/3/post_attention_norm": d["blk.3.post_attention_norm.weight"] - 1,
            "text/layers/3/attention/query_norm": d["blk.3.attn_q_norm.weight"] - 1,
            "text/final_norm": d["output_norm.weight"] - 1,
        }
        conv = d["blk.0.ssm_conv1d.weight"].reshape(2 * KG + VG, CONV)
        channels = list(range(2 * KG)) + [2 * KG + h * DV + c for h in heads for c in range(DV)]
        checks["text/layers/0/gdn/convolution"] = conv[channels].T
        for name, expected in checks.items():
            got = store.dequantize(name).numpy().reshape(np.shape(expected))
            tolerance = 1e-6 if name.endswith(("a_log", "dt_bias")) else 1e-2
            np.testing.assert_allclose(got, expected, rtol=tolerance, atol=tolerance, err_msg=name)
        for width in (H, INTER):
            np.testing.assert_array_equal(
                store.dequantize(f"text/hadamard/signs_{width}").numpy(), signs[width]
            )
        config = store.directory.components["text"]["config"]["prism_hadamard"]
    assert config["block_size"] == 1024 and config["sign_widths"] == [H, INTER]
    assert config["signs"] == {str(w): f"text/hadamard/signs_{w}" for w in (H, INTER)}
    assert config["embedding_inverse"] is True
    assert set(config["rotated_inputs"]) == {
        "attention/query", "attention/key", "attention/gate", "attention/value",
        "attention/output", "gdn/query", "gdn/key", "gdn/value", "gdn/z", "gdn/output",
        "mlp/gate", "mlp/up", "mlp/down", "output_head",
    }


def test_ternary_projections_admit_a8_activations(converted):
    _, _, _, out = converted
    with Artifact(out) as artifact:
        policies = {use["parameter"]: use.get("activation_policy") for use in artifact.directory.uses}
    ternary = {
        name for name in policies
        if name == "text/output_head"
        or (name.startswith("text/layers/") and name.endswith(("/query", "/key", "/gate", "/value",
                                                               "/output", "/z", "/up", "/down")))
    }
    assert "text/output_head" in ternary and "text/layers/1/mlp/down" in ternary
    assert {policies[name] for name in ternary} == {"AllowA8"}
    assert policies["text/layers/0/gdn/a_projection"] != "AllowA8"


def test_mtp_is_copied_word_for_word(converted):
    _, _, reference, out = converted
    with NInferArtifactStore(reference) as source, NInferArtifactStore(out) as result:
        names = [name for name in source.parameters() if name.startswith("mtp/")]
        assert names and set(names) == {n for n in result.parameters() if n.startswith("mtp/")}
        for name in names:
            assert result.stored_format(name) == source.stored_format(name)
            if source.stored_format(name) == "q8_g32_fp16":
                rows = result.dequantize(name).shape[0]
                a, b = source.encoded_rows(name, 0, rows), result.encoded_rows(name, 0, rows)
                assert torch.equal(a.codes, b.codes) and torch.equal(a.scales, b.scales), name
            else:
                assert torch.equal(source.dequantize(name), result.dequantize(name)), name


def test_t5_conversion_keeps_the_exact_gguf_trits(tmp_path, monkeypatch):
    # The recipe's stored ternary format switched to base-3 t5 (design doc 9.1, step 2).
    import tools.convert.official_recipes as recipes

    monkeypatch.setattr(recipes, "TERNARY_FORMAT", "t5_g128_fp16")
    fixture, _, _, out, _ = convert_bonsai(tmp_path)
    with Artifact(out) as artifact:
        for name, gguf in (
            ("text/output_head", "output.weight"),
            ("text/token_embedding", "token_embd.weight"),
            ("text/layers/1/gdn/output", "blk.1.ssm_out.weight"),
        ):
            obj = _parent(artifact, name)
            assert (obj.format, obj.layout) == ("t5_g128_fp16", "ternary_row_k128_v1"), name
            codes, scales = decode_ternary_words(artifact.read_object(obj.id), obj.shape, obj.format)
            stored_codes, stored_scales = fixture.ternary[gguf]
            np.testing.assert_array_equal(unpack_ternary_codes(codes, obj.format).numpy(), stored_codes)
            np.testing.assert_array_equal(scales.numpy(), stored_scales)
