"""End-to-end `bonsai2_27b` conversion of a synthetic Prism GGUF plus a reference artifact.

The oracle is independent of the converter: the fixture keeps its stored ternary codes,
and the expected artifact contents are rebuilt here from the design-doc conventions
(sections 1.5, 2.1 and `bonsai-ternary-conversion.md`) with literal loops and a
Kronecker-built Hadamard matrix.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest
import torch

from tools.artifact.codecs.row_split import dequantize_row_split
from tools.artifact.codecs.ternary import decode_ternary_words, unpack_ternary_codes
from tools.artifact.reader import Artifact
from tools.convert.__main__ import main as convert_main
from tools.convert.bonsai_base import write_base
from tools.convert.pipeline import convert
from tools.convert.qwen3_5 import build_model
from tools.convert.recipe import Recipe
from tools.convert.sources.logical import array_source
from tools.convert.sources.ninfer_artifact import NInferArtifactStore
from tools.convert.sources.safetensors import SafetensorsSource

from .gguf_fixtures import build_gguf

H, INTER, VOCAB, LAYERS = 1024, 2048, 8, 4
NK, NV, DK, DV = 2, 4, 256, 256
HEADS, KV, HEAD_DIM = 4, 1, 256
KG, VG = NK * DK, NV * DV
Q, KVW = HEADS * HEAD_DIM, KV * HEAD_DIM
CONV = 4
REP = NV // NK


def _config():
    return {
        "architectures": ["Qwen3_5ForCausalLM"],
        "hidden_size": H,
        "vocab_size": VOCAB,
        "num_hidden_layers": LAYERS,
        "max_position_embeddings": 4096,
        "full_attention_interval": 4,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV,
        "head_dim": HEAD_DIM,
        "rope_parameters": {"partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10]},
        "linear_num_key_heads": NK,
        "linear_key_head_dim": DK,
        "linear_num_value_heads": NV,
        "linear_value_head_dim": DV,
        "linear_conv_kernel_dim": CONV,
        "intermediate_size": INTER,
        "tie_word_embeddings": False,
    }


def _resources(path):
    path.mkdir()
    (path / "config.json").write_text(json.dumps(_config()))
    (path / "tokenizer.json").write_text(
        json.dumps({"model": {"vocab": {str(i): i for i in range(6)}}})
    )
    (path / "tokenizer_config.json").write_text("{}")
    (path / "generation_config.json").write_text("{}")
    (path / "chat_template.jinja").write_text("{{ messages }}")


def _hadamard(n):
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.kron(np.array([[1.0, 1.0], [1.0, -1.0]]), h)
    return h / np.sqrt(n)


class _Fixture:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.ternary = {}  # gguf name -> (codes uint8 [N, K], scales fp16 [N, K/128])
        self.dense = {}  # gguf name -> float32 array in GGUF memory order
        self.tensors = []

    def ternary_tensor(self, name, n, k):
        codes = self.rng.integers(0, 3, (n, k), dtype=np.uint8)
        scales = (self.rng.random((n, k // 128)) * 0.1 + 0.01).astype(np.float16)
        raw = bytearray()
        for row in range(n):
            for block in range(k // 128):
                c = codes[row, block * 128 : (block + 1) * 128]
                raw += scales[row, block].tobytes()
                raw += bytes(
                    int(c[4 * i] | c[4 * i + 1] << 2 | c[4 * i + 2] << 4 | c[4 * i + 3] << 6)
                    for i in range(32)
                )
        self.ternary[name] = (codes, scales)
        self.tensors.append((name, (k, n), 142, bytes(raw)))

    def f32(self, name, ne, values):
        values = np.asarray(values, dtype=np.float32)
        self.dense[name] = values
        self.tensors.append((name, ne, 0, values.tobytes()))

    def bf16(self, name, n, k):
        values = torch.randn(n, k, generator=torch.Generator().manual_seed(len(self.tensors)))
        words = values.to(torch.bfloat16)
        self.dense[name] = words.float().numpy()
        self.tensors.append((name, (k, n), 30, words.view(torch.int16).numpy().tobytes()))


def _gguf(path):
    fixture = _Fixture()
    rng = fixture.rng
    fixture.ternary_tensor("token_embd.weight", VOCAB, H)
    fixture.ternary_tensor("output.weight", VOCAB, H)
    fixture.f32("output_norm.weight", (H,), 1.0 + rng.random(H) * 0.1)
    rotated = ["output.weight"]
    for i in range(LAYERS):
        b = f"blk.{i}."
        fixture.f32(b + "attn_norm.weight", (H,), 1.0 + rng.random(H) * 0.1)
        fixture.f32(b + "post_attention_norm.weight", (H,), 1.0 + rng.random(H) * 0.1)
        for name, n, k in (("ffn_gate", INTER, H), ("ffn_up", INTER, H), ("ffn_down", H, INTER)):
            fixture.ternary_tensor(b + name + ".weight", n, k)
            rotated.append(b + name + ".weight")
        if i == 3:
            for name, n, k in (
                ("attn_q", 2 * Q, H),
                ("attn_k", KVW, H),
                ("attn_v", KVW, H),
                ("attn_output", H, Q),
            ):
                fixture.ternary_tensor(b + name + ".weight", n, k)
                rotated.append(b + name + ".weight")
            fixture.f32(b + "attn_q_norm.weight", (HEAD_DIM,), 1.0 + rng.random(HEAD_DIM))
            fixture.f32(b + "attn_k_norm.weight", (HEAD_DIM,), 1.0 + rng.random(HEAD_DIM))
        else:
            for name, n, k in (
                ("attn_qkv", 2 * KG + VG, H),
                ("attn_gate", VG, H),
                ("ssm_out", H, VG),
            ):
                fixture.ternary_tensor(b + name + ".weight", n, k)
                rotated.append(b + name + ".weight")
            fixture.bf16(b + "ssm_alpha.weight", NV, H)
            fixture.bf16(b + "ssm_beta.weight", NV, H)
            fixture.f32(b + "ssm_a", (NV,), -np.exp(rng.random(NV)))
            fixture.f32(b + "ssm_dt.bias", (NV,), rng.random(NV))
            # Memory order is (channels, taps) although ne is recorded as [taps, channels].
            fixture.f32(b + "ssm_conv1d.weight", (CONV, 2 * KG + VG), rng.random((2 * KG + VG, CONV)))
            fixture.f32(b + "ssm_norm.weight", (DV,), rng.random(DV))
    signs = {w: rng.choice([-1, 1], w).astype(np.int8) for w in (H, INTER)}
    metadata = {
        "general.architecture": (8, "qwen35"),
        "general.file_type": (4, 142),
        "qwen35.block_count": (4, LAYERS),
        "qwen35.embedding_length": (4, H),
        "qwen35.feed_forward_length": (4, INTER),
        "qwen35.attention.head_count": (4, HEADS),
        "qwen35.attention.head_count_kv": (4, KV),
        "qwen35.attention.key_length": (4, HEAD_DIM),
        "qwen35.ssm.conv_kernel": (4, CONV),
        "qwen35.ssm.state_size": (4, DK),
        "qwen35.ssm.group_count": (4, NK),
        "qwen35.ssm.time_step_rank": (4, NV),
        "qwen35.ssm.inner_size": (4, VG),
        "qwen35.full_attention_interval": (4, 4),
        "prism.hadamard.version": (4, 1),
        "prism.hadamard.block_size": (4, 1024),
        "prism.hadamard.sign_mode": (8, "explicit"),
        "prism.hadamard.sign_widths": (9, (4, [H, INTER])),
        "prism.hadamard.sign_values": (9, (1, [int(v) for w in (H, INTER) for v in signs[w]])),
        "prism.hadamard.weight_names": (9, (8, rotated)),
        "prism.hadamard.inverse_weight_names": (9, (8, ["token_embd.weight"])),
        "prism.hadamard.gdn_v_grouped": (7, True),
        "tokenizer.chat_template": (8, "{{ messages }}"),
    }
    path.write_bytes(build_gguf(metadata, fixture.tensors))
    return fixture, signs


def _reference(tmp_path):
    """A Qwen3.8-shaped artifact whose MTP head is Q8 (what the recipe copies)."""
    base = tmp_path / "reference-src"
    _resources(base)
    with SafetensorsSource(base) as store:
        model = build_model(store, components=("text", "mtp"))
        recipe = Recipe(model)
        generator = torch.Generator().manual_seed(11)
        for name, parameter in model.parameters.items():
            values = torch.randn(parameter.shape, generator=generator)
            source = array_source(values, name)
            if parameter.projection and name.startswith("mtp/"):
                recipe.assign(name, format="q8_g32_fp16", method="grouped_absmax", source=source)
            else:
                recipe.assign(name, source=source)
        path = tmp_path / "reference.ninfer"
        convert(model, recipe, path, device="cpu")
    return path


def _grouped_to_tiled(j):
    k_head, r = divmod(j, REP)
    return r * NK + k_head


def _rows(codes, heads, width):
    return np.concatenate([codes[h * width : (h + 1) * width] for h in heads])


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("bonsai")
    fixture, signs = _gguf(tmp_path / "bonsai.gguf")
    reference = _reference(tmp_path)
    report = write_base(tmp_path / "bonsai.gguf", reference, tmp_path / "base")
    assert report["chat_template_matches_gguf"] is True
    out = tmp_path / "bonsai.ninfer"
    convert_main(
        [
            "--model", str(tmp_path / "base"),
            "--recipe", "bonsai2_27b",
            "--components", "text,mtp",
            "--source", f"gguf={tmp_path / 'bonsai.gguf'}",
            "--source", f"mtp={reference}",
            "--device", "cpu",
            "--rows-per-chunk", "100",
            "--out", str(out),
        ]
    )
    return fixture, signs, reference, out


def _parent(artifact, name):
    binding = artifact.directory.bindings[name]
    object_id = binding["object"] if "object" in binding else binding["parts"][0]["object"]
    return artifact.object(object_id)


def test_gdn_input_projection_is_one_t2_parent_in_grouped_head_order(converted):
    fixture, _, _, out = converted
    with Artifact(out) as artifact:
        obj = _parent(artifact, "text/layers/0/gdn/query")
        assert (obj.format, obj.layout) == ("t2_g128_fp16", "ternary_row_k128_v1")
        assert obj.shape == (2 * KG + 2 * VG, H)
        for role in ("key", "value", "z"):
            assert _parent(artifact, f"text/layers/0/gdn/{role}").id == obj.id
        codes, scales = decode_ternary_words(artifact.read_object(obj.id), obj.shape)
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
    np.testing.assert_array_equal(unpack_ternary_codes(codes).numpy(), expected)
    np.testing.assert_array_equal(scales.numpy(), expected_scales)


def test_attention_and_mlp_parents_keep_the_bf16_path_row_assembly(converted):
    fixture, _, _, out = converted
    with Artifact(out) as artifact:
        attention = _parent(artifact, "text/layers/3/attention/query")
        codes, _ = decode_ternary_words(artifact.read_object(attention.id), attention.shape)
        mlp = _parent(artifact, "text/layers/1/mlp/gate")
        mlp_codes, _ = decode_ternary_words(artifact.read_object(mlp.id), mlp.shape)
        down = _parent(artifact, "text/layers/1/gdn/output")
        down_codes, _ = decode_ternary_words(artifact.read_object(down.id), down.shape)
    q = fixture.ternary["blk.3.attn_q.weight"][0]
    query = np.concatenate([q[2 * h * HEAD_DIM : (2 * h + 1) * HEAD_DIM] for h in range(HEADS)])
    gate = np.concatenate([q[(2 * h + 1) * HEAD_DIM : (2 * h + 2) * HEAD_DIM] for h in range(HEADS)])
    expected = np.concatenate(
        (query, fixture.ternary["blk.3.attn_k.weight"][0], gate, fixture.ternary["blk.3.attn_v.weight"][0])
    )
    np.testing.assert_array_equal(unpack_ternary_codes(codes).numpy(), expected)
    np.testing.assert_array_equal(
        unpack_ternary_codes(mlp_codes).numpy(),
        np.concatenate((fixture.ternary["blk.1.ffn_gate.weight"][0], fixture.ternary["blk.1.ffn_up.weight"][0])),
    )
    # out_proj's input axis is already grouped: no column permutation.
    np.testing.assert_array_equal(
        unpack_ternary_codes(down_codes).numpy(), fixture.ternary["blk.1.ssm_out.weight"][0]
    )


def test_output_head_and_embedding_are_primal_q8(converted):
    fixture, signs, _, out = converted
    h = _hadamard(1024)
    with Artifact(out) as artifact:
        for name, gguf in (("text/output_head", "output.weight"), ("text/token_embedding", "token_embd.weight")):
            obj = _parent(artifact, name)
            assert obj.format == "q8_g32_fp16"
            got = dequantize_row_split(
                artifact.read_object(obj.id), obj.format, obj.shape, dtype=torch.float32
            ).numpy()
            codes, scales = fixture.ternary[gguf]
            rotated = (codes.astype(np.float64) - 1) * np.repeat(scales.astype(np.float64), 128, axis=1)
            primal = (rotated @ h) * signs[H][None, :].astype(np.float64)
            error = np.linalg.norm(got - primal) / np.linalg.norm(primal)
            assert error < 1e-2, (name, error)


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
    assert set(config["rotated_inputs"]) == {
        "attention/query", "attention/key", "attention/gate", "attention/value",
        "attention/output", "gdn/query", "gdn/key", "gdn/value", "gdn/z", "gdn/output",
        "mlp/gate", "mlp/up", "mlp/down",
    }


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
