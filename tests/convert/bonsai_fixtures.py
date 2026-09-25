"""Synthetic Prism Bonsai inputs shared by the converter tests and the C++ loading driver.

`write_gguf` builds a Bonsai-shaped PQ2_0 GGUF (hidden 1024, 4 layers, a nontrivial 2 x 2
GDN head permutation) and keeps the stored codes for independent oracles; `write_reference`
converts a random Qwen3.8-shaped reference artifact with a Q8 MTP head; `convert_bonsai`
runs `bonsai_base` and the `bonsai2_27b` (or a `bonsai2_27b_mtp_*`) CLI on both.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from tools.convert.__main__ import main as convert_main
from tools.convert.bonsai_base import write_base
from tools.convert.pipeline import convert
from tools.convert.qwen3_5 import build_model
from tools.convert.recipe import Recipe
from tools.convert.sources.logical import array_source
from tools.convert.sources.safetensors import SafetensorsSource

from .gguf_fixtures import build_gguf

H, INTER, VOCAB, LAYERS = 1024, 2048, 272, 4
NK, NV, DK, DV = 2, 4, 256, 256
HEADS, KV, HEAD_DIM = 4, 1, 256
KG, VG = NK * DK, NV * DV
Q, KVW = HEADS * HEAD_DIM, KV * HEAD_DIM
CONV = 4
REP = NV // NK


def config():
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


SPECIAL_TOKENS = (
    "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>",
    "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>",
)


def _added(index, content):
    return {"id": index, "content": content, "special": True, "single_word": False,
            "lstrip": False, "rstrip": False, "normalized": False}


def write_resources(path):
    """Config plus a byte-level BPE tokenizer the C++ frontend accepts (as test_loading.cpp)."""
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config()))
    vocab, extra = {}, 256
    for byte in range(256):
        visible = 33 <= byte <= 126 or 161 <= byte <= 172 or byte >= 174
        codepoint = byte if visible else extra
        extra += 0 if visible else 1
        vocab[chr(codepoint)] = byte
    specials = [_added(256 + i, token) for i, token in enumerate(SPECIAL_TOKENS)]
    tokenizer = {"model": {"type": "BPE", "vocab": vocab, "merges": []}, "added_tokens": specials[:8]}
    decoder = {str(token["id"]): token for token in specials}
    (path / "tokenizer.json").write_text(json.dumps(tokenizer))
    (path / "tokenizer_config.json").write_text(json.dumps({"added_tokens_decoder": decoder}))
    (path / "generation_config.json").write_text(json.dumps({"eos_token_id": [256, 258]}))
    (path / "chat_template.jinja").write_text("{{ messages }}")


def hadamard(n):
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.kron(np.array([[1.0, 1.0], [1.0, -1.0]]), h)
    return h / np.sqrt(n)


class GgufFixture:
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


def write_gguf(path):
    fixture = GgufFixture()
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


def write_reference(tmp_path):
    """A Qwen3.8-shaped artifact whose MTP head is Q8 (what the recipe copies)."""
    base = tmp_path / "reference-src"
    write_resources(base)
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


def convert_bonsai(tmp_path, *extra, recipe="bonsai2_27b"):
    """Write the GGUF, reference and base directory, then run the CLI; return all paths."""
    fixture, signs = write_gguf(tmp_path / "bonsai.gguf")
    reference = write_reference(tmp_path)
    report = write_base(tmp_path / "bonsai.gguf", reference, tmp_path / "base")
    out = tmp_path / "bonsai.ninfer"
    convert_main(
        [
            "--model", str(tmp_path / "base"),
            "--recipe", recipe,
            "--components", "text,mtp",
            "--source", f"gguf={tmp_path / 'bonsai.gguf'}",
            "--source", f"mtp={reference}",
            "--device", "cpu",
            "--rows-per-chunk", "100",
            "--out", str(out),
            *extra,
        ]
    )
    return fixture, signs, reference, out, report
