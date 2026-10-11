"""PyTorch MTP layer of NInfer's Qwen3.5 models, for self-distillation of Bonsai's draft head.

Mirrors src/models/qwen3_5/execution/text.cpp (mtp stem :283-319, tail :321-410):
  e  = rms1p(embed(next_token), embedding_norm); h = rms1p(hidden, hidden_norm)
  x  = input_projection([e; h]);  a = rms1p(x, input_norm)
  q, gate, k, v projections; per-head rms1p(q, query_norm), rms1p(k, key_norm); partial RoPE
  x += output(causal_gqa_attention(q, k, v) * sigmoid(gate))
  x += down(silu(gate(m)) * up(m)), m = rms1p(x, post_attention_norm)
  mtp_hidden = rms1p(x, final_norm); draft = token_ids[argmax(proposal_head @ mtp_hidden)]
rms1p(x, w) = x / sqrt(mean(x^2) + eps) * (1 + w).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tools.artifact.codecs.ternary import dequantize_ternary_words, decode_ternary_words
from tools.artifact.formats import get_format
from tools.convert.quantization.hadamard import unrotate_rows
from tools.convert.sources.ninfer_artifact import NInferArtifactStore

H, HEADS, KV_HEADS, HEAD_DIM, INTER = 5120, 24, 4, 256, 17408
ROTARY, THETA, EPS = 64, 10_000_000.0, 1e-6
LAYER = "mtp/layers/0/"
WEIGHTS = [
    "mtp/input_projection", "mtp/embedding_norm", "mtp/hidden_norm", "mtp/final_norm",
    LAYER + "input_norm", LAYER + "post_attention_norm",
    LAYER + "attention/query", LAYER + "attention/gate", LAYER + "attention/key",
    LAYER + "attention/value", LAYER + "attention/query_norm", LAYER + "attention/key_norm",
    LAYER + "attention/output", LAYER + "mlp/gate", LAYER + "mlp/up", LAYER + "mlp/down",
]


def load_mtp_weights(artifact: str) -> dict[str, torch.Tensor]:
    with NInferArtifactStore(artifact) as store:
        return {name: store.dequantize(name) for name in WEIGHTS}


def load_proposal(artifact: str) -> tuple[torch.Tensor, torch.Tensor]:
    with NInferArtifactStore(artifact) as store:
        head = store.dequantize("proposal/head").to(torch.bfloat16)
        ids = store.dequantize("proposal/token_ids").to(torch.int64)
    return head, ids


def load_embedding(artifact: str, cache: Path) -> torch.Tensor:
    """Bonsai's primal embedding table, BF16 [vocab, H]: signs * H_1024(decoded t5 row)."""
    if cache.exists():
        return torch.load(cache)
    with NInferArtifactStore(artifact) as store:
        parts, objects = store._parts("text/token_embedding")
        (object_id, begin, end), = parts
        obj = objects[object_id]
        n, k = obj.shape
        if begin != 0 or end != n * k:
            raise ValueError("token_embedding is not a whole object")
        codes, scales = decode_ternary_words(store.read_object(object_id), obj.shape, obj.format)
        signs = store.dequantize("text/hadamard/signs_5120")
    table = torch.empty((n, k), dtype=torch.bfloat16)
    for row in range(0, n, 8192):
        values = dequantize_ternary_words(codes[row : row + 8192], scales[row : row + 8192],
                                          format=obj.format)
        table[row : row + 8192] = unrotate_rows(values, signs).to(torch.bfloat16)
    torch.save(table, cache)
    return table


def rms1p(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + w.float())).to(x.dtype)


def rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Partial rotary on the first ROTARY dims, rotate-half layout. x: [T, heads, 256]."""
    inv = THETA ** (-torch.arange(0, ROTARY, 2, device=x.device, dtype=torch.float32) / ROTARY)
    freqs = positions.float()[:, None] * inv[None, :]
    cos = torch.cat([freqs.cos(), freqs.cos()], -1)[:, None, :]
    sin = torch.cat([freqs.sin(), freqs.sin()], -1)[:, None, :]
    rot = x[..., :ROTARY].float()
    half = ROTARY // 2
    rotated = torch.cat([-rot[..., half:], rot[..., :half]], -1)
    out = rot * cos + rotated * sin
    return torch.cat([out.to(x.dtype), x[..., ROTARY:]], -1)


class Mtp(torch.nn.Module):
    def __init__(self, weights: dict[str, torch.Tensor]):
        super().__init__()
        self.p = torch.nn.ParameterDict(
            {name.replace("/", "__"): torch.nn.Parameter(value.float()) for name, value in weights.items()}
        )

    def w(self, name: str) -> torch.Tensor:
        return self.p[name.replace("/", "__")]

    def step(self, hidden: torch.Tensor, next_embedding: torch.Tensor, positions: torch.Tensor,
             past: list[tuple[torch.Tensor, torch.Tensor]] | None = None):
        """One MTP pass over T columns of one sequence; returns (mtp_hidden [T, H], (k, v)).

        Without `past` the pass is the verified/prefill pass: column t attends to columns 0..t.
        With `past` = [(k1, v1), ..., (k_{s-1}, v_{s-1})] the pass is draft step s: column t
        attends to the first pass's columns 0..t, to each later pass's own column t, and to
        itself, as the runtime's autoregressive draft attends to its MTP KV.
        """
        dt = torch.bfloat16
        w = lambda n: self.w(n).to(dt)  # noqa: E731
        e = rms1p(next_embedding.to(dt), self.w("mtp/embedding_norm"))
        h = rms1p(hidden.to(dt), self.w("mtp/hidden_norm"))
        x = torch.cat([e, h], -1) @ w("mtp/input_projection").T
        a = rms1p(x, self.w(LAYER + "input_norm"))
        t = a.shape[0]
        q = (a @ w(LAYER + "attention/query").T).view(t, HEADS, HEAD_DIM)
        g = a @ w(LAYER + "attention/gate").T
        k = (a @ w(LAYER + "attention/key").T).view(t, KV_HEADS, HEAD_DIM)
        v = (a @ w(LAYER + "attention/value").T).view(t, KV_HEADS, HEAD_DIM)
        q = rope(rms1p(q, self.w(LAYER + "attention/query_norm")), positions)
        k = rope(rms1p(k, self.w(LAYER + "attention/key_norm")), positions)
        scale = 1.0 / math.sqrt(HEAD_DIM)
        if not past:
            o = F.scaled_dot_product_attention(
                q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
                is_causal=True, enable_gqa=True, scale=scale,
            )
        else:
            keys = torch.cat([pk[:t] for pk, _ in past] + [k], 0)
            values = torch.cat([pv[:t] for _, pv in past] + [v], 0)
            idx = torch.arange(t, device=q.device)
            causal = idx[None, :] <= idx[:, None]
            diagonal = idx[None, :] == idx[:, None]
            mask = torch.cat([causal] + [diagonal] * len(past), 1)
            o = F.scaled_dot_product_attention(
                q.transpose(0, 1)[None], keys.transpose(0, 1)[None], values.transpose(0, 1)[None],
                attn_mask=mask[None, None], enable_gqa=True, scale=scale,
            )
        o = o[0].transpose(0, 1).reshape(t, HEADS * HEAD_DIM)
        x = x + (o * torch.sigmoid(g)) @ w(LAYER + "attention/output").T
        m = rms1p(x, self.w(LAYER + "post_attention_norm"))
        x = x + (F.silu(m @ w(LAYER + "mlp/gate").T) * (m @ w(LAYER + "mlp/up").T)) @ w(LAYER + "mlp/down").T
        return rms1p(x, self.w("mtp/final_norm")), (k, v)

    def forward(self, hidden: torch.Tensor, next_embedding: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        """hidden, next_embedding: [T, H] for one sequence; returns mtp_hidden [T, H]."""
        return self.step(hidden, next_embedding, positions)[0]

    def chain(self, hidden: torch.Tensor, embeddings: list[torch.Tensor], positions: torch.Tensor):
        """Draft steps 1..S teacher-forced: step s takes step s-1's output and the embedding of
        the token s positions ahead, at positions + s - 1. Step s has len(embeddings[s-1])
        columns (later steps have fewer valid columns). Returns the S mtp_hidden tensors."""
        outs, past = [], []
        h = hidden
        for s, emb in enumerate(embeddings):
            t = emb.shape[0]
            out, kv = self.step(h[:t], emb, positions[:t] + s, past if s else None)
            outs.append(out)
            past.append(kv)
            h = out
        return outs


class Dataset:
    """ninfer-score-export output: per text, tokens, hidden rows, teacher argmax rows."""

    def __init__(self, directory: str):
        d = Path(directory)
        lines = (d / "index.jsonl").read_text().splitlines()
        self.header = json.loads(lines[0])
        self.texts = [json.loads(line) for line in lines[1:]]
        hs = self.header["hidden_size"]
        self.tokens = np.memmap(d / "tokens.i32", dtype=np.int32, mode="r")
        self.argmax = np.memmap(d / "argmax.i32", dtype=np.int32, mode="r")
        self.hidden = np.memmap(d / "hidden.bf16", dtype=np.uint16, mode="r").reshape(-1, hs)

    def sequence(self, i: int):
        """Columns t = 0..n-3: hidden h_t, next token tokens[t+1], target argmax at row t+1."""
        meta = self.texts[i]
        n, to, ro = meta["tokens"], meta["token_offset"], meta["row_offset"]
        tokens = torch.from_numpy(np.array(self.tokens[to : to + n], dtype=np.int64))
        hidden = torch.from_numpy(np.array(self.hidden[ro : ro + n - 2])).view(torch.bfloat16)
        target = torch.from_numpy(np.array(self.argmax[ro + 1 : ro + n - 1], dtype=np.int64))
        return hidden, tokens[1 : n - 1], target

    def chain(self, i: int, steps: int):
        """Draft step s (1-based) at column t: next token tokens[t+s], target argmax row t+s,
        for t = 0..n-2-s. Returns hidden rows for step 1 and per-step (next tokens, targets)."""
        meta = self.texts[i]
        n, to, ro = meta["tokens"], meta["token_offset"], meta["row_offset"]
        tokens = torch.from_numpy(np.array(self.tokens[to : to + n], dtype=np.int64))
        argmax = torch.from_numpy(np.array(self.argmax[ro : ro + n - 1], dtype=np.int64))
        hidden = torch.from_numpy(np.array(self.hidden[ro : ro + n - 2])).view(torch.bfloat16)
        per_step = []
        for s in range(1, steps + 1):
            count = n - 1 - s
            if count <= 0:
                break
            per_step.append((tokens[s : s + count], argmax[s : s + count]))
        return hidden, per_step
