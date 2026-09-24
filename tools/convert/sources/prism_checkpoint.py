"""A Prism Bonsai GGUF presented as the equivalent Hugging Face Qwen3.5 checkpoint.

The Qwen3.5 builder (`tools/convert/qwen3_5.py`) addresses weights by HF name. This store
answers the same names from a PrismML `qwen35` GGUF (design doc sections 1 and 4) with two
views per ternary matrix:

- values: the HF-equivalent primal-basis float32 weight `W = (W' @ H) * s` per 1024 block
  (`unrotate_rows`), for the embedding the primal table. A recipe that re-quantizes a
  ternary tensor (the v1 Q8 output head and embedding) therefore folds the rotation
  automatically.
- encoded rows: the exact rotated ternary words as `t2_g128_fp16`, for `import_encoded`.
  Their column axis stays in the rotated basis; the runtime applies the Hadamard to a
  projection's activation, or to a gathered embedding row (`prism_hadamard` text config).
  Projections (`prism.hadamard.weight_names`) and the embedding
  (`inverse_weight_names`) share the algebra: the logical matrix is `W' H S`.

Conventions measured by M0 (`docs/maintainer/bonsai-ternary-conversion.md`):

- GDN tensors indexed by value head are stored in llama.cpp's tiled order
  (`i = rep * nk + k_head`); HF/NInfer use the grouped order (`j = rep_count * k_head +
  rep`). HF row/element `j` reads GGUF head `(j % rep_count) * nk + j // rep_count`. This
  applies to `in_proj_qkv`'s value rows, `in_proj_z`, `in_proj_a`, `in_proj_b`, `A_log`,
  `dt_bias` and the value channels of `conv1d`. `out_proj`'s input axis is already grouped
  (`prism.hadamard.gdn_v_grouped`).
- `A_log = log(-ssm_a)`.
- Every RMSNorm weight except the GDN gated norm is stored as `1 + w`; HF stores `w`.
- `ssm_conv1d.weight`'s bytes are row-major `(channels, taps)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from tools.convert.quantization.hadamard import unrotate_rows
from .gguf_reader import GgufStore
from .logical import EncodedRows, LogicalSource
from .prism_gguf import (
    PQ2_0_TYPE,
    PTQ1_0_TYPE,
    assert_prism_ternary_gguf,
    dequantize_rows,
    sign_vectors,
    ternary_rows,
)
from .safetensors import TensorInfo

TERNARY_FORMAT = "t2_g128_fp16"
_TERNARY_TYPES = (PTQ1_0_TYPE, PQ2_0_TYPE)
_F32, _BF16 = 0, 30


@dataclass(frozen=True, slots=True)
class _Entry:
    gguf: str
    shape: tuple[int, ...]
    kind: str  # "matrix" or "vector"
    row_map: np.ndarray | None = None
    vector: Callable[[np.ndarray], np.ndarray] | None = None


def grouped_head_sources(nk: int, rep: int) -> np.ndarray:
    """GGUF (tiled) head index that holds HF (grouped) value head `j`."""
    j = np.arange(nk * rep)
    return (j % rep) * nk + j // rep


def _expand(heads: np.ndarray, width: int) -> np.ndarray:
    return (heads[:, None] * width + np.arange(width)[None, :]).reshape(-1)


class PrismCheckpoint:
    """Read-only HF-named view over a Prism `qwen35` GGUF, duck-typed like SafetensorsSource."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.config: dict = {}
        self.bytes_read = 0
        self._store = GgufStore(self.path)
        try:
            assert_prism_ternary_gguf(self._store)
            self._entries = self._map()
            self.signs = {
                width: signs.to(torch.float32)
                for width, signs in sign_vectors(self._store).items()
            }
        except BaseException:
            self._store.close()
            raise
        meta = self._store.metadata
        self.rotated = frozenset(meta.get("prism.hadamard.weight_names", ()))
        self.inverse = frozenset(meta.get("prism.hadamard.inverse_weight_names", ()))
        if not meta.get("prism.hadamard.gdn_v_grouped", False):
            raise ValueError(f"{self.path}: expected prism.hadamard.gdn_v_grouped=true")

    @property
    def metadata(self) -> dict:
        return self._store.metadata

    def _meta(self, key: str) -> int:
        value = self._store.metadata.get("qwen35." + key)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{self.path}: missing positive qwen35.{key}")
        return value

    def _map(self) -> dict[str, _Entry]:
        store = self._store
        h = self._meta("embedding_length")
        nk, nv = self._meta("ssm.group_count"), self._meta("ssm.time_step_rank")
        dk, inner = self._meta("ssm.state_size"), self._meta("ssm.inner_size")
        if nv % nk or inner % nv:
            raise ValueError(f"{self.path}: inconsistent GDN head geometry")
        rep, dv = nv // nk, inner // nv
        kg = nk * dk
        channels = 2 * kg + inner
        heads = grouped_head_sources(nk, rep)
        value_rows = _expand(heads, dv)
        entries: dict[str, _Entry] = {}

        def matrix(hf, gguf, row_map=None):
            info = store.tensor(gguf)
            if len(info.shape) != 2:
                raise ValueError(f"{gguf}: expected a matrix, got ne={info.shape}")
            k, n = info.shape
            if row_map is not None and sorted(row_map.tolist()) != list(range(n)):
                raise ValueError(f"{gguf}: head permutation does not cover {n} rows")
            entries[hf] = _Entry(gguf, (n, k), "matrix", row_map)

        def vector(hf, gguf, shape, transform=lambda x: x):
            entries[hf] = _Entry(gguf, tuple(shape), "vector", vector=transform)

        def norm(x):
            return x - 1.0

        def a_log(x):
            if bool((x >= 0).any()):
                raise ValueError("ssm_a must be negative (ssm_a = -exp(A_log))")
            return np.log(-x[heads])

        channel_map = np.concatenate((np.arange(2 * kg), 2 * kg + value_rows))

        def conv(x):
            taps = x.size // channels
            return x.reshape(channels, taps)[channel_map].reshape(channels, 1, taps)

        matrix("embed_tokens.weight", "token_embd.weight")
        matrix("lm_head.weight", "output.weight")
        vector("norm.weight", "output_norm.weight", (h,), norm)
        layers = self._meta("block_count")
        for i in range(layers):
            p, g = f"layers.{i}.", f"blk.{i}."
            vector(p + "input_layernorm.weight", g + "attn_norm.weight", (h,), norm)
            vector(
                p + "post_attention_layernorm.weight",
                g + "post_attention_norm.weight",
                (h,),
                norm,
            )
            for role, name in (("gate", "ffn_gate"), ("up", "ffn_up"), ("down", "ffn_down")):
                matrix(p + f"mlp.{role}_proj.weight", g + name + ".weight")
            if g + "attn_qkv.weight" in store.tensors:
                a = p + "linear_attn."
                matrix(
                    a + "in_proj_qkv.weight",
                    g + "attn_qkv.weight",
                    np.concatenate((np.arange(2 * kg), 2 * kg + value_rows)),
                )
                matrix(a + "in_proj_z.weight", g + "attn_gate.weight", value_rows)
                matrix(a + "in_proj_a.weight", g + "ssm_alpha.weight", heads)
                matrix(a + "in_proj_b.weight", g + "ssm_beta.weight", heads)
                matrix(a + "out_proj.weight", g + "ssm_out.weight")
                vector(a + "A_log", g + "ssm_a", (nv,), a_log)
                vector(a + "dt_bias", g + "ssm_dt.bias", (nv,), lambda x: x[heads])
                taps = self._meta("ssm.conv_kernel")
                vector(a + "conv1d.weight", g + "ssm_conv1d.weight", (channels, 1, taps), conv)
                vector(a + "norm.weight", g + "ssm_norm.weight", (dv,))
            else:
                s = p + "self_attn."
                for role, name in (("q", "attn_q"), ("k", "attn_k"), ("v", "attn_v")):
                    matrix(s + f"{role}_proj.weight", g + name + ".weight")
                matrix(s + "o_proj.weight", g + "attn_output.weight")
                d = self._meta("attention.key_length")
                vector(s + "q_norm.weight", g + "attn_q_norm.weight", (d,), norm)
                vector(s + "k_norm.weight", g + "attn_k_norm.weight", (d,), norm)
        return entries

    # SafetensorsSource interface used by qwen3_5._Builder and tensor_source.

    def _entry(self, name: str) -> _Entry:
        for prefix in ("model.language_model.", "model."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
        try:
            return self._entries[name]
        except KeyError:
            raise ValueError(f"{self.path}: no Bonsai tensor for {name!r}") from None

    def has(self, name: str) -> bool:
        try:
            self._entry(name)
        except ValueError:
            return False
        return True

    def describe(self, name: str) -> TensorInfo:
        entry = self._entry(name)
        return TensorInfo(self.path, entry.shape, "F32", 0, prod(entry.shape) * 4)

    def ternary(self, name: str) -> bool:
        entry = self._entry(name)
        return self._store.tensor(entry.gguf).ggml_type in _TERNARY_TYPES

    def _runs(self, entry: _Entry, begin: int, end: int):
        """Consecutive GGUF row runs covering HF rows [begin, end)."""
        if entry.row_map is None:
            yield begin, end
            return
        rows = entry.row_map[begin:end]
        start = 0
        for index in range(1, len(rows) + 1):
            if index == len(rows) or rows[index] != rows[index - 1] + 1:
                yield int(rows[start]), int(rows[index - 1]) + 1
                start = index

    def _raw_rows(self, entry: _Entry, begin: int, end: int) -> torch.Tensor:
        info = self._store.tensor(entry.gguf)
        k = entry.shape[1]
        pieces = []
        for low, high in self._runs(entry, begin, end):
            if info.ggml_type in _TERNARY_TYPES:
                pieces.append(dequantize_rows(self._store, entry.gguf, low, high))
                continue
            if info.ggml_type not in (_F32, _BF16):
                raise ValueError(f"{entry.gguf}: unsupported GGUF type {info.type_name}")
            word = 4 if info.ggml_type == _F32 else 2
            raw = self._store.read_tensor_raw(entry.gguf, low * k * word, high * k * word)
            dtype = torch.float32 if info.ggml_type == _F32 else torch.bfloat16
            pieces.append(torch.frombuffer(bytearray(raw), dtype=dtype).float().reshape(-1, k))
        self.bytes_read += (end - begin) * self._store.row_bytes(entry.gguf)
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces)

    def matrix_rows(self, name: str, begin: int, end: int) -> torch.Tensor:
        """HF-equivalent float32 rows [begin, end) of a matrix (primal basis)."""
        entry = self._entry(name)
        if entry.kind != "matrix" or not 0 <= begin <= end <= entry.shape[0]:
            raise ValueError(f"{name}: invalid matrix row range [{begin},{end})")
        if begin == end:
            return torch.empty((0, entry.shape[1]))
        rows = self._raw_rows(entry, begin, end)
        if entry.gguf in self.rotated or entry.gguf in self.inverse:
            rows = unrotate_rows(rows, self._signs(entry.shape[1]))
        return rows

    def _signs(self, width: int) -> torch.Tensor:
        try:
            return self.signs[width]
        except KeyError:
            raise ValueError(f"{self.path}: no Hadamard sign vector of width {width}") from None

    def encoded_rows(self, name: str, begin: int, end: int) -> EncodedRows:
        """Exact rotated `t2_g128_fp16` words for HF rows [begin, end)."""
        entry = self._entry(name)
        if not self.ternary(name):
            raise ValueError(f"{name}: {entry.gguf} is not a ternary tensor")
        if entry.gguf not in self.rotated and entry.gguf not in self.inverse:
            raise ValueError(
                f"{name}: {entry.gguf} is not Hadamard-folded; its ternary words are not "
                "a rotated weight"
            )
        if not 0 <= begin < end <= entry.shape[0]:
            raise ValueError(f"{name}: invalid encoded rows [{begin},{end})")
        codes, scales = [], []
        for low, high in self._runs(entry, begin, end):
            c, s = ternary_rows(self._store, entry.gguf, low, high)
            codes.append(c)
            scales.append(s)
        self.bytes_read += (end - begin) * self._store.row_bytes(entry.gguf)
        return EncodedRows(TERNARY_FORMAT, torch.cat(codes), torch.cat(scales))

    def vector_values(self, name: str) -> torch.Tensor:
        entry = self._entry(name)
        if entry.kind != "vector":
            raise ValueError(f"{name}: not a vector tensor")
        info = self._store.tensor(entry.gguf)
        if info.ggml_type != _F32:
            raise ValueError(f"{entry.gguf}: expected F32, got {info.type_name}")
        raw = np.frombuffer(self._store.read_tensor_raw(entry.gguf), dtype="<f4").astype(
            np.float64
        )
        values = np.asarray(entry.vector(raw), dtype=np.float64)
        if values.size != prod(entry.shape):
            raise ValueError(f"{name}: {entry.gguf} has {values.size} values, expected {entry.shape}")
        return torch.from_numpy(values.astype(np.float32).reshape(-1))

    def read_flat(self, name: str, begin: int = 0, end: int | None = None) -> torch.Tensor:
        entry = self._entry(name)
        elements = prod(entry.shape)
        end = elements if end is None else end
        if not 0 <= begin <= end <= elements:
            raise ValueError(f"{name}: element range [{begin},{end}) exceeds {entry.shape}")
        if entry.kind == "vector":
            return self.vector_values(name)[begin:end]
        k = entry.shape[1]
        first, last = begin // k, (end + k - 1) // k
        rows = self.matrix_rows(name, first, last)
        return rows.reshape(-1)[begin - first * k : end - first * k]

    def matrix_source(
        self, name: str, shape: tuple[int, int], format: str | None = None
    ) -> LogicalSource:
        """Matrix source with values and, for ternary tensors, exact encoded rows."""
        entry = self._entry(name)
        if tuple(shape) != entry.shape:
            raise ValueError(f"{name}: expected shape {entry.shape}, got {tuple(shape)}")
        ternary = self.ternary(name)
        if format not in (None, TERNARY_FORMAT) or (format and not ternary):
            raise ValueError(f"{name}: {entry.gguf} does not provide {format} rows")
        return LogicalSource(
            entry.shape,
            f"{self.path}:{entry.gguf} as {name}",
            lambda begin, end: self.read_flat(name, begin, end),
            (lambda begin, end: self.encoded_rows(name, begin, end)) if ternary else None,
        )

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> PrismCheckpoint:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


__all__ = ["PrismCheckpoint", "TERNARY_FORMAT", "grouped_head_sources"]
