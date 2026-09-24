"""A llama.cpp `qwen3vl_merger` mmproj GGUF presented as the HF Qwen3.5 vision checkpoint.

The Qwen3.5 builder addresses the Vision tower by HF name (`model.visual.*`). This store
answers those names from the mmproj written by llama.cpp's `Qwen3VLVisionModel`
(`conversion/qwen3vl.py` in PrismML-Eng/llama.cpp, branch `prism`), whose only transforms
are renames and one split:

- `visual.patch_embed.proj.weight` [h, 3, 2, p, p] is stored as two Conv2D halves along the
  temporal axis, `v.patch_embd.weight` (t = 0) and `v.patch_embd.weight.1` (t = 1);
- `visual.merger.linear_fc1` / `linear_fc2` / `norm` are `mm.0` / `mm.2` / `v.post_ln`;
- blocks: `norm1` / `norm2` / `attn.qkv` / `attn.proj` / `mlp.linear_fc1` / `mlp.linear_fc2`
  are `v.blk.N.ln1` / `ln2` / `attn_qkv` / `attn_out` / `ffn_up` / `ffn_down`;
- `visual.pos_embed.weight` is `v.position_embd.weight`.

Tensors may be F32, F16, BF16 or Q8_0 (Prism's Q8_0 pack keeps `ffn_down`, whose 4304-wide
rows are not a multiple of 32, in F16). GGUF shapes are ggml `ne` order, the reverse of the
torch shape. `config` is `{"vision_config": ...}` derived from the `clip.vision.*` metadata.
"""

from __future__ import annotations

from math import prod
from pathlib import Path

import numpy as np
import torch

from .gguf_reader import GgufStore
from .logical import LogicalSource
from .safetensors import TensorInfo

_F32, _F16, _Q8_0, _BF16 = 0, 1, 8, 30


def dequantize(store: GgufStore, name: str) -> torch.Tensor:
    """Float32 values of a whole F32/F16/BF16/Q8_0 tensor in torch (reversed `ne`) shape."""
    info = store.tensor(name)
    raw = store.read_tensor_raw(name)
    count = prod(info.shape)
    if info.ggml_type == _F32:
        values = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    elif info.ggml_type == _F16:
        values = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    elif info.ggml_type == _BF16:
        bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        values = bits.view(np.float32)
    elif info.ggml_type == _Q8_0:
        blocks = np.frombuffer(raw, dtype=np.dtype([("d", "<f2"), ("q", "i1", 32)]))
        values = (blocks["q"].astype(np.float32) * blocks["d"].astype(np.float32)[:, None])
        values = values.reshape(-1)
    else:
        raise ValueError(f"{name}: unsupported mmproj tensor type {info.type_name}")
    if values.size != count:
        raise ValueError(f"{name}: {values.size} values for shape {info.shape}")
    return torch.from_numpy(values.copy()).reshape(tuple(reversed(info.shape)))


class MmprojCheckpoint:
    """Read-only `model.visual.*` view over a Qwen3-VL mmproj GGUF."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.bytes_read = 0
        self._store = GgufStore(self.path)
        try:
            self.config = {"vision_config": self._vision_config()}
            self._names = self._map()
        except BaseException:
            self._store.close()
            raise
        self._cached: tuple[str, torch.Tensor] | None = None

    @property
    def metadata(self) -> dict:
        return self._store.metadata

    def _meta(self, key: str) -> int:
        value = self._store.metadata.get("clip.vision." + key)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{self.path}: missing positive clip.vision.{key}")
        return value

    def _vision_config(self) -> dict:
        meta = self._store.metadata
        if meta.get("general.type") != "mmproj" or not meta.get("clip.has_vision_encoder"):
            raise ValueError(f"{self.path}: not a vision mmproj GGUF")
        if meta.get("clip.projector_type") != "qwen3vl_merger":
            raise ValueError(
                f"{self.path}: projector {meta.get('clip.projector_type')!r} is not qwen3vl_merger"
            )
        if any(meta.get("clip.vision.is_deepstack_layers", ())):
            raise ValueError(f"{self.path}: deepstack layers are not part of Qwen3.5 Vision")
        if not meta.get("clip.use_gelu", False):
            raise ValueError(f"{self.path}: expected clip.use_gelu (gelu_pytorch_tanh)")
        h = self._meta("embedding_length")
        positions = self._store.tensor("v.position_embd.weight").shape
        if len(positions) != 2 or positions[0] != h:
            raise ValueError(f"{self.path}: v.position_embd.weight has shape {positions}")
        temporal = 2 if "v.patch_embd.weight.1" in self._store.tensors else 1
        return {
            "depth": self._meta("block_count"),
            "hidden_size": h,
            "intermediate_size": self._meta("feed_forward_length"),
            "num_heads": self._meta("attention.head_count"),
            "patch_size": self._meta("patch_size"),
            "temporal_patch_size": temporal,
            "spatial_merge_size": self._meta("spatial_merge_size"),
            "num_position_embeddings": positions[1],
            "out_hidden_size": self._meta("projection_dim"),
            "in_channels": 3,
            "hidden_act": "gelu_pytorch_tanh",
            "deepstack_visual_indexes": [],
        }

    def _map(self) -> dict[str, str]:
        names = {
            "pos_embed.weight": "v.position_embd.weight",
            "patch_embed.proj.bias": "v.patch_embd.bias",
            "merger.norm.weight": "v.post_ln.weight",
            "merger.norm.bias": "v.post_ln.bias",
            "merger.linear_fc1.weight": "mm.0.weight",
            "merger.linear_fc1.bias": "mm.0.bias",
            "merger.linear_fc2.weight": "mm.2.weight",
            "merger.linear_fc2.bias": "mm.2.bias",
        }
        roles = {
            "norm1": "ln1",
            "norm2": "ln2",
            "attn.qkv": "attn_qkv",
            "attn.proj": "attn_out",
            "mlp.linear_fc1": "ffn_up",
            "mlp.linear_fc2": "ffn_down",
        }
        for i in range(self.config["vision_config"]["depth"]):
            for hf, gguf in roles.items():
                for kind in ("weight", "bias"):
                    names[f"blocks.{i}.{hf}.{kind}"] = f"v.blk.{i}.{gguf}.{kind}"
        missing = [g for g in names.values() if g not in self._store.tensors]
        if missing or "v.patch_embd.weight" not in self._store.tensors:
            raise ValueError(f"{self.path}: missing mmproj tensors {missing[:4]}")
        return names

    def _key(self, name: str) -> str:
        for prefix in ("model.visual.", "visual."):
            if name.startswith(prefix):
                return name[len(prefix) :]
        raise ValueError(f"{self.path}: {name!r} is not a Vision tensor")

    def _shape(self, key: str) -> tuple[int, ...]:
        if key == "patch_embed.proj.weight":
            half = tuple(reversed(self._store.tensor("v.patch_embd.weight").shape))
            t = self.config["vision_config"]["temporal_patch_size"]
            return (half[0], half[1], t, half[2], half[3])
        return tuple(reversed(self._store.tensor(self._names[key]).shape))

    def has(self, name: str) -> bool:
        try:
            key = self._key(name)
        except ValueError:
            return False
        return key == "patch_embed.proj.weight" or key in self._names

    def describe(self, name: str) -> TensorInfo:
        key = self._key(name)
        if key != "patch_embed.proj.weight" and key not in self._names:
            raise ValueError(f"{self.path}: no mmproj tensor for {name!r}")
        shape = self._shape(key)
        return TensorInfo(self.path, shape, "F32", 0, prod(shape) * 4)

    def values(self, name: str) -> torch.Tensor:
        """Float32 HF-shaped values (the last tensor read is cached)."""
        key = self._key(name)
        if self._cached is not None and self._cached[0] == key:
            return self._cached[1]
        if key == "patch_embed.proj.weight":
            halves = [dequantize(self._store, "v.patch_embd.weight")]
            if self.config["vision_config"]["temporal_patch_size"] == 2:
                halves.append(dequantize(self._store, "v.patch_embd.weight.1"))
            tensor = torch.stack(halves, dim=2).contiguous()
            self.bytes_read += sum(
                self._store.tensor_bytes(n)
                for n in ("v.patch_embd.weight", "v.patch_embd.weight.1")
                if n in self._store.tensors
            )
        else:
            gguf = self._names[key]
            tensor = dequantize(self._store, gguf)
            self.bytes_read += self._store.tensor_bytes(gguf)
        self._cached = (key, tensor)
        return tensor

    def read_flat(self, name: str, begin: int = 0, end: int | None = None) -> torch.Tensor:
        flat = self.values(name).reshape(-1)
        end = flat.numel() if end is None else end
        if not 0 <= begin <= end <= flat.numel():
            raise ValueError(f"{name}: element range [{begin},{end}) exceeds {flat.numel()}")
        return flat[begin:end]

    def matrix_source(
        self, name: str, shape: tuple[int, int], format: str | None = None
    ) -> LogicalSource:
        if format is not None:
            raise ValueError(f"{name}: an mmproj provides values, not {format} rows")
        actual = self._shape(self._key(name))
        if tuple(shape) != actual:
            raise ValueError(f"{name}: expected shape {actual}, got {tuple(shape)}")
        return LogicalSource(
            actual,
            f"{self.path}:{name}",
            lambda begin, end: self.read_flat(name, begin, end),
        )

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> MmprojCheckpoint:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def is_mmproj(path: str | Path) -> bool:
    """True when a GGUF declares `general.type = mmproj`."""
    with GgufStore(path) as store:
        return store.metadata.get("general.type") == "mmproj"


__all__ = ["MmprojCheckpoint", "dequantize", "is_mmproj"]
