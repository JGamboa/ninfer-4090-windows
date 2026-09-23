#!/usr/bin/env python3
"""Dependency-free GGUF v3 store: header/metadata/tensor-table parsing, then bounded
tensor-data reads through :mod:`tools.artifact.os_compat`.

The header, key-value metadata and tensor table are small (kilobytes for this model) and
are read sequentially through an ordinary file object. Tensor *data* can be gigabytes
(the Bonsai checkpoint is 5.95 GB) and is never read in full: callers request byte or row
ranges through :meth:`GgufStore.read_tensor_raw` / :meth:`GgufStore.read_rows_raw`, which
go through a bounded ``pread`` exactly like ``tools/convert/sources/safetensors.py``.

Running this file directly keeps its original behavior: a dependency-free Markdown
inventory dump of a GGUF file's metadata and tensor table (used to produce
``docs/maintainer/bonsai-gguf-inventory.md``).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path
import struct
import sys
from typing import Any, BinaryIO

from tools.artifact import os_compat

GGUF_MAGIC = b"GGUF"
SUPPORTED_VERSION = 3

# ggml_type id -> short display name. Only used for the human-readable inventory; the
# byte layout used for bounded reads comes from _BLOCK_LAYOUT below, which is the table
# that actually matters for decoding. IDs 142 (PQ2_0) and 143 (PTQ1_0) are PrismML's own
# ggml_type extensions and are intentionally absent from upstream ggml's own registry.
GGUF_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
    29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4",
    142: "PQ2_0", 143: "PTQ1_0",
}

# ggml_type id -> (elements per block, bytes per block). Only the types this converter
# actually reads are registered; anything else raises rather than guessing a layout.
_BLOCK_LAYOUT = {
    0: (1, 4),      # F32
    1: (1, 2),      # F16
    30: (1, 2),     # BF16
    142: (128, 34),  # PQ2_0:  ggml_half d (2) + qs[32]
    143: (128, 28),  # PTQ1_0: qs[24] + qh[2] + ggml_half d (2)
}

_SCALAR_READERS = {
    0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32",
    6: "f32", 7: "boolean", 10: "u64", 11: "i64", 12: "f64",
}


def block_layout(ggml_type: int) -> tuple[int, int]:
    """Return ``(elements_per_block, bytes_per_block)`` for a supported ggml type id."""
    try:
        return _BLOCK_LAYOUT[ggml_type]
    except KeyError:
        raise ValueError(f"unsupported GGUF tensor type id {ggml_type}") from None


class _HeaderReader:
    """Sequential little-endian primitive reader over the header/table region only."""

    __slots__ = ("f",)

    def __init__(self, f: BinaryIO) -> None:
        self.f = f

    def u8(self) -> int:
        return struct.unpack("<B", self.f.read(1))[0]

    def i8(self) -> int:
        return struct.unpack("<b", self.f.read(1))[0]

    def u16(self) -> int:
        return struct.unpack("<H", self.f.read(2))[0]

    def i16(self) -> int:
        return struct.unpack("<h", self.f.read(2))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.f.read(4))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.f.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.f.read(8))[0]

    def i64(self) -> int:
        return struct.unpack("<q", self.f.read(8))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self.f.read(4))[0]

    def f64(self) -> float:
        return struct.unpack("<d", self.f.read(8))[0]

    def boolean(self) -> bool:
        return self.u8() != 0

    def string(self) -> str:
        n = self.u64()
        return self.f.read(n).decode("utf-8", errors="replace")

    def value(self, value_type: int) -> Any:
        if value_type == 8:
            return self.string()
        if value_type == 9:
            return self.array()
        try:
            reader_name = _SCALAR_READERS[value_type]
        except KeyError:
            raise ValueError(f"unsupported GGUF metadata value type {value_type}") from None
        return getattr(self, reader_name)()

    def array(self) -> list[Any]:
        element_type = self.u32()
        count = self.u64()
        return [self.value(element_type) for _ in range(count)]


@dataclass(frozen=True, slots=True)
class GgufTensorInfo:
    """One tensor-table entry. ``shape`` is ggml ``ne`` order: ``ne[0]`` is fastest-varying
    (the input/K axis for a 2-D weight); NInfer's own ``[N, K]`` order is the reverse."""

    name: str
    shape: tuple[int, ...]
    ggml_type: int
    type_name: str
    relative_offset: int
    elements: int


class GgufStore:
    """Parsed GGUF v3 structure plus bounded reads of tensor data.

    Opening a store reads only the header, metadata and tensor table (kilobytes); it never
    reads tensor payloads. Tensor data is read on demand through :meth:`read_tensor_raw` /
    :meth:`read_rows_raw`, each a single bounded ``pread`` per call, so the whole file is
    never held in memory at once.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.metadata: dict[str, Any] = {}
        self.tensors: dict[str, GgufTensorInfo] = {}
        self.tensor_order: list[str] = []
        self._fd = os.open(self.path, os.O_RDONLY | os_compat.BINARY_FLAG)
        try:
            with open(self.path, "rb") as f:
                r = _HeaderReader(f)
                magic = f.read(4)
                if magic != GGUF_MAGIC:
                    raise ValueError(f"{self.path}: not a GGUF file (bad magic {magic!r})")
                self.version = r.u32()
                if self.version != SUPPORTED_VERSION:
                    raise ValueError(
                        f"{self.path}: expected GGUF v{SUPPORTED_VERSION}, got v{self.version}"
                    )
                n_tensors = r.u64()
                n_kv = r.u64()
                for _ in range(n_kv):
                    key = r.string()
                    value_type = r.u32()
                    self.metadata[key] = r.value(value_type)
                raw_infos = []
                for _ in range(n_tensors):
                    name = r.string()
                    n_dims = r.u32()
                    shape = tuple(r.u64() for _ in range(n_dims))
                    ggml_type = r.u32()
                    offset = r.u64()
                    raw_infos.append((name, shape, ggml_type, offset))
                header_end = f.tell()
            alignment = int(self.metadata.get("general.alignment", 32))
            self.data_offset = (header_end + alignment - 1) // alignment * alignment
            self.file_bytes = os.fstat(self._fd).st_size
            for name, shape, ggml_type, offset in raw_infos:
                elements = 1
                for dim in shape:
                    elements *= dim
                info = GgufTensorInfo(
                    name, shape, ggml_type, GGUF_TYPES.get(ggml_type, f"type{ggml_type}"),
                    offset, elements,
                )
                if name in self.tensors:
                    raise ValueError(f"{self.path}: duplicate tensor name {name!r}")
                self.tensors[name] = info
                self.tensor_order.append(name)
        except BaseException:
            os.close(self._fd)
            raise

    def tensor(self, name: str) -> GgufTensorInfo:
        try:
            return self.tensors[name]
        except KeyError:
            raise KeyError(f"{self.path}: missing tensor {name!r}") from None

    def tensor_bytes(self, name: str) -> int:
        info = self.tensor(name)
        block_elements, block_bytes = block_layout(info.ggml_type)
        if info.elements % block_elements:
            raise ValueError(
                f"{name}: {info.elements} elements is not a multiple of the "
                f"{info.type_name} block size {block_elements}"
            )
        return info.elements // block_elements * block_bytes

    def row_bytes(self, name: str) -> int:
        """Bytes per row for a 2-D tensor, over the ``ne[0]`` (K) axis."""
        info = self.tensor(name)
        if len(info.shape) != 2:
            raise ValueError(f"{name}: row access requires a 2-D tensor, got {info.shape}")
        k = info.shape[0]
        block_elements, block_bytes = block_layout(info.ggml_type)
        if k % block_elements:
            raise ValueError(
                f"{name}: K={k} is not a multiple of the {info.type_name} "
                f"block size {block_elements}"
            )
        return k // block_elements * block_bytes

    def read_tensor_raw(self, name: str, byte_begin: int = 0, byte_end: int | None = None) -> bytes:
        """Bounded raw-byte read of one tensor's data, ``[byte_begin, byte_end)``."""
        total = self.tensor_bytes(name)
        end = total if byte_end is None else byte_end
        if not 0 <= byte_begin <= end <= total:
            raise ValueError(f"{name}: byte range [{byte_begin},{end}) exceeds {total}")
        if byte_begin == end:
            return b""
        info = self.tensors[name]
        absolute = self.data_offset + info.relative_offset + byte_begin
        data = os_compat.pread(self._fd, end - byte_begin, absolute)
        if len(data) != end - byte_begin:
            raise ValueError(f"{name}: short read at file offset {absolute}")
        return data

    def read_rows_raw(self, name: str, row_begin: int, row_end: int) -> bytes:
        """Bounded raw-byte read of a contiguous row range of a 2-D block-quantized tensor."""
        info = self.tensor(name)
        n = info.shape[1]
        if not 0 <= row_begin <= row_end <= n:
            raise ValueError(f"{name}: row range [{row_begin},{row_end}) exceeds {n} rows")
        row_bytes = self.row_bytes(name)
        return self.read_tensor_raw(name, row_begin * row_bytes, row_end * row_bytes)

    def close(self) -> None:
        os.close(self._fd)

    def __enter__(self) -> "GgufStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def main(path: str) -> None:
    """Dependency-free Markdown inventory dump (metadata + tensor table)."""
    with GgufStore(path) as store:
        print(f"# GGUF v{store.version}: {len(store.tensor_order)} tensors, {len(store.metadata)} metadata keys\n")
        print("## Metadata")
        for key, val in store.metadata.items():
            if isinstance(val, list):
                head = val[:8]
                summary = f"array[{len(val)}] head={head}"
                if key.startswith("prism") or "hadamard" in key or "sign" in key:
                    counter = Counter(val) if len(val) < 100000 else None
                    summary += f" counter={dict(counter) if counter and len(counter) < 8 else 'n/a'}"
                if key.startswith("tokenizer"):
                    summary = f"array[{len(val)}] (tokenizer, omitted)"
                print(f"- `{key}`: {summary}")
            else:
                s = str(val)
                if len(s) > 200:
                    s = s[:200] + "..."
                print(f"- `{key}`: {s}")
        print("\n## Tensors")
        print("| name | shape (ne) | type | offset |")
        print("|---|---|---|---|")
        types: Counter = Counter()
        for name in store.tensor_order:
            info = store.tensors[name]
            types[info.type_name] += 1
            print(f"| {name} | {list(info.shape)} | {info.type_name} | {info.relative_offset} |")
        print("\n## Type histogram")
        for type_name, count in types.most_common():
            print(f"- {type_name}: {count}")


if __name__ == "__main__":
    main(sys.argv[1])
