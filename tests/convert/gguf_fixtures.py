"""Byte-level GGUF v3 fixture builder shared by the gguf_reader/prism_gguf tests.

Not a test module itself (pytest does not collect it): it is the inverse of
`tools/convert/sources/gguf_reader.py`'s parser, built independently so the parser tests
have a ground truth that does not reuse the parser's own encoding logic.
"""

from __future__ import annotations

import struct
from typing import Any, Sequence

_SCALAR_FORMATS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 10: "<Q", 11: "<q", 12: "<d",
}


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _value(type_code: int, value: Any) -> bytes:
    if type_code == 8:
        return _string(value)
    if type_code == 9:
        element_type, items = value
        out = struct.pack("<I", element_type) + struct.pack("<Q", len(items))
        for item in items:
            out += _value(element_type, item)
        return out
    if type_code == 7:
        return struct.pack("<B", 1 if value else 0)
    return struct.pack(_SCALAR_FORMATS[type_code], value)


def build_gguf(
    metadata: dict[str, tuple[int, Any]],
    tensors: Sequence[tuple[str, tuple[int, ...], int, bytes]],
    *,
    alignment: int = 32,
) -> bytes:
    """Encode a minimal GGUF v3 file.

    ``metadata`` maps key -> (gguf value type code, value). ``tensors`` is a sequence of
    ``(name, ne, ggml_type, raw_bytes)``; ``ne`` is ggml axis order (``ne[0]`` fastest).
    """
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(metadata))
    for key, (type_code, value) in metadata.items():
        header += _string(key) + struct.pack("<I", type_code) + _value(type_code, value)

    table = b""
    blobs: list[bytes] = []
    cursor = 0
    for name, ne, ggml_type, raw in tensors:
        aligned = (cursor + alignment - 1) // alignment * alignment
        table += (
            _string(name)
            + struct.pack("<I", len(ne))
            + b"".join(struct.pack("<Q", dim) for dim in ne)
            + struct.pack("<I", ggml_type)
            + struct.pack("<Q", aligned)
        )
        blobs.append(b"\x00" * (aligned - cursor) + raw)
        cursor = aligned + len(raw)

    body = header + table
    data_start = (len(body) + alignment - 1) // alignment * alignment
    body += b"\x00" * (data_start - len(body))
    body += b"".join(blobs)
    return body
