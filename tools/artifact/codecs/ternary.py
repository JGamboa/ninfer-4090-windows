"""Exact ternary codes and FP16 group scales in ternary_row_k128_v1 layout.

Codes are uint8 in {0, 1, 2} (weight ``c - 1``); a format's ``packing`` defines the code
bytes (`tools/artifact/formats.py` `TernaryFormat`): ``slot2`` for `t2_g128_fp16`, scaled
base 3 in 13-byte units of 64 columns for `t5_g128_fp16`.
"""

from __future__ import annotations

from typing import Sequence

import torch

from ..formats import TernaryFormat, get_format
from ..layouts import ternary_geometry
from ._tensor_bytes import Payload, _payload_length, _payload_tensor

_SHIFTS = (0, 2, 4, 6)


def _ternary(format: str | TernaryFormat) -> TernaryFormat:
    spec = get_format(format) if isinstance(format, str) else format
    if not isinstance(spec, TernaryFormat):
        raise ValueError(f"{spec.name} is not a ternary format")
    return spec


def _base3_columns() -> torch.Tensor:
    """Unit column of trit m of byte i ([13, 5]); 64 marks byte 12's always-zero trit 4."""
    table = torch.empty((13, 5), dtype=torch.long)
    for i in range(12):
        g, j = divmod(i, 4)
        for m in range(5):
            table[i, m] = 20 * g + 4 * m + j
    table[12] = torch.tensor([60, 61, 62, 63, 64])
    return table


_BASE3_COLUMNS = _base3_columns()
_BASE3_WEIGHTS = torch.tensor([81, 27, 9, 3, 1], dtype=torch.int32)


def _check_codes(codes: torch.Tensor, multiple: int) -> None:
    if codes.dtype != torch.uint8 or codes.dim() != 2 or codes.shape[1] % multiple:
        raise TypeError(f"ternary codes must be uint8 [rows, K] with K divisible by {multiple}")
    if bool((codes > 2).any()):
        raise ValueError("ternary codes must be 0, 1 or 2")


def pack_ternary_codes(
    codes: torch.Tensor, format: str | TernaryFormat
) -> torch.Tensor:
    """Pack uint8 codes in {0, 1, 2} [rows, K] into the format's code bytes."""
    spec = _ternary(format)
    if spec.packing == "slot2":
        # Weight k occupies bits 2 * (k % 4) of byte k // 4 (PQ2_0 order).
        _check_codes(codes, 4)
        quads = codes.reshape(codes.shape[0], -1, 4).to(torch.int32)
        packed = quads[..., 0] | (quads[..., 1] << 2) | (quads[..., 2] << 4) | (quads[..., 3] << 6)
        return packed.to(torch.uint8)
    _check_codes(codes, 64)
    rows = codes.shape[0]
    units = codes.reshape(rows, -1, 64).to(torch.int32)
    padded = torch.cat((units, torch.zeros_like(units[..., :1])), dim=-1)
    trits = padded[..., _BASE3_COLUMNS]  # [rows, U, 13, 5]
    value = (trits * _BASE3_WEIGHTS).sum(dim=-1)
    return ((256 * value + 242) // 243).to(torch.uint8).reshape(rows, -1)


def _base3_trits(packed: torch.Tensor) -> torch.Tensor:
    """Trits [rows, U, 13, 5] of scaled base-3 bytes [rows, 13 U]."""
    r = packed.to(torch.int32).reshape(packed.shape[0], -1, 13)
    trits = []
    for _ in range(5):
        r = r * 3
        trits.append(r >> 8)
        r = r & 255
    return torch.stack(trits, dim=-1)


def unpack_ternary_codes(
    packed: torch.Tensor, format: str | TernaryFormat
) -> torch.Tensor:
    """Inverse of :func:`pack_ternary_codes`; returns uint8 codes [rows, K]."""
    spec = _ternary(format)
    if packed.dtype != torch.uint8 or packed.dim() != 2:
        raise TypeError("packed ternary codes must be a uint8 matrix")
    if spec.packing == "slot2":
        words = packed.to(torch.int32)
        slots = [(words >> shift) & 3 for shift in _SHIFTS]
        return torch.stack(slots, dim=-1).reshape(packed.shape[0], -1).to(torch.uint8)
    if packed.shape[1] % 13:
        raise TypeError("base-3 ternary rows must be whole 13-byte units")
    trits = _base3_trits(packed)
    rows, units = trits.shape[0], trits.shape[1]
    out = torch.zeros((rows, units, 65), dtype=torch.int32)
    out.scatter_(
        2,
        _BASE3_COLUMNS.reshape(1, 1, 65).expand(rows, units, 65),
        trits.reshape(rows, units, 65),
    )
    return out[..., :64].reshape(rows, -1).to(torch.uint8)


def validate_ternary_words(
    codes: torch.Tensor, scales: torch.Tensor, format: str | TernaryFormat
) -> None:
    """Reject invalid code bytes (slot 3; a base-3 byte outside the 243 encodings or a
    nonzero padding trit) and non-finite FP16 scales."""
    spec = _ternary(format)
    if spec.packing == "slot2":
        words = codes.to(torch.int32)
        if bool(torch.stack([((words >> s) & 3) == 3 for s in _SHIFTS]).any()):
            raise ValueError("ternary codes must not contain the invalid slot value 3")
    else:
        trits = _base3_trits(codes)
        value = (trits * _BASE3_WEIGHTS).sum(dim=-1)
        canonical = (256 * value + 242) // 243
        if bool((canonical != codes.to(torch.int32).reshape(value.shape)).any()):
            raise ValueError("base-3 ternary bytes must be canonical encodings")
        if bool((trits[..., 12, 4] != 0).any()):
            raise ValueError("base-3 ternary byte 12 must have a zero fifth trit")
    if not bool(torch.isfinite(scales.float()).all()):
        raise ValueError("ternary scales must be finite FP16 values")


def _exact(tensor: torch.Tensor, dtype: torch.dtype, shape: tuple, label: str):
    if tensor.dtype != dtype or tuple(tensor.shape) != shape:
        raise TypeError(f"{label} must be {dtype} with shape {shape}")
    return tensor.detach().contiguous().cpu()


def encode_ternary(
    codes: torch.Tensor,
    scales: torch.Tensor,
    shape: Sequence[int],
    format: str | TernaryFormat,
) -> bytes:
    """Encode packed code bytes [N, code_row_bytes] and scales float16 [N, K/128]."""
    geometry = ternary_geometry(format, shape)
    codes = _exact(
        codes, torch.uint8, (geometry.n, geometry.code_row_bytes), "ternary codes"
    )
    scales = _exact(
        scales, torch.float16, (geometry.n, geometry.groups_per_row), "ternary scales"
    )
    validate_ternary_words(codes, scales, format)
    payload = bytearray(geometry.payload_bytes)
    payload[: geometry.code_plane_bytes] = codes.numpy().tobytes()
    begin = geometry.scale_plane_offset
    payload[begin : begin + geometry.scale_plane_bytes] = scales.numpy().tobytes()
    return bytes(payload)


def decode_ternary_words(
    payload: Payload, shape: Sequence[int], format: str | TernaryFormat
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact packed code bytes [N, code_row_bytes] and scales float16 [N, K/128]."""
    geometry = ternary_geometry(format, shape)
    if _payload_length(payload) != geometry.payload_bytes:
        raise ValueError(
            f"ternary payload has {_payload_length(payload)} bytes, "
            f"expected {geometry.payload_bytes}"
        )
    raw = _payload_tensor(payload, torch.device("cpu"))
    codes = raw[: geometry.code_plane_bytes].clone().reshape(
        geometry.n, geometry.code_row_bytes
    )
    begin = geometry.scale_plane_offset
    scales = (
        raw[begin : begin + geometry.scale_plane_bytes]
        .clone()
        .view(torch.float16)
        .reshape(geometry.n, geometry.groups_per_row)
    )
    validate_ternary_words(codes, scales, format)
    return codes, scales


def dequantize_ternary_words(
    codes: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
    *,
    format: str | TernaryFormat,
) -> torch.Tensor:
    """Reconstruct float32 ``(code - 1) * scale`` from packed codes and group scales."""
    values = unpack_ternary_codes(codes, format).to(torch.float32) - 1.0
    return values * scales.float().repeat_interleave(group_size, dim=1)


def dequantize_ternary(
    payload: Payload,
    shape: Sequence[int],
    dtype: torch.dtype = torch.float32,
    *,
    format: str | TernaryFormat,
) -> torch.Tensor:
    codes, scales = decode_ternary_words(payload, shape, format)
    return dequantize_ternary_words(codes, scales, format=format).to(dtype)
