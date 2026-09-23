"""Exact 2-bit ternary codes and FP16 group scales in ternary_row_k128_v1 layout."""

from __future__ import annotations

from typing import Sequence

import torch

from ..layouts import ternary_geometry
from ._tensor_bytes import Payload, _payload_length, _payload_tensor

_SHIFTS = (0, 2, 4, 6)


def pack_ternary_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack uint8 codes in {0, 1, 2} of shape [rows, K] into [rows, K / 4] bytes.

    Weight ``k`` occupies bits ``2 * (k % 4)`` of byte ``k // 4`` (PQ2_0 order).
    """
    if codes.dtype != torch.uint8 or codes.dim() != 2 or codes.shape[1] % 4:
        raise TypeError("ternary codes must be uint8 [rows, K] with K divisible by 4")
    if bool((codes > 2).any()):
        raise ValueError("ternary codes must be 0, 1 or 2")
    quads = codes.reshape(codes.shape[0], -1, 4).to(torch.int32)
    packed = quads[..., 0] | (quads[..., 1] << 2) | (quads[..., 2] << 4) | (quads[..., 3] << 6)
    return packed.to(torch.uint8)


def unpack_ternary_codes(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_ternary_codes`; returns uint8 [rows, 4 * bytes]."""
    if packed.dtype != torch.uint8 or packed.dim() != 2:
        raise TypeError("packed ternary codes must be a uint8 matrix")
    words = packed.to(torch.int32)
    slots = [(words >> shift) & 3 for shift in _SHIFTS]
    return torch.stack(slots, dim=-1).reshape(packed.shape[0], -1).to(torch.uint8)


def validate_ternary_words(codes: torch.Tensor, scales: torch.Tensor) -> None:
    """Reject invalid slot code 3 and non-finite FP16 scales."""
    words = codes.to(torch.int32)
    if bool(torch.stack([((words >> s) & 3) == 3 for s in _SHIFTS]).any()):
        raise ValueError("ternary codes must not contain the invalid slot value 3")
    if not bool(torch.isfinite(scales.float()).all()):
        raise ValueError("ternary scales must be finite FP16 values")


def _exact(tensor: torch.Tensor, dtype: torch.dtype, shape: tuple, label: str):
    if tensor.dtype != dtype or tuple(tensor.shape) != shape:
        raise TypeError(f"{label} must be {dtype} with shape {shape}")
    return tensor.detach().contiguous().cpu()


def encode_ternary(
    codes: torch.Tensor, scales: torch.Tensor, shape: Sequence[int]
) -> bytes:
    """Encode packed codes uint8 [N, K/4] and scales float16 [N, K/128]."""
    geometry = ternary_geometry("t2_g128_fp16", shape)
    codes = _exact(
        codes, torch.uint8, (geometry.n, geometry.code_row_bytes), "ternary codes"
    )
    scales = _exact(
        scales, torch.float16, (geometry.n, geometry.groups_per_row), "ternary scales"
    )
    validate_ternary_words(codes, scales)
    payload = bytearray(geometry.payload_bytes)
    payload[: geometry.code_plane_bytes] = codes.numpy().tobytes()
    begin = geometry.scale_plane_offset
    payload[begin : begin + geometry.scale_plane_bytes] = scales.numpy().tobytes()
    return bytes(payload)


def decode_ternary_words(
    payload: Payload, shape: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact packed codes uint8 [N, K/4] and scales float16 [N, K/128]."""
    geometry = ternary_geometry("t2_g128_fp16", shape)
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
    validate_ternary_words(codes, scales)
    return codes, scales


def dequantize_ternary_words(
    codes: torch.Tensor, scales: torch.Tensor, group_size: int = 128
) -> torch.Tensor:
    """Reconstruct float32 ``(code - 1) * scale`` from packed codes and group scales."""
    values = unpack_ternary_codes(codes).to(torch.float32) - 1.0
    return values * scales.float().repeat_interleave(group_size, dim=1)


def dequantize_ternary(
    payload: Payload, shape: Sequence[int], dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    codes, scales = decode_ternary_words(payload, shape)
    return dequantize_ternary_words(codes, scales).to(dtype)
