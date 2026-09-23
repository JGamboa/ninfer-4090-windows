from __future__ import annotations

import struct

import pytest
import torch

from tools.artifact.codecs.ternary import (
    decode_ternary_words,
    dequantize_ternary,
    encode_ternary,
    pack_ternary_codes,
    unpack_ternary_codes,
)
from tools.artifact.layouts import encoded_size, ternary_geometry


def test_ternary_layout_known_words_padding_and_reconstruction():
    shape = (2, 128)
    geometry = ternary_geometry("t2_g128_fp16", shape)
    assert (
        geometry.code_row_bytes,
        geometry.code_plane_bytes,
        geometry.scale_plane_offset,
        geometry.scale_plane_bytes,
        geometry.payload_bytes,
    ) == (32, 64, 256, 4, 260)
    assert encoded_size("ternary_row_k128_v1", "t2_g128_fp16", shape) == 260

    # Row 0 cycles -1, 0, +1, 0; row 1 is all +1.
    logical = torch.tensor([[0, 1, 2, 1] * 32, [2] * 128], dtype=torch.uint8)
    packed = pack_ternary_codes(logical)
    assert packed[0, 0].item() == 0b01_10_01_00 and packed[1, 0].item() == 0b10101010
    scales = torch.tensor([[0.5], [-2.0]], dtype=torch.float16)
    payload = encode_ternary(packed, scales, shape)

    assert payload[:64] == packed.numpy().tobytes()
    assert payload[64:256] == bytes(192)
    assert payload[256:] == struct.pack("<ee", 0.5, -2.0)

    codes, decoded_scales = decode_ternary_words(payload, shape)
    assert torch.equal(codes, packed) and torch.equal(decoded_scales, scales)
    expected = (logical.float() - 1.0) * torch.tensor([[0.5], [-2.0]])
    assert torch.equal(dequantize_ternary(payload, shape), expected)


def test_ternary_slot_order_matches_the_pq2_0_definition():
    generator = torch.Generator().manual_seed(7)
    logical = torch.randint(0, 3, (3, 256), dtype=torch.uint8, generator=generator)
    packed = pack_ternary_codes(logical)
    for row in range(3):
        for j in range(256):
            code = (packed[row, j // 4].item() >> ((j % 4) * 2)) & 3
            assert code == logical[row, j].item()
    assert torch.equal(unpack_ternary_codes(packed), logical)


def test_ternary_group_scales_cover_consecutive_128_columns():
    shape = (1, 256)
    logical = torch.full((1, 256), 2, dtype=torch.uint8)
    scales = torch.tensor([[1.0, 3.0]], dtype=torch.float16)
    values = dequantize_ternary(
        encode_ternary(pack_ternary_codes(logical), scales, shape), shape
    )
    assert torch.equal(values[0, :128], torch.ones(128))
    assert torch.equal(values[0, 128:], torch.full((128,), 3.0))


def test_ternary_rejects_invalid_codes_scales_and_shapes():
    shape = (1, 128)
    scales = torch.ones((1, 1), dtype=torch.float16)
    invalid = torch.full((1, 32), 0xFF, dtype=torch.uint8)
    with pytest.raises(ValueError, match="invalid slot value 3"):
        encode_ternary(invalid, scales, shape)
    with pytest.raises(ValueError, match="0, 1 or 2"):
        pack_ternary_codes(torch.full((1, 4), 3, dtype=torch.uint8))
    codes = torch.zeros((1, 32), dtype=torch.uint8)
    with pytest.raises(ValueError, match="finite"):
        encode_ternary(codes, torch.full((1, 1), float("inf"), dtype=torch.float16), shape)
    with pytest.raises(TypeError, match="float16"):
        encode_ternary(codes, scales.float(), shape)
    with pytest.raises(ValueError, match="divisible by 128"):
        encoded_size("ternary_row_k128_v1", "t2_g128_fp16", (1, 64))
    with pytest.raises(ValueError, match="does not accept"):
        encoded_size("row_split_k128_v1", "t2_g128_fp16", (1, 128))
