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

T2, T5 = "t2_g128_fp16", "t5_g128_fp16"


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
    packed = pack_ternary_codes(logical, T2)
    assert packed[0, 0].item() == 0b01_10_01_00 and packed[1, 0].item() == 0b10101010
    scales = torch.tensor([[0.5], [-2.0]], dtype=torch.float16)
    payload = encode_ternary(packed, scales, shape, T2)

    assert payload[:64] == packed.numpy().tobytes()
    assert payload[64:256] == bytes(192)
    assert payload[256:] == struct.pack("<ee", 0.5, -2.0)

    codes, decoded_scales = decode_ternary_words(payload, shape, T2)
    assert torch.equal(codes, packed) and torch.equal(decoded_scales, scales)
    expected = (logical.float() - 1.0) * torch.tensor([[0.5], [-2.0]])
    assert torch.equal(dequantize_ternary(payload, shape, format=T2), expected)


def test_ternary_slot_order_matches_the_pq2_0_definition():
    generator = torch.Generator().manual_seed(7)
    logical = torch.randint(0, 3, (3, 256), dtype=torch.uint8, generator=generator)
    packed = pack_ternary_codes(logical, T2)
    for row in range(3):
        for j in range(256):
            code = (packed[row, j // 4].item() >> ((j % 4) * 2)) & 3
            assert code == logical[row, j].item()
    assert torch.equal(unpack_ternary_codes(packed, T2), logical)


@pytest.mark.parametrize("format", [T2, T5])
def test_ternary_group_scales_cover_consecutive_128_columns(format):
    shape = (1, 256)
    logical = torch.full((1, 256), 2, dtype=torch.uint8)
    scales = torch.tensor([[1.0, 3.0]], dtype=torch.float16)
    values = dequantize_ternary(
        encode_ternary(pack_ternary_codes(logical, format), scales, shape, format),
        shape,
        format=format,
    )
    assert torch.equal(values[0, :128], torch.ones(128))
    assert torch.equal(values[0, 128:], torch.full((128,), 3.0))


def test_ternary_rejects_invalid_codes_scales_and_shapes():
    shape = (1, 128)
    scales = torch.ones((1, 1), dtype=torch.float16)
    invalid = torch.full((1, 32), 0xFF, dtype=torch.uint8)
    with pytest.raises(ValueError, match="invalid slot value 3"):
        encode_ternary(invalid, scales, shape, T2)
    with pytest.raises(ValueError, match="0, 1 or 2"):
        pack_ternary_codes(torch.full((1, 4), 3, dtype=torch.uint8), T2)
    codes = torch.zeros((1, 32), dtype=torch.uint8)
    with pytest.raises(ValueError, match="finite"):
        encode_ternary(codes, torch.full((1, 1), float("inf"), dtype=torch.float16), shape, T2)
    with pytest.raises(TypeError, match="float16"):
        encode_ternary(codes, scales.float(), shape, T2)
    with pytest.raises(ValueError, match="divisible by 128"):
        encoded_size("ternary_row_k128_v1", "t2_g128_fp16", (1, 64))
    with pytest.raises(ValueError, match="does not accept"):
        encoded_size("row_split_k128_v1", "t2_g128_fp16", (1, 128))


def _literal_t5_bytes(codes):
    """The base-3 layout transcribed from its definition (design doc 9.1), one byte at a time."""
    rows, k = codes.shape
    out = []
    for row in range(rows):
        line = []
        for unit in range(k // 64):
            c = [int(x) for x in codes[row, 64 * unit : 64 * unit + 64]]
            for i in range(13):
                if i < 12:
                    g, j = divmod(i, 4)
                    trits = [c[20 * g + 4 * m + j] for m in range(5)]
                else:
                    trits = [c[60], c[61], c[62], c[63], 0]
                v = 0
                for t in trits:  # t_0 is the most significant trit
                    v = 3 * v + t
                line.append(-(-256 * v // 243))  # ceil(256 v / 243)
        out.append(line)
    return torch.tensor(out, dtype=torch.uint8)


def test_t5_layout_matches_its_literal_definition_and_round_trips():
    shape = (3, 1024)
    geometry = ternary_geometry(T5, shape)
    assert (geometry.code_row_bytes, geometry.code_plane_bytes, geometry.scale_plane_offset) == (
        208,
        624,
        768,
    )
    assert encoded_size("ternary_row_k128_v1", T5, shape) == 768 + 3 * 8 * 2
    generator = torch.Generator().manual_seed(11)
    logical = torch.randint(0, 3, shape, dtype=torch.uint8, generator=generator)
    packed = pack_ternary_codes(logical, T5)
    assert torch.equal(packed, _literal_t5_bytes(logical))
    assert torch.equal(unpack_ternary_codes(packed, T5), logical)
    scales = torch.rand((3, 8), generator=generator).to(torch.float16)
    payload = encode_ternary(packed, scales, shape, T5)
    codes, decoded = decode_ternary_words(payload, shape, T5)
    assert torch.equal(codes, packed) and torch.equal(decoded, scales)
    expected = (logical.float() - 1.0) * scales.float().repeat_interleave(128, dim=1)
    assert torch.equal(dequantize_ternary(payload, shape, format=T5), expected)


def test_t5_decode_is_exact_for_every_byte_value():
    # All 243 values of five trits: the decode recovers each trit from the scaled byte.
    trits = torch.tensor(
        [[(v // 3 ** (4 - m)) % 3 for m in range(5)] for v in range(243)], dtype=torch.uint8
    )
    logical = torch.zeros((243, 64), dtype=torch.uint8)
    for m in range(5):
        logical[:, 4 * m] = trits[:, m]  # byte 0 (g = 0, j = 0) holds columns 4 m
    packed = pack_ternary_codes(logical, T5)
    assert torch.equal(unpack_ternary_codes(packed, T5), logical)
    assert len(set(packed[:, 0].tolist())) == 243


def test_t5_rejects_non_canonical_bytes_and_padding():
    shape = (1, 128)
    scales = torch.ones((1, 1), dtype=torch.float16)
    packed = pack_ternary_codes(torch.ones(shape, dtype=torch.uint8), T5)
    canonical = {-(-256 * v // 243) for v in range(243)}
    bad = packed.clone()
    bad[0, 0] = min(set(range(256)) - canonical)  # not ceil(256 v / 243) of any v
    with pytest.raises(ValueError, match="canonical"):
        encode_ternary(bad, scales, shape, T5)
    padded = packed.clone()
    padded[0, 12] = -(-256 * 1 // 243)  # byte 12 with its fifth trit set to 1
    with pytest.raises(ValueError, match="canonical|fifth trit"):
        encode_ternary(padded, scales, shape, T5)
