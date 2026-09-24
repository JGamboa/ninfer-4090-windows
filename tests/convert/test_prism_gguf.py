"""Independent scalar oracle for the PTQ1_0/PQ2_0 decoders.

Each oracle is a literal, unvectorized transliteration of the C reference loop in
`docs/maintainer/bonsai-ternary-design.md` section 1.4 (ggml-org/llama.cpp PR #29077),
written independently of `tools/convert/sources/prism_gguf.py`'s vectorized numpy
reshape/concatenate structure. If the vectorized decoder's output order ever drifts from
the reference (e.g. a stage's `n`/`m` or `n`/`h` loop nesting flipped), this test catches it
even though both paths compute the same trit-extraction arithmetic.
"""

from __future__ import annotations

import random
import struct

import numpy as np
import pytest

from tools.convert.sources.gguf_reader import GgufStore
from tools.convert.sources.prism_gguf import (
    BLOCK_ELEMENTS,
    assert_prism_ternary_gguf,
    dequantize_pq2_0_blocks,
    dequantize_ptq1_0_blocks,
    dequantize_rows,
    sign_vectors,
)

from .gguf_fixtures import build_gguf

_POW3 = (1, 3, 9, 27, 81, 243)


def _scalar_ptq1_0_block(block: bytes) -> list[float]:
    qs = list(block[0:24])
    qh = list(block[24:26])
    d = struct.unpack("<e", block[26:28])[0]
    y: list[float] = []
    j = 0
    for c in (32, 16, 8):
        while j + c <= 24:
            for n in range(5):
                for m in range(c):
                    q = (qs[j + m] * _POW3[n]) & 0xFF
                    xi = (q * 3) >> 8
                    y.append(float(xi - 1) * d)
            j += c
    for n in range(4):
        for h in range(2):
            q = (qh[h] * _POW3[n]) & 0xFF
            xi = (q * 3) >> 8
            y.append(float(xi - 1) * d)
    assert len(y) == BLOCK_ELEMENTS
    return y


def _scalar_pq2_0_block(block: bytes) -> list[float]:
    d = struct.unpack("<e", block[0:2])[0]
    qs = list(block[2:34])
    y = []
    for j in range(BLOCK_ELEMENTS):
        q = (qs[j // 4] >> ((j % 4) * 2)) & 0x3
        y.append(float(q - 1) * d)
    return y


def _random_blocks(
    rng: random.Random, block_bytes: int, count: int, *, d_high_byte_offset: int
) -> bytes:
    """Random block bytes, with the fp16 scale's exponent kept off all-ones (NaN/Inf)."""
    data = bytearray(rng.randrange(256) for _ in range(block_bytes * count))
    for block in range(count):
        offset = block * block_bytes + d_high_byte_offset
        data[offset] &= 0b11111011  # clear one exponent bit: never NaN/Inf
    return bytes(data)


def test_ptq1_0_vectorized_decode_matches_scalar_oracle():
    rng = random.Random(1234)
    raw = _random_blocks(rng, 28, 17, d_high_byte_offset=27)
    expected = np.array(
        [_scalar_ptq1_0_block(raw[i * 28 : (i + 1) * 28]) for i in range(17)], dtype=np.float32
    )
    got = dequantize_ptq1_0_blocks(raw)
    np.testing.assert_array_equal(got, expected)


def test_pq2_0_vectorized_decode_matches_scalar_oracle():
    rng = random.Random(5678)
    raw = _random_blocks(rng, 34, 13, d_high_byte_offset=1)
    expected = np.array(
        [_scalar_pq2_0_block(raw[i * 34 : (i + 1) * 34]) for i in range(13)], dtype=np.float32
    )
    got = dequantize_pq2_0_blocks(raw)
    np.testing.assert_array_equal(got, expected)


def test_ptq1_0_rejects_misaligned_byte_length():
    with pytest.raises(ValueError):
        dequantize_ptq1_0_blocks(bytes(27))


def test_pq2_0_rejects_misaligned_byte_length():
    with pytest.raises(ValueError):
        dequantize_pq2_0_blocks(bytes(33))


def test_dequantize_rows_dispatches_by_type_and_reshapes(tmp_path):
    rng = random.Random(42)
    raw = _random_blocks(rng, 28, 2, d_high_byte_offset=27)  # one row, K = 128 (one block)
    gguf_bytes = build_gguf({}, [("blk.0.test.weight", (128, 1), 143, raw[:28])])
    path = tmp_path / "one_row.gguf"
    path.write_bytes(gguf_bytes)
    with GgufStore(path) as store:
        rows = dequantize_rows(store, "blk.0.test.weight", 0, 1)
        expected = _scalar_ptq1_0_block(raw[:28])
        assert rows.shape == (1, 128)
        np.testing.assert_allclose(rows[0].numpy(), expected)


def test_sign_vectors_extracts_widths_in_order():
    gguf_bytes = build_gguf(
        {
            "prism.hadamard.sign_widths": (9, (4, (2, 3))),
            "prism.hadamard.sign_values": (9, (1, (1, -1, -1, 1, 1))),
        },
        [],
    )
    path_bytes = gguf_bytes
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "signs.gguf"
        path.write_bytes(path_bytes)
        with GgufStore(path) as store:
            signs = sign_vectors(store)
            assert set(signs) == {2, 3}
            assert signs[2].tolist() == [1, -1]
            assert signs[3].tolist() == [-1, 1, 1]


def test_sign_vectors_rejects_length_mismatch_and_bad_values(tmp_path):
    mismatched = build_gguf(
        {
            "prism.hadamard.sign_widths": (9, (4, (2,))),
            "prism.hadamard.sign_values": (9, (1, (1,))),
        },
        [],
    )
    path = tmp_path / "mismatch.gguf"
    path.write_bytes(mismatched)
    with GgufStore(path) as store:
        with pytest.raises(ValueError):
            sign_vectors(store)

    bad_values = build_gguf(
        {
            "prism.hadamard.sign_widths": (9, (4, (2,))),
            "prism.hadamard.sign_values": (9, (1, (1, 0))),
        },
        [],
    )
    path2 = tmp_path / "bad_values.gguf"
    path2.write_bytes(bad_values)
    with GgufStore(path2) as store:
        with pytest.raises(ValueError):
            sign_vectors(store)


def test_assert_prism_ternary_gguf_validates_required_metadata(tmp_path):
    good = build_gguf(
        {
            "general.architecture": (8, "qwen35"),
            "prism.hadamard.version": (4, 1),
            "prism.hadamard.block_size": (4, 1024),
            "prism.hadamard.sign_mode": (8, "explicit"),
            "general.file_type": (4, 143),
        },
        [],
    )
    path = tmp_path / "good.gguf"
    path.write_bytes(good)
    with GgufStore(path) as store:
        assert_prism_ternary_gguf(store)

    bad = build_gguf({"general.architecture": (8, "not-qwen35")}, [])
    path2 = tmp_path / "bad.gguf"
    path2.write_bytes(bad)
    with GgufStore(path2) as store:
        with pytest.raises(ValueError):
            assert_prism_ternary_gguf(store)


@pytest.mark.parametrize("format", ["t2_g128_fp16", "t5_g128_fp16"])
def test_ternary_rows_repack_the_stored_codes_exactly(tmp_path, format):
    from tools.artifact.codecs.ternary import dequantize_ternary_words
    from tools.convert.sources.prism_gguf import ternary_rows

    rng = random.Random(99)
    ptq = _random_blocks(rng, 28, 6, d_high_byte_offset=27)  # 3 rows, K = 256
    pq_codes = [rng.randrange(3) for _ in range(3 * 256)]
    pq = b""
    for block in range(6):
        codes = pq_codes[block * 128 : (block + 1) * 128]
        qs = bytes(
            codes[4 * i] | codes[4 * i + 1] << 2 | codes[4 * i + 2] << 4 | codes[4 * i + 3] << 6
            for i in range(32)
        )
        pq += struct.pack("<e", 0.25 * (block + 1)) + qs
    path = tmp_path / "rows.gguf"
    path.write_bytes(
        build_gguf({}, [("ptq.weight", (256, 3), 143, ptq), ("pq.weight", (256, 3), 142, pq)])
    )
    with GgufStore(path) as store:
        for name in ("ptq.weight", "pq.weight"):
            codes, scales = ternary_rows(store, name, 1, 3, format)
            row_bytes = 64 if format == "t2_g128_fp16" else 52
            assert codes.shape == (2, row_bytes) and scales.shape == (2, 2)
            np.testing.assert_array_equal(
                dequantize_ternary_words(codes, scales, format=format).numpy(),
                dequantize_rows(store, name, 1, 3).numpy(),
            )


@pytest.mark.parametrize("format", ["t2_g128_fp16", "t5_g128_fp16"])
def test_ternary_rows_reject_the_invalid_pq2_0_code(tmp_path, format):
    from tools.convert.sources.prism_gguf import ternary_rows

    path = tmp_path / "bad.gguf"
    path.write_bytes(build_gguf({}, [("pq.weight", (128, 1), 142, struct.pack("<e", 1.0) + bytes([0xFF]) * 32)]))
    with GgufStore(path) as store:
        with pytest.raises(ValueError, match="0, 1 or 2"):
            ternary_rows(store, "pq.weight", 0, 1, format)
