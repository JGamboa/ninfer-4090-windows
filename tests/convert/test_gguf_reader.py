from __future__ import annotations

import pytest

from tools.convert.sources.gguf_reader import GgufStore, block_layout

from .gguf_fixtures import build_gguf


def test_header_metadata_tensor_table_and_bounded_row_reads(tmp_path):
    row0 = bytes(range(28))
    row1 = bytes(range(28, 56))
    raw = build_gguf(
        metadata={
            "general.architecture": (8, "qwen35"),
            "general.file_type": (4, 143),
            "prism.hadamard.sign_widths": (9, (4, (2, 3))),
        },
        tensors=[("blk.0.test.weight", (128, 2), 143, row0 + row1)],
    )
    path = tmp_path / "tiny.gguf"
    path.write_bytes(raw)

    with GgufStore(path) as store:
        assert store.version == 3
        assert store.metadata["general.architecture"] == "qwen35"
        assert store.metadata["general.file_type"] == 143
        assert store.metadata["prism.hadamard.sign_widths"] == [2, 3]
        assert store.tensor_order == ["blk.0.test.weight"]

        info = store.tensor("blk.0.test.weight")
        assert info.shape == (128, 2)
        assert info.type_name == "PTQ1_0"
        assert info.elements == 256

        assert store.row_bytes("blk.0.test.weight") == 28
        assert store.tensor_bytes("blk.0.test.weight") == 56
        assert store.read_tensor_raw("blk.0.test.weight") == row0 + row1
        assert store.read_rows_raw("blk.0.test.weight", 0, 1) == row0
        assert store.read_rows_raw("blk.0.test.weight", 1, 2) == row1
        assert store.read_tensor_raw("blk.0.test.weight", 0, 28) == row0

        with pytest.raises(KeyError):
            store.tensor("missing")
        with pytest.raises(ValueError):
            store.read_rows_raw("blk.0.test.weight", 0, 3)
        with pytest.raises(ValueError):
            store.read_tensor_raw("blk.0.test.weight", 0, 57)


def test_rejects_bad_magic_and_unsupported_version(tmp_path):
    good = build_gguf({}, [("t", (1,), 0, b"\x00\x00\x00\x00")])

    bad_magic = tmp_path / "bad_magic.gguf"
    bad_magic.write_bytes(b"NOPE" + good[4:])
    with pytest.raises(ValueError):
        GgufStore(bad_magic)

    bad_version = tmp_path / "bad_version.gguf"
    import struct

    bad_version.write_bytes(good[:4] + struct.pack("<I", 2) + good[8:])
    with pytest.raises(ValueError):
        GgufStore(bad_version)


def test_block_layout_rejects_unregistered_types():
    assert block_layout(143) == (128, 28)
    assert block_layout(142) == (128, 34)
    with pytest.raises(ValueError):
        block_layout(999)


def test_row_bytes_requires_matrix_and_block_aligned_k(tmp_path):
    raw = build_gguf({}, [("vector", (128,), 143, bytes(28)), ("odd", (100, 1), 143, bytes(0))])
    path = tmp_path / "odd.gguf"
    path.write_bytes(raw)
    with GgufStore(path) as store:
        with pytest.raises(ValueError):
            store.row_bytes("vector")
        with pytest.raises(ValueError):
            store.row_bytes("odd")
