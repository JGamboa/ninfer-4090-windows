"""PrismML PTQ1_0 / PQ2_0 ternary decoders and Hadamard sign vectors over a GgufStore.

The two block-quantized codecs are ports, verbatim in trit/bit order, of the reference C
loops in `docs/maintainer/bonsai-ternary-design.md` section 1.4 (from ggml-org/llama.cpp
PR #29077). Do not re-derive the output order independently; `tests/convert/
test_prism_gguf.py` checks the vectorized decoders here against a literal scalar
transliteration of the same C loops.

Both codecs first produce the stored codes {0, 1, 2} and FP16 block scales;
:func:`ternary_rows` repacks them exactly into `t5_g128_fp16` words (design doc sections 2
and 9.1)
and :func:`dequantize_rows` reproduces the reference C dequantization `(code - 1) * d`.
"""

from __future__ import annotations

import numpy as np
import torch

from tools.artifact.codecs.ternary import pack_ternary_codes

from .gguf_reader import GgufStore

PTQ1_0_TYPE = 143
PQ2_0_TYPE = 142
BLOCK_ELEMENTS = 128  # QK_PTQ1_0 == QK_PQ2_0 == 128

_PTQ1_0_BLOCK_BYTES = 28
_PQ2_0_BLOCK_BYTES = 34
_POW3 = (1, 3, 9, 27, 81, 243)


def _trit(byte_column: np.ndarray, n: int) -> np.ndarray:
    """`(uint8_t)(byte * pow3[n])` truncation, then the `((uint16_t)q*3)>>8` trit
    extraction: the stored code in {0, 1, 2} (value `code - 1`). `byte_column` is any-shaped
    uint32."""
    q = (byte_column * np.uint32(_POW3[n])) & np.uint32(0xFF)
    return ((q * np.uint32(3)) >> np.uint32(8)).astype(np.uint8)


def ptq1_0_block_codes(raw: bytes | bytearray | memoryview) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized, order-preserving port of `dequantize_row_ptq1_0`, stopping at codes.

    `raw` must hold a whole number of 28-byte `block_ptq1_0` records. Returns uint8 codes
    in {0, 1, 2} of shape `(nblocks, 128)` and the float16 block scales `(nblocks,)`, in
    exactly the reference order: the stage loop emits 5 trits x 16 bytes (`qs[0:16]`, n
    outer/m inner), then 5 trits x 8 bytes (`qs[16:24]`, n outer/m inner), then 4 trits x
    2 `qh` bytes (n outer/h inner).
    """
    if len(raw) % _PTQ1_0_BLOCK_BYTES:
        raise ValueError("PTQ1_0 raw byte length is not a multiple of the 28-byte block")
    nblocks = len(raw) // _PTQ1_0_BLOCK_BYTES
    block_dtype = np.dtype([("qs", np.uint8, 24), ("qh", np.uint8, 2), ("d", "<f2")])
    blocks = np.frombuffer(raw, dtype=block_dtype, count=nblocks)
    qs = blocks["qs"].astype(np.uint32)  # [nblocks, 24]
    qh = blocks["qh"].astype(np.uint32)  # [nblocks, 2]

    stage16 = np.concatenate([_trit(qs[:, 0:16], n) for n in range(5)], axis=1)  # [nblocks, 80]
    stage8 = np.concatenate([_trit(qs[:, 16:24], n) for n in range(5)], axis=1)  # [nblocks, 40]
    stageh = np.concatenate([_trit(qh, n) for n in range(4)], axis=1)  # [nblocks, 8]
    codes = np.concatenate([stage16, stage8, stageh], axis=1)  # [nblocks, 128]
    assert codes.shape[1] == BLOCK_ELEMENTS
    return codes, blocks["d"].copy()


def pq2_0_block_codes(raw: bytes | bytearray | memoryview) -> tuple[np.ndarray, np.ndarray]:
    """PQ2_0 codes `(qs[j/4] >> ((j%4)*2)) & 3` and float16 block scales.

    `raw` must hold a whole number of 34-byte `block_pq2_0` records (`ggml_half d` then
    `qs[32]`). Code 3 is returned as stored (the reference dequantizes it to `2 * d`);
    :func:`ternary_rows` rejects it when packing ternary words.
    """
    if len(raw) % _PQ2_0_BLOCK_BYTES:
        raise ValueError("PQ2_0 raw byte length is not a multiple of the 34-byte block")
    nblocks = len(raw) // _PQ2_0_BLOCK_BYTES
    block_dtype = np.dtype([("d", "<f2"), ("qs", np.uint8, 32)])
    blocks = np.frombuffer(raw, dtype=block_dtype, count=nblocks)
    qs = blocks["qs"]  # [nblocks, 32]

    codes = np.empty((nblocks, BLOCK_ELEMENTS), dtype=np.uint8)
    for shift in range(4):
        # j = 4*byte_index + shift, so this fills columns shift, shift+4, shift+8, ...
        codes[:, shift::4] = (qs >> np.uint8(shift * 2)) & np.uint8(0x3)
    return codes, blocks["d"].copy()


def dequantize_ptq1_0_blocks(raw: bytes | bytearray | memoryview) -> np.ndarray:
    """`(code - 1) * d` per block, float32 `(nblocks, 128)`; see :func:`ptq1_0_block_codes`."""
    codes, d = ptq1_0_block_codes(raw)
    return (codes.astype(np.float32) - 1.0) * d.astype(np.float32)[:, None]


def dequantize_pq2_0_blocks(raw: bytes | bytearray | memoryview) -> np.ndarray:
    """`(code - 1) * d` per block, float32 `(nblocks, 128)`; see :func:`pq2_0_block_codes`."""
    codes, d = pq2_0_block_codes(raw)
    return (codes.astype(np.float32) - 1.0) * d.astype(np.float32)[:, None]


_DECODERS = {
    PTQ1_0_TYPE: dequantize_ptq1_0_blocks,
    PQ2_0_TYPE: dequantize_pq2_0_blocks,
}


def dequantize_rows(store: GgufStore, name: str, row_begin: int, row_end: int) -> torch.Tensor:
    """Dequantize a contiguous row range of a PTQ1_0/PQ2_0 tensor as `[rows, K]` float32."""
    info = store.tensor(name)
    try:
        decode = _DECODERS[info.ggml_type]
    except KeyError:
        raise ValueError(
            f"{name}: unsupported Prism ternary type {info.type_name} (id {info.ggml_type})"
        ) from None
    k = info.shape[0]
    if k % BLOCK_ELEMENTS:
        raise ValueError(f"{name}: K={k} is not a multiple of the ternary block size {BLOCK_ELEMENTS}")
    raw = store.read_rows_raw(name, row_begin, row_end)
    decoded = decode(raw)
    rows = row_end - row_begin
    return torch.from_numpy(decoded.reshape(rows, k).copy())


_CODE_DECODERS = {
    PTQ1_0_TYPE: ptq1_0_block_codes,
    PQ2_0_TYPE: pq2_0_block_codes,
}


def ternary_rows(
    store: GgufStore, name: str, row_begin: int, row_end: int, format: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact ternary words of a row range in `format` (`t5_g128_fp16`):
    packed uint8 code bytes `[rows, code_row_bytes]` and float16 scales `[rows, K/128]`.
    The GGUF's trits are repacked exactly; nothing is expanded to floats."""
    info = store.tensor(name)
    try:
        decode = _CODE_DECODERS[info.ggml_type]
    except KeyError:
        raise ValueError(
            f"{name}: unsupported Prism ternary type {info.type_name} (id {info.ggml_type})"
        ) from None
    k = info.shape[0]
    if k % BLOCK_ELEMENTS:
        raise ValueError(f"{name}: K={k} is not a multiple of the ternary block size {BLOCK_ELEMENTS}")
    rows = row_end - row_begin
    codes, scales = decode(store.read_rows_raw(name, row_begin, row_end))
    packed = pack_ternary_codes(torch.from_numpy(codes.reshape(rows, k)), format)
    return packed, torch.from_numpy(scales.reshape(rows, k // BLOCK_ELEMENTS))


def sign_vectors(store: GgufStore) -> dict[int, torch.Tensor]:
    """`{width: int8[width] of +-1}` from `prism.hadamard.sign_widths`/`sign_values`."""
    widths = store.metadata.get("prism.hadamard.sign_widths")
    values = store.metadata.get("prism.hadamard.sign_values")
    if widths is None or values is None:
        raise ValueError(f"{store.path}: missing prism.hadamard.sign_widths/sign_values metadata")
    if sum(widths) != len(values):
        raise ValueError(
            f"{store.path}: sign_values has {len(values)} entries, "
            f"expected sum(sign_widths)={sum(widths)}"
        )
    result: dict[int, torch.Tensor] = {}
    offset = 0
    for width in widths:
        if width in result:
            raise ValueError(f"{store.path}: duplicate sign width {width}")
        chunk = torch.tensor(values[offset : offset + width], dtype=torch.int8)
        if not bool(torch.all((chunk == 1) | (chunk == -1))):
            raise ValueError(f"{store.path}: sign vector for width {width} is not all +-1")
        result[width] = chunk
        offset += width
    return result


def assert_prism_ternary_gguf(store: GgufStore) -> None:
    """Validate the metadata assumptions this decoder and the Bonsai recipe depend on."""
    metadata = store.metadata
    if metadata.get("general.architecture") != "qwen35":
        raise ValueError(f"{store.path}: expected general.architecture=qwen35")
    if metadata.get("prism.hadamard.version") != 1:
        raise ValueError(f"{store.path}: expected prism.hadamard.version=1")
    if metadata.get("prism.hadamard.block_size") != 1024:
        raise ValueError(f"{store.path}: expected prism.hadamard.block_size=1024")
    if metadata.get("prism.hadamard.sign_mode") != "explicit":
        raise ValueError(f"{store.path}: expected prism.hadamard.sign_mode=explicit")
    if metadata.get("general.file_type") not in (PQ2_0_TYPE, PTQ1_0_TYPE):
        raise ValueError(f"{store.path}: expected general.file_type in {{142, 143}}")


__all__ = [
    "PTQ1_0_TYPE",
    "PQ2_0_TYPE",
    "BLOCK_ELEMENTS",
    "dequantize_ptq1_0_blocks",
    "dequantize_pq2_0_blocks",
    "dequantize_rows",
    "ptq1_0_block_codes",
    "pq2_0_block_codes",
    "ternary_rows",
    "sign_vectors",
    "assert_prism_ternary_gguf",
]
