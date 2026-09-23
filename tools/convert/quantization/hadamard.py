"""The Prism normalized Sylvester Walsh-Hadamard rotation (design doc section 1.5).

`H` is the normalized Sylvester Walsh-Hadamard matrix of order `n` (a power of two),
`H = (1/sqrt(n)) * H_2^{(x) log2(n)}`, `H_2 = [[1, 1], [1, -1]]`. It is symmetric and
involutory (`H @ H = I`), so the inverse transform reuses the same matrix. Bonsai applies
it blockwise over consecutive 1024-element blocks of a projection's input axis, with an
explicit +-1 sign vector applied before the butterfly (`rotate_rows`) or after it, for the
embedding's inverse transform (`unrotate_embedding_rows`).

This is a verbatim port of the reference code in the design doc
(`docs/maintainer/bonsai-ternary-design.md`, section 1.5); do not re-derive it independently
of that reference and the numeric test in `tests/convert/test_hadamard.py`.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

DEFAULT_BLOCK = 1024


@lru_cache(maxsize=8)
def sylvester_hadamard(n: int) -> np.ndarray:
    """Normalized Sylvester-ordered Hadamard matrix of order `n` (`n` a power of two)."""
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"sylvester_hadamard requires a power of two, got {n}")
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def rotate_rows(x: np.ndarray, signs: np.ndarray, block: int = DEFAULT_BLOCK) -> np.ndarray:
    """Forward transform for a rotated projection input: `y = H_blk(signs * x)`.

    `x` is `[T, K]` with `K % block == 0`; `signs` is a `+-1` vector of length `K`.
    """
    if x.shape[-1] % block:
        raise ValueError(f"rotate_rows requires K % {block} == 0, got K={x.shape[-1]}")
    if signs.shape != (x.shape[-1],):
        raise ValueError(f"signs must have shape ({x.shape[-1]},), got {signs.shape}")
    h = sylvester_hadamard(block)
    xs = (x * signs[None, :]).reshape(x.shape[0], -1, block)
    return (xs @ h.T).reshape(x.shape)  # h is symmetric, so h.T == h


def unrotate_embedding_rows(
    z: np.ndarray, signs: np.ndarray, block: int = DEFAULT_BLOCK
) -> np.ndarray:
    """Inverse transform for the embedding gather: `h = signs * H_blk(z)`."""
    if z.shape[-1] % block:
        raise ValueError(f"unrotate_embedding_rows requires K % {block} == 0, got K={z.shape[-1]}")
    if signs.shape != (z.shape[-1],):
        raise ValueError(f"signs must have shape ({z.shape[-1]},), got {signs.shape}")
    h = sylvester_hadamard(block)
    return (z.reshape(z.shape[0], -1, block) @ h.T).reshape(z.shape) * signs[None, :]


__all__ = ["DEFAULT_BLOCK", "sylvester_hadamard", "rotate_rows", "unrotate_embedding_rows"]
