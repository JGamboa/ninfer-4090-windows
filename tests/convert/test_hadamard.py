from __future__ import annotations

import numpy as np
import pytest

from tools.convert.quantization.hadamard import (
    rotate_rows,
    sylvester_hadamard,
    unrotate_embedding_rows,
)


@pytest.mark.parametrize("n", [2, 4, 8, 1024])
def test_sylvester_hadamard_is_normalized_symmetric_and_involutory(n):
    h = sylvester_hadamard(n)
    assert h.shape == (n, n)
    np.testing.assert_allclose(h, h.T, atol=1e-12)
    np.testing.assert_allclose(h @ h, np.eye(n), atol=1e-9)


def test_sylvester_hadamard_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        sylvester_hadamard(1000)


def test_rotate_and_unrotate_embedding_are_exact_inverses():
    rng = np.random.default_rng(0)
    block = 1024
    k = block * 3
    x = rng.standard_normal((4, k))
    signs = rng.choice([-1.0, 1.0], size=k)

    rotated = rotate_rows(x, signs, block=block)
    restored = unrotate_embedding_rows(rotated, signs, block=block)
    np.testing.assert_allclose(restored, x, atol=1e-9)


def test_rotate_rows_requires_block_aligned_k_and_matching_signs():
    with pytest.raises(ValueError):
        rotate_rows(np.zeros((1, 100)), np.ones(100), block=1024)
    with pytest.raises(ValueError):
        rotate_rows(np.zeros((1, 1024)), np.ones(512), block=1024)


def test_unrotate_embedding_rows_requires_block_aligned_k_and_matching_signs():
    with pytest.raises(ValueError):
        unrotate_embedding_rows(np.zeros((1, 100)), np.ones(100), block=1024)
    with pytest.raises(ValueError):
        unrotate_embedding_rows(np.zeros((1, 1024)), np.ones(512), block=1024)
