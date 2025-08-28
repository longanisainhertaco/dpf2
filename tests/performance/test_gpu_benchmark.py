"""Basic correctness check for the GPU-enabled linear solver."""

import numpy as np
import pytest

from dpf2.gpu_utils import xp, solve_linear


@pytest.mark.parametrize(
    "A,b,expected",
    [
        (np.array([[3.0, 1.0], [1.0, 2.0]]), np.array([9.0, 8.0]), np.array([2.0, 3.0])),
    ],
)
def test_solve_linear_matches_reference(A, b, expected):
    x = solve_linear(A, b)
    if hasattr(xp, "asnumpy"):
        x = xp.asnumpy(x)
    assert np.allclose(x, expected, atol=1e-5)
