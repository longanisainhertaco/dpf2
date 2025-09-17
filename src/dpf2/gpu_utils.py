"""Utility helpers for optional GPU acceleration.

This module provides a light abstraction over the array backend used by the
code base.  When :mod:`cupy` is available it is used to execute operations on
GPU; otherwise it falls back to :mod:`numpy`.

The :func:`solve_linear` helper implements a mixed–precision linear solver with
iterative refinement.  The system is first solved in single precision and then
refined in double precision which provides a good balance between speed and
accuracy on modern GPU hardware.
"""

from __future__ import annotations

from typing import Any
import math
import types


try:  # pragma: no cover - optional GPU dependency
    import cupy as _cp

    xp = _cp

    def to_cpu(arr: Any) -> Any:
        """Return a host ``numpy`` array for ``arr``."""

        return _cp.asnumpy(arr)

except Exception:  # pragma: no cover - fallback to CPU numpy or a very small stub
    try:
        import numpy as _np  # type: ignore

        xp = _np

        def to_cpu(arr: Any) -> Any:
            return arr

    except Exception:  # pragma: no cover - final pure Python fallback
        xp = types.SimpleNamespace(
            array=lambda data, dtype=None: (
                [float(x) for x in data]
                if isinstance(data, (list, tuple))
                else [float(data)]
            ),
            zeros=lambda shape, dtype=None: (
                [0.0 for _ in range(shape)]
                if isinstance(shape, int)
                else [[0.0 for _ in range(shape[1])] for _ in range(shape[0])]
            ),
            sin=math.sin,
            pi=math.pi,
            dot=lambda a, b: sum(x * y for x, y in zip(a, b)),
        )

        def to_cpu(arr: Any) -> Any:
            return arr


def solve_linear(M: Any, b: Any, *, refine: bool = True) -> Any:
    """Solve ``M x = b`` using the active array module ``xp``.

    Parameters
    ----------
    M, b:
        Matrix and right‑hand side vector defining the linear system.
    refine:
        When ``True`` the routine performs a single iteration of iterative
        refinement using double precision to improve accuracy of the initial
        single precision solution.
    """

    try:
        A32 = xp.array(M, dtype=getattr(xp, "float32", float))
        b32 = xp.array(b, dtype=getattr(xp, "float32", float))
        x = xp.linalg.solve(A32, b32)
        if refine and hasattr(xp, "float64"):
            A64 = xp.array(M, dtype=getattr(xp, "float64", float))
            b64 = xp.array(b, dtype=getattr(xp, "float64", float))
            x64 = (
                x.astype(getattr(xp, "float64", float))
                if hasattr(x, "astype")
                else [float(v) for v in x]
            )
            r = b64 - xp.dot(A64, x64)
            delta = xp.linalg.solve(A64, r)
            x = x64 + delta
        return x
    except Exception:
        # Very small dense Gaussian elimination suitable for tests and stub backends
        M = [[float(M[i][j]) for j in range(len(b))] for i in range(len(b))]
        b = [float(bb) for bb in b]
        n = len(b)
        for i in range(n):
            pivot = M[i][i]
            if pivot == 0.0:
                for j in range(i + 1, n):
                    if M[j][i] != 0.0:
                        M[i], M[j] = M[j], M[i]
                        b[i], b[j] = b[j], b[i]
                        pivot = M[i][i]
                        break
            factor = pivot
            for j in range(i, n):
                M[i][j] /= factor
            b[i] /= factor
            for k in range(n):
                if k == i:
                    continue
                factor = M[k][i]
                for j in range(i, n):
                    M[k][j] -= factor * M[i][j]
                b[k] -= factor * b[i]
        return xp.array(b) if hasattr(xp, "array") else b


__all__ = ["xp", "to_cpu", "solve_linear"]
