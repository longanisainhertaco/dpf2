"""Simple CPU/GPU scaling benchmark for linear solves.

This script compares the execution time of a dense linear solve using
``numpy`` on the CPU against ``cupy`` on the GPU when available.  It is not
intended to be an exhaustive performance study but rather a quick check that
the GPU pathway is operational.
"""

import time
import random
import sys
from pathlib import Path

try:  # optional CPU baseline using numpy if available
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy not installed
    np = None

try:  # optional GPU benchmark
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover - GPU unavailable
    cp = None


def _rand_matrix(size: int):
    return [[random.random() for _ in range(size)] for _ in range(size)]


def _rand_vector(size: int):
    return [random.random() for _ in range(size)]


def benchmark_numpy(size: int, iterations: int) -> float:
    A = np.random.rand(size, size).astype(np.float32)
    b = np.random.rand(size).astype(np.float32)
    np.linalg.solve(A, b)
    start = time.time()
    for _ in range(iterations):
        np.linalg.solve(A, b)
    return time.time() - start


def benchmark_python(size: int, iterations: int) -> float:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from dpf2.gpu_utils import solve_linear

    A = _rand_matrix(size)
    b = _rand_vector(size)
    solve_linear(A, b)
    start = time.time()
    for _ in range(iterations):
        solve_linear(A, b)
    return time.time() - start


def benchmark_cupy(size: int, iterations: int) -> float:
    A = cp.random.rand(size, size).astype(cp.float32)
    b = cp.random.rand(size).astype(cp.float32)
    cp.linalg.solve(A, b)
    cp.cuda.Stream.null.synchronize()
    start = time.time()
    for _ in range(iterations):
        cp.linalg.solve(A, b)
    cp.cuda.Stream.null.synchronize()
    return time.time() - start


def main() -> None:
    size = 256
    iterations = 5
    if np is not None:
        cpu = benchmark_numpy(size, iterations)
        print(f"CPU (numpy) : {cpu:.6f}s")
    else:
        cpu = benchmark_python(size, iterations)
        print(f"CPU (python) : {cpu:.6f}s")

    if cp is not None:
        gpu = benchmark_cupy(size, iterations)
        print(f"GPU (cupy) : {gpu:.6f}s")
    else:
        print("CuPy not available - GPU benchmark skipped")


if __name__ == "__main__":
    main()
