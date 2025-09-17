"""Collect basic strong/weak scaling and Roofline data."""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List

import numpy as np


def _kernel(size: int) -> None:
    arr = np.ones(size, dtype=np.float64)
    np.sqrt(arr, out=arr)



def strong_scaling(workers: int, problem_size: int) -> float:
    chunk = problem_size // workers
    with ProcessPoolExecutor(max_workers=workers) as ex:
        start = time.perf_counter()
        list(ex.map(_kernel, [chunk] * workers))
    return time.perf_counter() - start


def weak_scaling(workers: int, problem_size: int) -> float:
    with ProcessPoolExecutor(max_workers=workers) as ex:
        start = time.perf_counter()
        list(ex.map(_kernel, [problem_size] * workers))
    return time.perf_counter() - start


def roofline(size: int) -> Dict[str, float]:
    a = np.ones((size, size))
    b = np.ones((size, size))
    start = time.perf_counter()
    np.dot(a, b)
    elapsed = time.perf_counter() - start
    flops = 2 * size ** 3
    gflops = flops / elapsed / 1e9
    bytes_moved = a.nbytes + b.nbytes + a.shape[0] * b.shape[1] * 8
    bandwidth = bytes_moved / elapsed / 1e9
    return {"size": size, "gflops": gflops, "bandwidth": bandwidth}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-workers", type=int, default=4, help="maximum processes to use")
    parser.add_argument("--problem-size", type=int, default=100_000, help="base problem size")
    parser.add_argument("--outdir", type=Path, default=Path("docs/performance"), help="output directory")
    args = parser.parse_args()

    workers = [2 ** i for i in range(int(np.log2(args.max_workers)) + 1)]

    strong: List[Dict[str, float]] = []
    for w in workers:
        strong.append({"procs": w, "time": strong_scaling(w, args.problem_size)})

    weak: List[Dict[str, float]] = []
    for w in workers:
        weak.append({"procs": w, "time": weak_scaling(w, args.problem_size)})

    roof = roofline(int(np.sqrt(args.problem_size)))

    args.outdir.mkdir(parents=True, exist_ok=True)
    result = {"strong": strong, "weak": weak, "roofline": roof}
    (args.outdir / "scaling.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
