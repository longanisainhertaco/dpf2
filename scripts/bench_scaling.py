"""Benchmark strong/weak scaling and parallel HDF5 I/O.

The script launches itself under ``mpiexec`` for a range of process counts
and stores simple performance plots.  A small compute kernel is timed for
strong and weak scaling.  A parallel HDF5 write benchmark measures I/O
performance using the ``mpio`` driver in :mod:`h5py`.

Example
-------
::

    python scripts/bench_scaling.py --max-procs 4 --size 10_000 --outdir results

This command generates three plots in ``results``: ``strong.png``,
``weak.png`` and ``hdf5_io.png``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np

try:  # pragma: no cover - MPI optional
    from mpi4py import MPI  # type: ignore
except Exception:  # pragma: no cover
    MPI = None

try:  # pragma: no cover - parallel HDF5 optional
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None


# ---------------------------------------------------------------------------
# MPI kernels executed when ``--mode`` is supplied.  Each function returns the
# elapsed time in seconds across all ranks (max reduction).
# ---------------------------------------------------------------------------

def _strong_kernel(size: int) -> float:
    arr = np.ones(size, dtype=np.float64)
    start = time.perf_counter()
    np.sqrt(arr).sum()
    return time.perf_counter() - start


def _strong_scaling(size: int) -> float:
    comm = MPI.COMM_WORLD
    local = _strong_kernel(size // comm.size)
    return comm.allreduce(local, op=MPI.MAX)


def _weak_scaling(size: int) -> float:
    comm = MPI.COMM_WORLD
    local = _strong_kernel(size)
    return comm.allreduce(local, op=MPI.MAX)


def _io_bench(size: int, fname: Path) -> float:
    comm = MPI.COMM_WORLD
    chunk = size
    with h5py.File(fname, "w", driver="mpio", comm=comm) as h5f:
        dset = h5f.create_dataset("data", (chunk * comm.size,), dtype="f8")
        start = MPI.Wtime()
        offset = comm.rank * chunk
        dset[offset : offset + chunk] = np.ones(chunk)
        comm.Barrier()
        elapsed = MPI.Wtime() - start
    return comm.allreduce(elapsed, op=MPI.MAX)


# ---------------------------------------------------------------------------
# Orchestrator invoked without ``--mode``.  Launches the script under MPI for
# each process count, collects timing data and produces plots.
# ---------------------------------------------------------------------------

def _launch(mode: str, procs: int, size: int, outdir: Path) -> float:
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    if launcher is None:
        raise RuntimeError("mpiexec or mpirun is required")
    cmd = [
        launcher,
        "-n",
        str(procs),
        sys.executable,
        __file__,
        "--mode",
        mode,
        "--size",
        str(size),
        "--outfile",
        str(outdir / f"{mode}_{procs}.json"),
    ]
    subprocess.run(cmd, check=True)
    data = json.loads((outdir / f"{mode}_{procs}.json").read_text())
    return data["time"]


def _plot(x: Iterable[int], y: Iterable[float], ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot(list(x), list(y), marker="o")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log", base=2)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def driver(max_procs: int, size: int, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    ranks = [2**i for i in range(int(np.log2(max_procs)) + 1)]

    strong: List[float] = []
    for p in ranks:
        strong.append(_launch("strong", p, size, outdir))
    speedup = [strong[0] / t for t in strong]
    _plot(ranks, speedup, "Speedup", outdir / "strong.png")

    weak: List[float] = []
    for p in ranks:
        weak.append(_launch("weak", p, size, outdir))
    _plot(ranks, weak, "Runtime (s)", outdir / "weak.png")

    if h5py is not None:
        io_times: List[float] = []
        for p in ranks:
            io_times.append(_launch("io", p, size, outdir))
        bandwidth = [ (size * 8 * p) / t / 1e6 for p, t in zip(ranks, io_times) ]
        _plot(ranks, bandwidth, "Write bandwidth (MB/s)", outdir / "hdf5_io.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-procs", type=int, default=4, help="Maximum number of MPI ranks")
    parser.add_argument("--size", type=int, default=1_000_000, help="Problem size for each kernel")
    parser.add_argument("--outdir", type=Path, default=Path("scaling_results"), help="Output directory")
    parser.add_argument("--mode", choices=["strong", "weak", "io"], help=argparse.SUPPRESS)
    parser.add_argument("--outfile", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.mode:
        if MPI is None:
            raise RuntimeError("mpi4py is required when running under MPI")
        if args.mode in {"strong", "weak"}:
            func = _strong_scaling if args.mode == "strong" else _weak_scaling
            t = func(args.size)
        else:
            if h5py is None:
                raise RuntimeError("h5py built with MPI support is required for I/O benchmark")
            t = _io_bench(args.size, args.outfile.with_suffix(".h5"))
        if MPI.COMM_WORLD.rank == 0:
            args.outfile.write_text(json.dumps({"procs": MPI.COMM_WORLD.size, "time": t}))
        return

    driver(args.max_procs, args.size, args.outdir)


if __name__ == "__main__":  # pragma: no cover - manual execution only
    main()
