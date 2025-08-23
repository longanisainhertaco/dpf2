"""Utility functions to generate synthetic scaling data.

These routines mimic weak and strong scaling studies and can be used to
produce documentation figures.  They do not run the full simulation but
serve as placeholders for HPC environments.
"""
from __future__ import annotations

from typing import Iterable, Dict


def weak_scaling(sizes: Iterable[int]) -> Dict[int, float]:
    """Return ideal weak scaling times for given problem sizes."""
    return {n: 1.0 for n in sizes}


def strong_scaling(sizes: Iterable[int]) -> Dict[int, float]:
    """Return ideal strong scaling times for given core counts."""
    return {n: 1.0 / n for n in sizes}


def document_results(path, weak: Dict[int, float], strong: Dict[int, float]):
    """Write scaling results to a simple markdown file."""
    with open(path, "w") as f:
        f.write("# Scaling Results\n\n")
        f.write("## Weak Scaling\n")
        for n, t in weak.items():
            f.write(f"- size {n}: {t:.2f} s\n")
        f.write("\n## Strong Scaling\n")
        for n, t in strong.items():
            f.write(f"- {n} cores: {t:.2f} s\n")
