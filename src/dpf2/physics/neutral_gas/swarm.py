from __future__ import annotations
"""Utilities for validating swarm parameters against reference data."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

from dpf2.neutral.dsmc import load_lxcat_table


@dataclass
class SwarmParameters:
    mobility: float
    diffusion: float


def compute_swarm_parameters(table: np.ndarray) -> SwarmParameters:
    """Compute simple swarm parameters from a cross‑section table.

    The implementation is intentionally simplistic: mobility and diffusion
    are taken to scale inversely with the mean cross section of the table.
    This is sufficient for validating that LXCat/Bolsig+ style reference
    data can be ingested correctly.
    """

    sigma = float(np.mean(table[:, 1]))
    if sigma <= 0:  # pragma: no cover - defensive programming
        raise ValueError("mean cross section must be positive")
    mobility = 1.0 / sigma
    diffusion = 1.0 / (3.0 * sigma)
    return SwarmParameters(mobility=mobility, diffusion=diffusion)


def validate_swarm_parameters(path: Path, reference: Dict[str, float]) -> SwarmParameters:
    """Validate swarm parameters against reference values.

    Parameters
    ----------
    path:
        Path to an LXCat style cross‑section table.
    reference:
        Mapping of parameter names to reference values.  Supported keys are
        ``"mobility"`` and ``"diffusion"``.
    """

    table = load_lxcat_table(Path(path))
    params = compute_swarm_parameters(table)
    for key, ref in reference.items():
        val = getattr(params, key)
        if not np.isclose(val, ref, rtol=0.05, atol=0.0):
            raise ValueError(f"{key} {val} disagrees with reference {ref}")
    return params


__all__ = ["SwarmParameters", "compute_swarm_parameters", "validate_swarm_parameters"]
