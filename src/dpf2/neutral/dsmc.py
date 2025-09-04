"""Simplified Direct Simulation Monte Carlo (DSMC) neutral gas solver.

This module implements a very small subset of the classical Bird
algorithm [1]_.  It is **not** meant to be a production ready solver –
only a light‑weight implementation to exercise coupling pathways in the
unit tests.  The goal is to provide a deterministic and easily
verifiable piece of code rather than a highly optimised DSMC package.

The solver supports two important features:

* Loading and validating collision cross sections from tabulated
  LXCat/Bolsig+ style datasets using a tiny "pedigree" framework.
* A tunable Knudsen number which controls the mean free path/number
  density relationship.

References
----------
.. [1] G.A. Bird, *Molecular Gas Dynamics and the Direct Simulation of
       Gas Flows*, Clarendon Press, 1994.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import random

from dpf2.io.datasets import load_dataset_manifest


def load_lxcat_table(path: Path) -> np.ndarray:
    """Return an ``(N, 2)`` array of energy [eV] and cross section [m^2]."""
    data = np.loadtxt(path, delimiter=",")
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("cross‑section table must have two columns")
    return data


def _validate_cross_sections(table: np.ndarray) -> None:
    """Validate tabulated cross sections.

    Parameters
    ----------
    table:
        ``(N, 2)`` array of energies [eV] and cross sections [m^2].
    """
    # The small numpy stub used in tests lacks ``np.any``/``np.diff`` so we
    # perform the checks using pure Python constructs.
    energies = [row[0] for row in table]
    sigmas = [row[1] for row in table]
    if any(s < 0 for s in sigmas):
        raise ValueError("cross sections must be non‑negative")
    if any(b <= a for a, b in zip(energies, energies[1:])):
        raise ValueError("energy grid must be strictly increasing")


@dataclass
class DSMC:
    """Very small DSMC solver with a tunable Knudsen number.

    Parameters
    ----------
    cross_sections:
        ``(N, 2)`` array with [energy_eV, cross_section_m2].
    knudsen_number:
        Dimensionless ``Kn = \lambda / L``.  ``L`` is assumed unity in
        this toy model which implies a number density of
        ``n = 1/(sqrt(2)*σ*Kn)`` using the average cross section ``σ``.
    velocities:
        Initial particle velocities in m/s.  A minimal Bird style
        collision operator is applied when :meth:`run` is executed.
    """

    cross_sections: np.ndarray
    knudsen_number: float = 1.0
    velocities: np.ndarray | None = None

    def __post_init__(self) -> None:  # pragma: no cover - simple validation
        _validate_cross_sections(self.cross_sections)
        if self.velocities is None:
            # Default to a single stationary particle – the actual values
            # are irrelevant for the coupling tests.
            self.velocities = np.zeros(1)
        else:
            self.velocities = np.array(self.velocities)

    # ------------------------------------------------------------------
    # Cross‑section handling
    @classmethod
    def from_lxcat(cls, species: str, dataset_id: str, knudsen_number: float = 1.0, velocities: Iterable[float] | None = None) -> "DSMC":
        """Create a solver loading cross sections from a dataset.

        The :mod:`dpf2.io.datasets` manifest acts as a light‑weight
        "pedigree" system mapping identifiers to local files.  Only the
        keys required for the unit tests are implemented.
        """
        manifest = load_dataset_manifest()
        try:
            rel = manifest["lxcat"][dataset_id][species]
        except KeyError as exc:  # pragma: no cover - configuration error
            raise ValueError(f"unknown LXCat dataset {dataset_id!r} for {species!r}") from exc
        table = load_lxcat_table(Path(rel))
        return cls(table, knudsen_number=knudsen_number, velocities=None if velocities is None else np.array(list(velocities)))

    # ------------------------------------------------------------------
    # Core DSMC algorithm
    def _select_pairs(self) -> Iterable[Tuple[int, int]]:
        indices = list(np.arange(len(self.velocities)))
        # ``numpy``'s RNG is not available in the minimal stub, so fall back to
        # Python's ``random`` module.
        random.shuffle(indices)
        for i in range(0, len(indices), 2):
            j = (i + 1) % len(indices)
            yield indices[i], indices[j]

    def _collision_probability(self, g_rel: float, dt: float) -> float:
        sigma = float(np.max(self.cross_sections[:, 1]))
        return sigma * g_rel * dt

    def _bird_step(self, dt: float) -> None:
        for i, j in self._select_pairs():
            g = abs(self.velocities[i] - self.velocities[j])
            p = self._collision_probability(g, dt)
            if random.random() < p:
                # Simple 1D scattering: exchange velocities
                self.velocities[i], self.velocities[j] = self.velocities[j], self.velocities[i]

    # ------------------------------------------------------------------
    def run(self, dt: float) -> float:
        """Advance the particle system by ``dt`` seconds and return density."""
        self._bird_step(dt)
        return self.compute_neutral_density()

    # ------------------------------------------------------------------
    def compute_neutral_density(self) -> float:
        sigma = float(np.mean(self.cross_sections[:, 1]))
        if sigma <= 0:  # pragma: no cover - defensive
            raise ValueError("mean cross section must be positive")
        return 1.0 / (np.sqrt(2.0) * sigma * self.knudsen_number)


__all__ = ["DSMC", "load_lxcat_table"]
