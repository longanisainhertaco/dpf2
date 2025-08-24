from __future__ import annotations

"""Time dependent ionisation and recombination kinetics.

This module provides a tiny collisional–radiative model that evolves the
populations of multiple charge states using tabulated ionisation and
recombination coefficients.  The implementation is intentionally compact
and aimed at unit tests rather than high fidelity simulation.  The rate
coefficients typically originate from reduced FLYCHK/CRM datasets.
"""

from pathlib import Path

# ``numpy`` is optional; when unavailable or when a minimal stub is provided
# in the test environment we fall back to lightweight implementations.
try:  # pragma: no cover - runtime check for real numpy
    import numpy as _np  # type: ignore
    # ``numpy_stub`` used in tests lacks ``loadtxt``/``interp``; trigger the
    # fallback below in that case.
    if not hasattr(_np, "loadtxt"):
        raise ModuleNotFoundError
    np = _np
except ModuleNotFoundError:  # pragma: no cover
    import csv
    import types

    def _array(x):
        if isinstance(x, (list, tuple)):
            return [float(v) for v in x]
        return [float(x)]

    def _loadtxt(path, delimiter=",", skiprows=0):
        data = []
        with open(path) as f:
            reader = csv.reader(f, delimiter=delimiter)
            for _ in range(skiprows):
                next(reader, None)
            for row in reader:
                data.append([float(v) for v in row])
        return data

    def _interp(T, xp, fp, left=None, right=None):
        result = []
        for t in T:
            if t <= xp[0]:
                result.append(fp[0] if left is None else left)
            elif t >= xp[-1]:
                result.append(fp[-1] if right is None else right)
            else:
                for i in range(1, len(xp)):
                    if t < xp[i]:
                        t0, t1 = xp[i - 1], xp[i]
                        f0, f1 = fp[i - 1], fp[i]
                        result.append(f0 + (f1 - f0) * (t - t0) / (t1 - t0))
                        break
        return result

    np = types.SimpleNamespace(
        array=_array,
        asarray=_array,
        loadtxt=_loadtxt,
        interp=lambda T, xp, fp, left=None, right=None: _interp(T, xp, fp, left, right),
        zeros_like=lambda arr: [0.0 for _ in arr],
        dot=lambda a, b: sum(x * y for x, y in zip(a, b)),
        arange=lambda n: list(range(n)),
        sum=lambda arr: sum(arr),
    )


class RateTable:
    """Tabulated ionisation and recombination rate coefficients."""

    def __init__(self, T: list[float], k_ion: list[float], k_rec: list[float]):
        self.T = T
        self.k_ion = k_ion
        self.k_rec = k_rec

    @classmethod
    def from_csv(cls, path: str | Path) -> "RateTable":
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        T = [row[0] for row in data]
        k_i = [row[1] for row in data]
        k_r = [row[2] for row in data]
        return cls(T=T, k_ion=k_i, k_rec=k_r)

    def ion_rate(self, T) -> list[float]:
        return np.interp(T, self.T, self.k_ion, left=self.k_ion[0], right=self.k_ion[-1])

    def rec_rate(self, T) -> list[float]:
        return np.interp(T, self.T, self.k_rec, left=self.k_rec[0], right=self.k_rec[-1])


class RateEquations:
    """Simple multi-species collisional–radiative model.

    The solver tracks the population ``n[i]`` of each charge state ``i``.
    A single set of ionisation and recombination coefficients is applied
    between adjacent charge states which suffices for the lightweight
    chemistry regression tests.
    """

    def __init__(self, rates: RateTable, levels: int = 2):
        self.rates = rates
        self.levels = levels

    # ------------------------------------------------------------------
    # Helper diagnostics
    # ------------------------------------------------------------------
    def electron_density(self, n: list[float]) -> float:
        """Return the free electron density for populations ``n``."""

        charges = np.arange(self.levels)
        return float(np.dot(charges, n))

    def mean_charge(self, n: list[float]) -> float:
        """Return the mean charge state ``<Z>`` for populations ``n``."""

        n_total = float(np.sum(n))
        if n_total == 0.0:
            return 0.0
        return self.electron_density(n) / n_total

    # ------------------------------------------------------------------
    # Rate equations
    # ------------------------------------------------------------------
    def rhs(self, n: list[float], T: float) -> list[float]:
        """Time derivatives for charge state populations ``n``."""

        k_i = self.rates.ion_rate([T])[0]
        k_r = self.rates.rec_rate([T])[0]
        ne = self.electron_density(n)

        dn = np.zeros_like(n)
        for i in range(self.levels):
            ion_in = k_i * ne * n[i - 1] if i > 0 else 0.0
            rec_in = k_r * ne * n[i + 1] if i < self.levels - 1 else 0.0
            ion_out = k_i * ne * n[i] if i < self.levels - 1 else 0.0
            rec_out = k_r * ne * n[i] if i > 0 else 0.0
            dn[i] = ion_in + rec_in - ion_out - rec_out

        return dn

    def step(self, n: list[float], T: float, dt: float) -> list[float]:
        """Advance populations ``n`` by a single explicit Euler step."""

        dn = self.rhs(n, T)
        return [ni + dt * dni for ni, dni in zip(n, dn)]

class MultiSpeciesTransport:
    """Very small multi‑species diffusion and wall ablation model.

    The model advects a set of species in one spatial dimension using a
    simple finite difference discretisation of Fick's law.  An optional
    wall ablation source can inject material for each species at the
    ``i=0`` boundary which is sufficient for regression and validation
    style tests.
    """

    def __init__(self, diffusion: dict[str, float], dx: float = 1.0) -> None:
        self.diffusion = diffusion
        self.dx = dx

    def step(
        self,
        n: dict[str, list[float]],
        dt: float,
        wall_ablation: dict[str, float] | None = None,
    ) -> dict[str, list[float]]:
        """Advance the species densities by ``dt``.

        Parameters
        ----------
        n:
            Mapping of species name to a list of cell averaged densities.
        dt:
            Time step in seconds.
        wall_ablation:
            Optional mapping of species name to a mass injection rate
            applied at the boundary cell ``i=0``.
        """

        wall_ablation = wall_ablation or {}
        cells = len(next(iter(n.values())))
        updated: dict[str, list[float]] = {
            sp: list(vals) for sp, vals in n.items()
        }
        for sp, values in n.items():
            D = self.diffusion.get(sp, 0.0)
            dv = [0.0] * cells
            for i in range(cells):
                flux_l = 0.0
                flux_r = 0.0
                if i > 0:
                    flux_l = -D * (values[i] - values[i - 1]) / self.dx
                if i < cells - 1:
                    flux_r = -D * (values[i + 1] - values[i]) / self.dx
                dv[i] = (flux_l - flux_r) / self.dx
            updated[sp] = [v + dt * dv_i for v, dv_i in zip(values, dv)]
            if sp in wall_ablation:
                updated[sp][0] += wall_ablation[sp] * dt
        return updated


__all__ = ["RateTable", "RateEquations", "MultiSpeciesTransport"]

