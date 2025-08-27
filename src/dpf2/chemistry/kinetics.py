from __future__ import annotations

"""Time dependent ionisation and recombination kinetics.

This module provides a tiny collisional–radiative model that evolves the
populations of multiple charge states using tabulated ionisation and
recombination coefficients.  The implementation is intentionally compact
and aimed at unit tests rather than high fidelity simulation.  The rate
coefficients typically originate from reduced FLYCHK/CRM datasets.  The
original version of the module supported only a single species.  For the
sheath/impurity tests we extend the tables to hold data for multiple
species while retaining backwards compatibility with the previous API.
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
    """Tabulated ionisation and recombination rate coefficients.

    Parameters
    ----------
    data:
        Mapping of species name to a dictionary containing temperature
        points and ionisation/recombination coefficients.  When a plain
        list is supplied (the legacy behaviour) the data are associated
        with a default species named ``"base"``.
    """

    def __init__(self, data: dict[str, dict[str, list[float]]]):
        self.data = data

    @classmethod
    def from_csv(cls, path: str | Path | dict[str, str | Path]) -> "RateTable":
        """Create a rate table from one or more CSV files.

        ``path`` may be either a single file (legacy behaviour) or a
        mapping of species name to file.  Each CSV is expected to contain
        three columns: temperature, ionisation rate and recombination
        rate.
        """

        if isinstance(path, (str, Path)):
            files = {"base": path}
        else:
            files = path

        out: dict[str, dict[str, list[float]]] = {}
        for sp, p in files.items():
            data = np.loadtxt(p, delimiter=",", skiprows=1)
            out[sp] = {
                "T": [row[0] for row in data],
                "k_ion": [row[1] for row in data],
                "k_rec": [row[2] for row in data],
            }
        return cls(out)

    # ------------------------------------------------------------------
    def ion_rate(self, T, species: str = "base") -> list[float]:
        table = self.data[species]
        return np.interp(
            T, table["T"], table["k_ion"], left=table["k_ion"][0], right=table["k_ion"][-1]
        )

    def rec_rate(self, T, species: str = "base") -> list[float]:
        table = self.data[species]
        return np.interp(
            T, table["T"], table["k_rec"], left=table["k_rec"][0], right=table["k_rec"][-1]
        )


class RateEquations:
    """Simple multi-species collisional–radiative model.

    The solver tracks the population ``n[i]`` of each charge state ``i``
    starting with the neutral at ``i=0``.  A single set of ionisation and
    recombination coefficients is applied between adjacent charge states.
    The ``species`` argument selects which table from :class:`RateTable`
    to use; by default the legacy ``"base"`` species is employed.
    """

    def __init__(self, rates: RateTable, levels: int = 2, species: str = "base"):
        self.rates = rates
        self.levels = levels
        self.species = species

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

        k_i = self.rates.ion_rate([T], self.species)[0]
        k_r = self.rates.rec_rate([T], self.species)[0]
        ne = self.electron_density(n)

        dn = np.zeros_like(n)
        for i in range(self.levels):
            ion_in = k_i * ne * n[i - 1] if i > 0 else 0.0
            rec_in = k_r * ne * n[i + 1] if i < self.levels - 1 else 0.0
            ion_out = k_i * ne * n[i] if i < self.levels - 1 else 0.0
            rec_out = k_r * ne * n[i] if i > 0 else 0.0
            dn[i] = ion_in + rec_in - ion_out - rec_out

        return dn

    def step(
        self,
        n: list[float],
        T: float,
        dt: float,
        sources: list[float] | None = None,
    ) -> list[float]:
        """Advance populations ``n`` by a single explicit Euler step.

        Parameters
        ----------
        n:
            Populations for each charge state starting from the neutral
            species.
        T:
            Electron temperature in eV.
        dt:
            Time step in seconds.
        sources:
            Optional source term for each charge state expressed as a rate of
            change in population (``dn/dt``).  This allows injection of
            neutrals from wall ablation or other external mechanisms.
        """

        dn = self.rhs(n, T)
        if sources is None:
            sources = [0.0] * self.levels
        elif len(sources) != self.levels:
            raise ValueError("sources must provide one entry per charge state")
        dn = [dni + src for dni, src in zip(dn, sources)]
        return [ni + dt * dni for ni, dni in zip(n, dn)]


class ImpurityModel:
    """Convenience wrapper for evolving impurity charge states.

    The class simply combines :class:`RateTable` and :class:`RateEquations`
    for a named species.  It exposes ``step``/``mean_charge`` helpers used
    by the Hall-MHD solver tests which need a lightweight impurity model.
    """

    def __init__(self, species: str, rates: RateTable, charge_states: int = 2) -> None:
        self.species = species
        self.equations = RateEquations(rates, levels=charge_states, species=species)

    def step(
        self,
        n: list[float],
        T: float,
        dt: float,
        sources: list[float] | None = None,
    ) -> list[float]:
        return self.equations.step(n, T, dt, sources)

    def mean_charge(self, n: list[float]) -> float:
        return self.equations.mean_charge(n)

    def electron_density(self, n: list[float]) -> float:
        return self.equations.electron_density(n)

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
        sources: dict[str, list[float]] | None = None,
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
        sources:
            Optional mapping of species name to per-cell source terms
            expressed as ``dn/dt``.
        """

        wall_ablation = wall_ablation or {}
        sources = sources or {}
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
            if sp in sources:
                src = sources[sp]
                if len(src) != cells:
                    raise ValueError(
                        f"sources for species {sp} must match number of cells"
                    )
                updated[sp] = [v + dt * s for v, s in zip(updated[sp], src)]
        return updated


__all__ = [
    "RateTable",
    "RateEquations",
    "ImpurityModel",
    "MultiSpeciesTransport",
]

