"""Regime diagnostic panel tracking dimensionless parameters over time."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import logging
import csv
import json
from typing import Dict
import numpy as np

from .plasma import (
    lundquist_number,
    plasma_beta,
    alfven_mach_number,
    magnetic_reynolds_number,
)
from .quality_dashboard import QualityDashboard

logger = logging.getLogger(__name__)

try:  # pragma: no cover - SciPy optional
    from scipy.constants import e, m_e
except Exception:  # pragma: no cover - fallback values
    e = 1.602176634e-19
    m_e = 9.1093837015e-31


@dataclass
class RegimePanel:
    """Compute and log dimensionless plasma regime parameters."""

    L: float
    quality: QualityDashboard | None = None
    thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "S": 1.0,
            "beta": 1.0,
            "M_A": 1.0,
            "R_m": 1.0,
            "K_n": 0.1,
            "omega_ce_tau_e": 1.0,
            "di_over_L": 0.01,
        }
    )
    history: list[Dict[str, float]] = field(default_factory=list)
    energy_history: list[Dict[str, float]] = field(default_factory=list)

    def log(
        self,
        step: int,
        n: float,
        T: float,
        B: float,
        v: float,
        eta: float,
        mfp: float,
        tau_e: float,
        magnetic_energy: float | None = None,
        kinetic_energy: float | None = None,
        radiation_energy: float | None = None,
        loss_energy: float | None = None,
    ) -> Dict[str, float]:
        """Compute parameters for a simulation step.

        Parameters are specified in SI units. ``n`` is number density, ``T``
        the temperature, ``B`` the magnetic-field magnitude, ``v`` a
        characteristic flow speed, ``eta`` the resistivity, ``mfp`` the mean
        free path and ``tau_e`` the electron collision time.
        """

        sigma = 1.0 / eta if eta > 0 else float("inf")
        S = lundquist_number(B, n, self.L, sigma)
        beta = plasma_beta(n, T, B)
        M_A = alfven_mach_number(v, B, n)
        R_m = magnetic_reynolds_number(v, self.L, sigma)
        K_n = mfp / self.L
        omega_ce_tau_e = (e * B / m_e) * tau_e
        di_over_L = 0.0
        try:
            # Ion inertial length d_i = sqrt(m_i / (mu0 n q^2))
            # Use proton mass as a proxy; this is sufficient for gating logic.
            import scipy.constants as const

            di = np.sqrt(const.m_p / (const.mu_0 * (const.e ** 2) * n)) if n > 0 else 0.0
            di_over_L = di / self.L if self.L > 0 else 0.0
        except Exception:  # pragma: no cover - optional SciPy
            di_over_L = 0.0

        entry = {
            "step": step,
            "S": float(S),
            "beta": float(beta),
            "M_A": float(M_A),
            "R_m": float(R_m),
            "K_n": float(K_n),
            "omega_ce_tau_e": float(omega_ce_tau_e),
            "di_over_L": float(di_over_L),
        }

        if any(val is not None for val in (magnetic_energy, kinetic_energy, radiation_energy, loss_energy)):
            mag = float(magnetic_energy) if magnetic_energy is not None else 0.0
            kin = float(kinetic_energy) if kinetic_energy is not None else 0.0
            rad = float(radiation_energy) if radiation_energy is not None else 0.0
            loss = float(loss_energy) if loss_energy is not None else 0.0
            total = mag + kin + rad
            total_non_negative = total if total > 0 else 1.0
            energy_partition = {
                "magnetic_energy": mag,
                "kinetic_energy": kin,
                "radiation_energy": rad,
                "loss_energy": loss,
                "total_energy": total,
                "net_energy": total - loss,
                "magnetic_fraction": mag / total_non_negative,
                "kinetic_fraction": kin / total_non_negative,
                "radiation_fraction": rad / total_non_negative,
                "loss_fraction": loss / total_non_negative,
                "balance_residual": (total - loss) / total_non_negative,
            }
            entry["energy_partition"] = energy_partition
            self.energy_history.append({"step": step, **energy_partition})
        self.history.append(entry)

        if self.quality is not None:
            self.quality.log_regime(
                step,
                entry["S"],
                entry["beta"],
                entry["M_A"],
                entry["R_m"],
                entry["K_n"],
                entry["omega_ce_tau_e"],
                entry["di_over_L"],
            )

        violations: Dict[str, bool] = {}
        for key, limit in self.thresholds.items():
            val = entry[key]
            if key in {"beta", "M_A", "K_n"}:
                violations[key] = val > limit
            else:
                violations[key] = val < limit
            if violations[key]:
                logger.warning("Regime parameter %s out of bounds: %.3g", key, val)
        entry["violations"] = violations

        return entry

    # ------------------------------------------------------------------
    def plot(self, path: str | Path) -> Path:
        """Render the logged regime parameters versus time.

        Parameters
        ----------
        path:
            Destination for the generated plot.  Parent directories are
            created automatically.  The plot contains six subplots for the
            Lundquist number ``S``, plasma beta ``β``, Alfvén Mach number
            ``M_A``, magnetic Reynolds number ``R_m``, Knudsen number ``K_n``
            and the magnetisation parameter ``ω_ce τ_e``.

        Returns
        -------
        Path
            The path where the plot was written.  If matplotlib is unavailable
            the file is not created but the path is still returned.
        """

        try:  # pragma: no cover - matplotlib optional
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:  # pragma: no cover - gracefully handle missing dep
            return Path(path)

        if not self.history:
            raise ValueError("no regime data logged")

        steps = [h["step"] for h in self.history]
        metrics = [
            ("S", "Lundquist"),
            ("beta", "beta"),
            ("M_A", "M_A"),
            ("R_m", "R_m"),
            ("K_n", "K_n"),
            ("omega_ce_tau_e", "ω_ce τ_e"),
            ("di_over_L", "d_i/L"),
        ]

        fig, axes = plt.subplots(4, 2, sharex=True)
        for ax, (key, label) in zip(axes.flat, metrics):
            ax.plot(steps, [h[key] for h in self.history])
            ax.set_ylabel(label)
        axes[-1, 0].set_xlabel("step")
        axes[-1, 1].set_xlabel("step")
        fig.tight_layout()

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p)
        plt.close(fig)
        return p

    # ------------------------------------------------------------------
    def plot_energy_partitions(self, path: str | Path) -> Path:
        """Render tracked energy partitions over time."""

        try:  # pragma: no cover - matplotlib optional
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:  # pragma: no cover - matplotlib optional
            return Path(path)

        if not self.energy_history:
            raise ValueError("no energy data logged")

        steps = [h["step"] for h in self.energy_history]
        mag = [h["magnetic_energy"] for h in self.energy_history]
        kin = [h["kinetic_energy"] for h in self.energy_history]
        rad = [h["radiation_energy"] for h in self.energy_history]
        loss = [h["loss_energy"] for h in self.energy_history]

        fig, ax = plt.subplots()
        ax.plot(steps, mag, label="magnetic")
        ax.plot(steps, kin, label="kinetic")
        ax.plot(steps, rad, label="radiation")
        ax.plot(steps, loss, label="losses")
        ax.set_xlabel("step")
        ax.set_ylabel("energy (J)")
        ax.legend()

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(p)
        plt.close(fig)
        return p

    # ------------------------------------------------------------------
    def to_csv(self, path: str | Path) -> Path:
        """Export the logged regime history to ``path`` in CSV format."""

        if not self.history:
            raise ValueError("no regime data logged")

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Prepare field names: all keys except the nested violation map which is
        # expanded into ``<metric>_violated`` columns.
        base_fields = [k for k in self.history[0] if k not in {"violations", "energy_partition"}]
        violation_fields = [f"{k}_violated" for k in self.thresholds]
        energy_fields = []
        if any("energy_partition" in h for h in self.history):
            sample = next(h for h in self.history if "energy_partition" in h)
            energy_fields = list(sample["energy_partition"].keys())
        fieldnames = base_fields + violation_fields + energy_fields

        with open(p, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.history:
                out = {k: row.get(k) for k in base_fields}
                viol = row.get("violations", {})
                for key in self.thresholds:
                    out[f"{key}_violated"] = bool(viol.get(key, False))
                if "energy_partition" in row:
                    out.update(row["energy_partition"])
                writer.writerow(out)

        return p

    # ------------------------------------------------------------------
    def to_json(self, path: str | Path) -> Path:
        """Export the full regime history, including energy partitions, to JSON."""

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(self.history, fh, indent=2)
        return p

    # ------------------------------------------------------------------
    def dashboard(self, plot_path: str | Path | None = None) -> Dict[str, object]:
        """Return a dashboard summary for dimensionless regime tracking.

        When ``plot_path`` is provided, :meth:`plot` is invoked and the saved path
        is included in the returned dictionary.
        """

        latest = self.history[-1] if self.history else {}
        dashboard: Dict[str, object] = {
            "latest": latest,
            "count": len(self.history),
            "thresholds": self.thresholds,
        }
        if self.energy_history:
            dashboard["energy_partition"] = self.energy_history[-1]
        if plot_path is not None:
            dashboard["plot_path"] = str(self.plot(plot_path))
        return dashboard
