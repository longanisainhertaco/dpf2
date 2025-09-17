from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QualityDashboard:
    """Collect and persist basic quality metrics for simulation steps."""

    output_dir: Path = Path("synthetic_diagnostics/quality")
    min_cfl: float | None = None
    min_lambda_D: float | None = None
    min_ppc: float | None = None
    max_dt: float | None = None
    abort_on_violation: bool = False
    history: list[dict[str, float]] = field(default_factory=list)
    max_l1_error: float | None = None
    max_divB_norm: float | None = None
    max_energy_drift: float | None = None
    numerics_history: list[dict[str, float]] = field(default_factory=list)

    min_S: float | None = None
    max_beta: float | None = None
    max_M_A: float | None = None
    min_R_m: float | None = None
    max_K_n: float | None = None
    min_omega_ce_tau_e: float | None = None
    regime_history: list[dict[str, float]] = field(default_factory=list)

    max_l1_error: float | None = None
    max_divB_norm: float | None = None

    max_divE_norm: float | None = None

    max_energy_drift: float | None = None
    numerics_history: list[dict[str, float]] = field(default_factory=list)


    def log(
        self,
        step: int,
        dt: float,
        cell_size: float,
        ppc: float,
        cfl: float,
        lambda_D: float,
        amr_level: int | None = None,
        lower_hybrid_power: float | None = None,
        lower_hybrid_phase_velocity: float | None = None,
        plasma_impedance: float | None = None,
        impedance_ratio: float | None = None,
        divergence_error: float = 0.0,
        energy_drift: float = 0.0,
        hall_active: bool | None = None,
        electron_inertia_active: bool | None = None,
        wce_tau_e: float | None = None,
        di_over_L: float | None = None,
        hall_threshold: float | None = None,
        ei_threshold: float | None = None,

    ) -> None:
        """Record a step's metrics and emit warnings if thresholds violated.

        Parameters
        ----------
        amr_level:
            Current refinement level, if adaptive mesh refinement is enabled.
            When provided the level is stored and visualised in the stability
            plot, allowing users to correlate mesh changes with other quality
            metrics.
        """
        entry = {
            "step": step,
            "dt": dt,
            "cell_size": cell_size,
            "ppc": ppc,
            "cfl": cfl,
            "lambda_D": lambda_D,
            "divergence_error": divergence_error,
            "energy_drift": energy_drift,
        }
        if amr_level is not None:
            entry["amr_level"] = amr_level

        if lower_hybrid_power is not None:
            entry["lower_hybrid_power"] = lower_hybrid_power
        if lower_hybrid_phase_velocity is not None:
            entry["lower_hybrid_phase_velocity"] = lower_hybrid_phase_velocity
        if plasma_impedance is not None:
            entry["plasma_impedance"] = plasma_impedance
        if impedance_ratio is not None:
            entry["impedance_ratio"] = impedance_ratio
        if hall_active is not None:
            entry["hall_active"] = hall_active
        if electron_inertia_active is not None:
            entry["electron_inertia_active"] = electron_inertia_active
        if wce_tau_e is not None:
            entry["wce_tau_e"] = wce_tau_e
        if di_over_L is not None:
            entry["di_over_L"] = di_over_L
        if hall_threshold is not None:
            entry["hall_threshold"] = hall_threshold
        if ei_threshold is not None:
            entry["ei_threshold"] = ei_threshold

        dt_violation = self.max_dt is not None and dt > self.max_dt
        lambda_violation = lambda_D < cell_size
        entry["dt_violation"] = dt_violation
        entry["lambda_D_violation"] = lambda_violation

        self.history.append(entry)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "dashboard.json", "w", encoding="utf-8") as fh:
            json.dump(self.history, fh, indent=2)

        def _warn_or_abort(msg: str) -> None:
            logger.warning(msg)
            if self.abort_on_violation:
                raise RuntimeError(msg)

        if self.min_cfl is not None and cfl < self.min_cfl:
            _warn_or_abort(f"CFL below threshold: {cfl:g} < {self.min_cfl:g}")
        if self.min_lambda_D is not None and lambda_D < self.min_lambda_D:
            _warn_or_abort(
                f"Debye length below threshold: {lambda_D:g} < {self.min_lambda_D:g}"
            )
        if self.min_ppc is not None and ppc < self.min_ppc:
            _warn_or_abort(
                f"Particles per cell below threshold: {ppc:g} < {self.min_ppc:g}"
            )
        if dt_violation:
            _warn_or_abort(
                f"Time step above stability limit: {dt:g} > {self.max_dt:g}"
            )
        if lambda_violation:
            _warn_or_abort(
                f"Debye length under-resolved: {lambda_D:g} < {cell_size:g}"
            )

        self._update_plot()

    def log_regime(
        self,
        step: int,
        S: float,
        beta: float,
        M_A: float,
        R_m: float,
        K_n: float,
        omega_ce_tau_e: float,
    ) -> None:
        """Record dimensionless regime parameters and flag threshold violations."""

        entry = {
            "step": step,
            "S": S,
            "beta": beta,
            "M_A": M_A,
            "R_m": R_m,
            "K_n": K_n,
            "omega_ce_tau_e": omega_ce_tau_e,
        }

        def _warn_or_abort(msg: str) -> None:
            logger.warning(msg)
            if self.abort_on_violation:
                raise RuntimeError(msg)

        violations = {
            "S": self.min_S is not None and S < self.min_S,
            "beta": self.max_beta is not None and beta > self.max_beta,
            "M_A": self.max_M_A is not None and M_A > self.max_M_A,
            "R_m": self.min_R_m is not None and R_m < self.min_R_m,
            "K_n": self.max_K_n is not None and K_n > self.max_K_n,
            "omega_ce_tau_e": self.min_omega_ce_tau_e is not None
            and omega_ce_tau_e < self.min_omega_ce_tau_e,
        }

        for key, violated in violations.items():
            if not violated:
                continue
            if key == "S":
                _warn_or_abort(
                    f"Lundquist number below threshold: {S:g} < {self.min_S:g}"
                )
            elif key == "beta":
                _warn_or_abort(
                    f"Plasma beta above threshold: {beta:g} > {self.max_beta:g}"
                )
            elif key == "M_A":
                _warn_or_abort(
                    f"Alfvén Mach number above threshold: {M_A:g} > {self.max_M_A:g}"
                )
            elif key == "R_m":
                _warn_or_abort(
                    f"Magnetic Reynolds number below threshold: {R_m:g} < {self.min_R_m:g}"
                )
            elif key == "K_n":
                _warn_or_abort(
                    f"Knudsen number above threshold: {K_n:g} > {self.max_K_n:g}"
                )
            elif key == "omega_ce_tau_e":
                _warn_or_abort(
                    "Cyclotron frequency–collision time below threshold: "
                    f"{omega_ce_tau_e:g} < {self.min_omega_ce_tau_e:g}"
                )

        entry["violations"] = violations

        self.regime_history.append(entry)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "regime.json", "w", encoding="utf-8") as fh:
            json.dump(self.regime_history, fh, indent=2)

        self._update_regime_plot()

    # ------------------------------------------------------------------
    def evaluate_numerics(self, metrics: dict[str, float]) -> bool:
        """Check numerical diagnostics against configured thresholds."""

        self.numerics_history.append(metrics)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "numerics.json", "w", encoding="utf-8") as fh:
            json.dump(self.numerics_history, fh, indent=2)

        def _warn_or_abort(msg: str) -> None:
            logger.warning(msg)
            if self.abort_on_violation:
                raise RuntimeError(msg)

        ok = True
        l1 = metrics.get("l1_error")
        if self.max_l1_error is not None and l1 is not None and l1 > self.max_l1_error:
            _warn_or_abort(
                f"L1 error above threshold: {l1:g} > {self.max_l1_error:g}"
            )
            ok = False
        divB = metrics.get("divB_norm")
        if self.max_divB_norm is not None and divB is not None and divB > self.max_divB_norm:
            _warn_or_abort(
                f"∇·B norm above threshold: {divB:g} > {self.max_divB_norm:g}"
            )
            ok = False
        divE = metrics.get("divE_norm")
        if self.max_divE_norm is not None and divE is not None and divE > self.max_divE_norm:
            _warn_or_abort(
                f"∇·E norm above threshold: {divE:g} > {self.max_divE_norm:g}"
            )
            ok = False
        drift = metrics.get("energy_drift")
        if (
            self.max_energy_drift is not None
            and drift is not None
            and abs(drift) > self.max_energy_drift
        ):
            _warn_or_abort(
                f"Energy drift above threshold: {drift:g} > {self.max_energy_drift:g}"
            )
            ok = False

        return ok

    # ------------------------------------------------------------------
    def _update_plot(self) -> None:
        """Render a simple plot of stability metrics."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:  # pragma: no cover - matplotlib optional
            return

        if not self.history:
            return

        steps = [h["step"] for h in self.history]
        dts = [h["dt"] for h in self.history]
        cells = [h["cell_size"] for h in self.history]
        ppcs = [h["ppc"] for h in self.history]
        levels = [h.get("amr_level") for h in self.history]

        has_levels = any(l is not None for l in levels)
        if has_levels:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

        ax1.plot(steps, dts, label="Δt")
        if self.max_dt is not None:
            ax1.axhspan(0, self.max_dt, color="lightgreen", alpha=0.3)
            ax1.axhline(self.max_dt, color="red", linestyle="--", label="Δt limit")
        ax1.set_ylabel("Δt")
        ax1.legend()

        ax2.plot(steps, cells, label="Δx")
        ax2.set_ylabel("Δx")
        ax2.legend()

        ax3.plot(steps, ppcs, label="ppc")
        ax3.set_ylabel("ppc")

        if has_levels:
            ax4.step(steps, [l if l is not None else 0 for l in levels], where="post", label="AMR level")
            ax4.set_ylabel("level")
            ax4.set_xlabel("step")
            ax4.legend()
        else:
            ax3.set_xlabel("step")
            ax3.legend()

        fig.tight_layout()
        fig.savefig(self.output_dir / "stability.png")
        plt.close(fig)

    # ------------------------------------------------------------------
    def _update_regime_plot(self) -> None:
        """Render a plot of regime parameters over time."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:  # pragma: no cover - matplotlib optional
            return

        if not self.regime_history:
            return

        steps = [h["step"] for h in self.regime_history]
        metrics = [
            ("S", "Lundquist"),
            ("beta", "beta"),
            ("M_A", "M_A"),
            ("R_m", "R_m"),
            ("K_n", "K_n"),
            ("omega_ce_tau_e", "ω_ce τ_e"),
        ]
        fig, axes = plt.subplots(3, 2, sharex=True)
        for ax, (key, label) in zip(axes.flat, metrics):
            ax.plot(steps, [h[key] for h in self.regime_history])
            ax.set_ylabel(label)
        axes[2, 0].set_xlabel("step")
        axes[2, 1].set_xlabel("step")
        fig.tight_layout()
        fig.savefig(self.output_dir / "regime.png")
        plt.close(fig)

    # ------------------------------------------------------------------
    def convergence_sweep(self) -> None:
        """Generate a simple convergence summary of recorded metrics."""
        summary = self.convergence_dashboard()
        if not summary:
            logger.info("No history available for convergence sweep")
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "convergence.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        logger.info(
            "Convergence sweep written to %s", self.output_dir / "convergence.json"
        )

    def convergence_dashboard(self) -> dict[str, float]:
        """Return aggregate statistics for Δt, Δx and particles per cell."""
        if not self.history:
            return {}
        dts = np.array([h["dt"] for h in self.history])
        dxs = np.array([h["cell_size"] for h in self.history])
        ppcs = np.array([h["ppc"] for h in self.history])
        return {
            "dt_min": float(dts.min()),
            "dt_max": float(dts.max()),
            "dt_mean": float(dts.mean()),
            "dx_min": float(dxs.min()),
            "dx_max": float(dxs.max()),
            "dx_mean": float(dxs.mean()),
            "ppc_min": float(ppcs.min()),
            "ppc_max": float(ppcs.max()),
            "ppc_mean": float(ppcs.mean()),
        }


def _main() -> None:  # pragma: no cover - CLI helper
    import argparse

    parser = argparse.ArgumentParser(description="Quality dashboard utilities")
    parser.add_argument("--sweep", action="store_true", help="run convergence sweep and exit")
    args = parser.parse_args()
    dash = QualityDashboard()
    if args.sweep:
        dash.convergence_sweep()


if __name__ == "__main__":  # pragma: no cover - CLI entry
    _main()
