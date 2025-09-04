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

    def log(
        self,
        step: int,
        dt: float,
        cell_size: float,
        ppc: float,
        cfl: float,
        lambda_D: float,
        divergence_error: float = 0.0,
        energy_drift: float = 0.0,
    ) -> None:
        """Record a step's metrics and emit warnings if thresholds violated."""
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
        lambdas = [h["lambda_D"] for h in self.history]
        cells = [h["cell_size"] for h in self.history]

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.plot(steps, dts, label="Δt")
        if self.max_dt is not None:
            ax1.axhspan(0, self.max_dt, color="lightgreen", alpha=0.3)
            ax1.axhline(self.max_dt, color="red", linestyle="--", label="Δt limit")
        ax1.set_ylabel("Δt")
        ax1.legend()

        ax2.plot(steps, lambdas, label="λ_D")
        ax2.plot(steps, cells, label="cell size", linestyle="--")
        arr_steps = np.array(steps)
        arr_cells = np.array(cells)
        arr_lambda = np.array(lambdas)
        ax2.fill_between(
            arr_steps,
            arr_cells,
            arr_lambda,
            where=arr_lambda >= arr_cells,
            color="lightgreen",
            alpha=0.3,
        )
        ax2.fill_between(
            arr_steps,
            arr_lambda,
            arr_cells,
            where=arr_lambda < arr_cells,
            color="red",
            alpha=0.3,
        )
        ax2.set_ylabel("λ_D")
        ax2.set_xlabel("step")
        ax2.legend()

        fig.tight_layout()
        fig.savefig(self.output_dir / "stability.png")
        plt.close(fig)
