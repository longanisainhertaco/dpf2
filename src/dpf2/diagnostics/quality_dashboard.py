from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class QualityDashboard:
    """Collect and persist basic quality metrics for simulation steps."""

    output_dir: Path = Path("synthetic_diagnostics/quality")
    min_cfl: float | None = None
    min_lambda_D: float | None = None
    min_ppc: float | None = None
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
    ) -> None:
        """Record a step's metrics and emit warnings if thresholds violated."""
        entry = {
            "step": step,
            "dt": dt,
            "cell_size": cell_size,
            "ppc": ppc,
            "cfl": cfl,
            "lambda_D": lambda_D,
        }
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
