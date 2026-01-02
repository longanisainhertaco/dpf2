"""Simple CLI/GUI helpers for running verification problems."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..diagnostics.quality_dashboard import QualityDashboard
from ..verification import VerificationPanel


@dataclass
class VerificationPanelUI:
    """Launch verification tests and report pass/fail status."""

    output_file: Path = Path("synthetic_diagnostics/verification.h5")
    quality: QualityDashboard | None = None

    def run_all(self) -> dict[str, Any]:
        panel = VerificationPanel(self.output_file, self.quality)
        return panel.run_all()

    def summarize(self) -> str:
        """Return a compact human-readable summary of observed orders."""

        results = self.run_all()
        lines = ["Numerics verification results:"]
        for key, res in results.items():
            obs = res.get("observed_order", [])
            order = obs[-1] if obs else 0.0
            lines.append(f"- {key}: observed order {order:.2f}")
        return "\n".join(lines)


def _main() -> None:  # pragma: no cover - CLI helper
    import argparse
    parser = argparse.ArgumentParser(description="Run verification problems")
    parser.add_argument("--output", default="synthetic_diagnostics/verification.h5")
    args = parser.parse_args()
    ui = VerificationPanelUI(Path(args.output))
    ui.run_all()
    print(f"Results written to {args.output}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    _main()
