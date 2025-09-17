"""Simple CLI/CLI helpers for running verification problems."""

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
        results = {
            "brio_wu": panel.run_brio_wu(),
            "orszag_tang": panel.run_orszag_tang(),
            "mms": panel.run_mms(),
        }
        return results


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
