"""Helpers for visualising and exporting regime diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from ..diagnostics.regime_panel import RegimePanel


@dataclass
class RegimeDashboard:
    """Simple interface around :class:`RegimePanel` for UI/CLI use."""

    panel: RegimePanel
    output_dir: Path = Path("synthetic_diagnostics/regime")

    def render(self) -> Path:
        """Render a plot of the regime history."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.panel.plot(self.output_dir / "regime.png")

    def export_csv(self, path: str | Path | None = None) -> Path:
        """Write logged history to CSV, returning the file path."""
        if path is None:
            path = self.output_dir / "regime.csv"
        else:
            path = Path(path)
        return self.panel.to_csv(path)


def _main() -> None:  # pragma: no cover - CLI helper
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Regime dashboard utilities")
    parser.add_argument("--history", help="JSON history file to load", default=None)
    parser.add_argument("--csv", help="CSV output path", default="regime.csv")
    args = parser.parse_args()

    panel = RegimePanel(L=1.0)
    if args.history:
        with open(args.history, "r", encoding="utf-8") as fh:
            panel.history = json.load(fh)
    panel.to_csv(args.csv)
    print(f"Regime history exported to {args.csv}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    _main()
