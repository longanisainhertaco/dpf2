from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Any

import numpy as np

from ..uq.analysis import sobol_indices
from ..uq.calibration import bayes_factor, posterior_summary


@dataclass
class UQPanelUI:
    """Lightweight front-end for basic UQ analyses."""

    def sobol_from_arrays(
        self, samples: Sequence[Sequence[float]], values: Sequence[float], names: Sequence[str]
    ) -> dict[str, float]:
        return sobol_indices(samples, values, names)

    def sobol_from_file(self, path: str | Path, names: Sequence[str]) -> dict[str, float]:
        data = np.load(path)
        samples = data["samples"]
        values = data["values"]
        return sobol_indices(samples, values, names)

    def summarise(self, path: str | Path) -> dict[str, Any]:
        data = np.load(path)
        samples = {name: data[name] for name in data.files}
        return posterior_summary(samples)

    def compare_models(self, logz_a: float, logz_b: float) -> float:
        return bayes_factor(logz_a, logz_b)


def _main() -> None:  # pragma: no cover - CLI helper
    import argparse
    parser = argparse.ArgumentParser(description="Run simple UQ analyses")
    parser.add_argument("file", help="NPZ file containing posterior samples")
    args = parser.parse_args()
    ui = UQPanelUI()
    stats = ui.summarise(Path(args.file))
    for name, s in stats.items():
        print(f"{name}: mean={s['mean']:.3f} std={s['std']:.3f}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    _main()

