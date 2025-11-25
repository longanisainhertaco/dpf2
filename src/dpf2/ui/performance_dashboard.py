"""Dashboards for regime checks, throughput metrics and cross-run comparisons."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


@dataclass
class RegimeSummary:
    """Aggregate view of a sweep's dimensionless regime coverage."""

    S_min: float
    S_max: float
    best_yield: float
    pinch_window: tuple[float, float]


@dataclass
class ThroughputSummary:
    """Simple throughput metrics for a run or sweep point."""

    yield_per_hour: float
    wall_plug_efficiency: float
    runtime_s: float


@dataclass
class ComparisonDashboard:
    """Collection of sweep comparisons for UI or CLI export."""

    regimes: Dict[str, RegimeSummary] = field(default_factory=dict)
    throughput: Dict[str, ThroughputSummary] = field(default_factory=dict)
    overlays: List[Dict[str, float]] = field(default_factory=list)

    @classmethod
    def from_summaries(cls, labels: Sequence[str], summary_paths: Sequence[str | Path]) -> "ComparisonDashboard":
        """Construct a dashboard from multiple ``summary.json`` files."""

        dashboard = cls()
        for label, path in zip(labels, summary_paths):
            p = Path(path)
            if not p.exists():
                continue
            data = json.loads(p.read_text())
            if not isinstance(data, Mapping):
                continue

            yields = [entry.get("yield", 0.0) for entry in data.values()]
            pinch_times = [entry.get("pinch_time", 0.0) for entry in data.values()]
            s_vals = [entry.get("S", 0.0) for entry in data.values()]
            throughput_vals = [entry.get("yield_per_hour", 0.0) for entry in data.values()]
            wall_plug_vals = [entry.get("wall_plug_efficiency", 0.0) for entry in data.values()]
            runtimes = [entry.get("runtime_s", 0.0) for entry in data.values()]

            if s_vals:
                dashboard.regimes[label] = RegimeSummary(
                    S_min=min(s_vals),
                    S_max=max(s_vals),
                    best_yield=max(yields) if yields else 0.0,
                    pinch_window=(min(pinch_times) if pinch_times else 0.0, max(pinch_times) if pinch_times else 0.0),
                )
            if throughput_vals:
                dashboard.throughput[label] = ThroughputSummary(
                    yield_per_hour=max(throughput_vals),
                    wall_plug_efficiency=max(wall_plug_vals) if wall_plug_vals else 0.0,
                    runtime_s=sum(runtimes) / len(runtimes) if runtimes else 0.0,
                )
            for val, entry in data.items():
                dashboard.overlays.append(
                    {
                        "run": label,
                        "value": float(val),
                        "yield": entry.get("yield", 0.0),
                        "S": entry.get("S", 0.0),
                        "yield_per_hour": entry.get("yield_per_hour", 0.0),
                        "wall_plug_efficiency": entry.get("wall_plug_efficiency", 0.0),
                    }
                )
        return dashboard

    def to_json(self, path: str | Path) -> Path:
        """Serialise the dashboard to JSON for UI consumption."""

        payload = {
            "regimes": {k: vars(v) for k, v in self.regimes.items()},
            "throughput": {k: vars(v) for k, v in self.throughput.items()},
            "overlays": list(self.overlays),
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        return out


def aggregate_comparisons(run_dirs: Iterable[str | Path], name_template: str = "run_{idx}") -> ComparisonDashboard:
    """Convenience helper to build a comparison dashboard from sweep outputs."""

    labels: list[str] = []
    summaries: list[Path] = []
    for idx, run_dir in enumerate(run_dirs):
        labels.append(name_template.format(idx=idx))
        summaries.append(Path(run_dir) / "summary.json")
    return ComparisonDashboard.from_summaries(labels, summaries)


__all__ = [
    "RegimeSummary",
    "ThroughputSummary",
    "ComparisonDashboard",
    "aggregate_comparisons",
]
