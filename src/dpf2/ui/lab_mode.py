from __future__ import annotations

"""Lightweight lab-mode UI helpers."""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from ..core.config import DPFConfig
from ..core.simulation import DPFSimulation
from ..cli.lab import write_manifest
from ..io.manifest import write_batch_manifest


@dataclass
class LabModeUI:
    """Simple interface for enabling jitter and exporting results."""

    jitter_enabled: bool = False

    def toggle_jitter(self, enable: bool) -> None:
        """Enable or disable stochastic jitter."""
        self.jitter_enabled = bool(enable)

    def export_results(self, results: Dict[str, Any], path: str | Path) -> Path:
        """Export ``results`` to ``path`` as JSON."""
        p = Path(path)
        p.write_text(json.dumps(results, indent=2))
        return p

    def run_shot_series(
        self,
        config: DPFConfig,
        shots: int,
        output_dir: str | Path,
        *,
        base_seed: int | None = None,
        datasets: Dict[str, Dict[str, Dict[str, object]]] | None = None,
    ) -> List[Dict[str, Any]]:
        """Execute a jittered shot series and persist manifests/checkpoints.

        The helper respects the ``jitter_enabled`` flag. When jitter is
        disabled the input configuration is reused for every shot while still
        emitting per-shot manifests for provenance.
        """

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(base_seed)
        shots_meta: List[Dict[str, Any]] = []

        for idx in range(shots):
            py_seed = rng.integers(0, 2**32 - 1, dtype="uint32").item()
            np_seed = rng.integers(0, 2**32 - 1, dtype="uint32").item()
            random.seed(int(py_seed))
            shot_rng = np.random.default_rng(int(np_seed))

            cfg = config
            jitter_report: Dict[str, float] = {}
            if self.jitter_enabled:
                cfg, jitter_report = config.apply_jitter(shot_rng)

            sim = DPFSimulation(cfg)
            shot_dir = out / f"shot_{idx:03d}"
            shot_dir.mkdir(parents=True, exist_ok=True)
            times, currents, voltages = sim.run(
                output_dir=shot_dir, seeds={"python": int(py_seed), "numpy": int(np_seed)}
            )

            write_manifest(
                shot_dir,
                config_paths=None,
                config=config if not hasattr(config, "model_dump") else config.model_dump(),
                seeds={"python": int(py_seed), "numpy": int(np_seed)},
                datasets=datasets,
            )

            shot_meta = {
                "shot": idx,
                "seeds": {"python": int(py_seed), "numpy": int(np_seed)},
                "jitter": jitter_report,
                "output": str(shot_dir),
                "peak_current": max(currents) if currents else 0.0,
                "pinch_time": times[currents.index(max(currents))] if currents else 0.0,
            }
            shots_meta.append(shot_meta)

            self._write_checkpoint(shot_dir, times, currents, voltages)

        write_batch_manifest(out, shots_meta)
        return shots_meta

    def _write_checkpoint(
        self,
        run_dir: Path,
        times: Iterable[float],
        currents: Iterable[float],
        voltages: Iterable[float],
    ) -> None:
        """Persist a lightweight HDF5 checkpoint when h5py is available."""

        try:
            import h5py  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            return

        run_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(run_dir / "lab_checkpoint.h5", "w") as h5:
            h5.create_dataset("time", data=list(times))
            h5.create_dataset("current", data=list(currents))
            h5.create_dataset("voltage", data=list(voltages))


__all__ = ["LabModeUI"]
