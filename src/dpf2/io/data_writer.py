"""Utilities for writing simulation output."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict

try:
    import h5py
except Exception:  # pragma: no cover - optional dependency
    h5py = None

try:
    import meshio
except Exception:  # pragma: no cover - optional dependency
    meshio = None

from ..mesh import Mesh2D


class DataWriter:
    """Write simulation data to disk with provenance metadata."""

    def __init__(self, output_dir: str, config: Dict[str, Any] | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = {
            "config_hash": self._hash_config(config),
            "git_commit": self._git_commit(),
        }

    @staticmethod
    def _hash_config(config: Dict[str, Any] | None) -> str:
        if config is None:
            return "unknown"
        data = json.dumps(config, sort_keys=True).encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def _git_commit() -> str:
        try:
            repo = Path(__file__).resolve().parents[2]
            return (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo)
                .decode()
                .strip()
            )
        except Exception:  # pragma: no cover - git not available
            return "unknown"

    def write_hdf5(self, data: Dict[str, float], time: float) -> Path:
        if h5py is None:
            raise RuntimeError("h5py is required for HDF5 output")
        fname = self.output_dir / f"data_{time:.6e}.h5"
        with h5py.File(fname, "w") as f:
            f.create_dataset("time", data=time)
            for key, value in data.items():
                f.create_dataset(key, data=value)
            f.create_dataset("metadata", data=json.dumps(self.metadata))
        return fname

    def write_json(self, data: Dict[str, float], time: float) -> Path:
        fname = self.output_dir / f"data_{time:.6e}.json"
        payload = {"time": time, "data": data, "metadata": self.metadata}
        with fname.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        return fname

    def write_vtk(self, mesh: Mesh2D, data: Dict[str, float], time: float) -> None:
        if meshio is None:
            raise RuntimeError("meshio is required for VTK output")
        fname = self.output_dir / f"data_{time:.6e}.vtu"
        points = [
            [cell.r_center, cell.z_center, 0.0] for cell in mesh.cells
        ]
        cells = {"vertex": [[i] for i in range(len(points))]}
        meshio.write_points_cells(fname, points, cells, point_data=data)
