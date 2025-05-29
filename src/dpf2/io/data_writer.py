"""Utilities for writing simulation output."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

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
    """Write simulation data to disk."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_hdf5(self, data: Dict[str, float], time: float) -> None:
        if h5py is None:
            raise RuntimeError("h5py is required for HDF5 output")
        fname = self.output_dir / f"data_{time:.6e}.h5"
        with h5py.File(fname, "w") as f:
            for key, value in data.items():
                f.create_dataset(key, data=value)

    def write_vtk(self, mesh: Mesh2D, data: Dict[str, float], time: float) -> None:
        if meshio is None:
            raise RuntimeError("meshio is required for VTK output")
        fname = self.output_dir / f"data_{time:.6e}.vtu"
        points = [
            [cell.r_center, cell.z_center, 0.0] for cell in mesh.cells
        ]
        cells = {"vertex": [[i] for i in range(len(points))]}
        meshio.write_points_cells(fname, points, cells, point_data=data)
