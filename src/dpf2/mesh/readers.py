from __future__ import annotations

"""Mesh import utilities for common CAD formats."""

from pathlib import Path
from typing import Dict, Any, List

try:  # pragma: no cover - optional dependency
    import meshio  # type: ignore
except Exception:  # pragma: no cover - meshio may not be installed
    meshio = None


def _ensure_meshio() -> None:
    if meshio is None:  # pragma: no cover - import guard
        raise RuntimeError("meshio is required for mesh import")


def read_stl(path: Path) -> Dict[str, Any]:
    """Read an STL surface mesh from ``path``."""
    _ensure_meshio()
    m = meshio.read(Path(path))
    nodes = m.points.tolist()
    elements: List[List[int]] = []
    for block in m.cells:
        if block.type == "triangle":
            elements.extend(block.data.tolist())
    return {"nodes": nodes, "elements": elements}


def read_vtk(path: Path) -> Dict[str, Any]:
    """Read an unstructured VTK mesh from ``path``."""
    _ensure_meshio()
    m = meshio.read(Path(path))
    elements: List[List[int]] = []
    for block in m.cells:
        elements.extend(block.data.tolist())
    return {"nodes": m.points.tolist(), "elements": elements}
