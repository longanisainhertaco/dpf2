from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

try:  # pragma: no cover - optional dependency
    import meshio  # type: ignore
except Exception:  # pragma: no cover - meshio may not be installed
    meshio = None


def _parse_step_like(lines: List[str]) -> Dict[str, Any]:
    """Parse a tiny subset of STEP/IGES style geometry.

    The parser understands lines beginning with ``NODE`` to define points and
    ``TRI`` to define triangular faces.  Indices in ``TRI`` statements are
    one-based to mimic common CAD conventions.
    """

    nodes: List[List[float]] = []
    elements: List[List[int]] = []
    for ln in lines:
        parts = ln.split()
        if not parts:
            continue
        tag, *rest = parts
        if tag.upper() == "NODE" and len(rest) == 3:
            nodes.append([float(v) for v in rest])
        elif tag.upper() == "TRI" and len(rest) == 3:
            elements.append([int(v) for v in rest])
    return {"nodes": nodes, "elements": elements}


def load_cad_geometry(path: Path) -> Dict[str, Any]:
    """Load a minimal CAD style geometry description from ``path``.

    The loader understands a few simple formats used in tests:

    * ``.json`` files containing ``nodes`` and ``elements`` lists.
    * ``.step``/``.stp`` and ``.iges``/``.igs`` files.  If :mod:`meshio` is
      available it is used to parse these files.  Otherwise a tiny custom text
      format is supported where each line is either ``NODE x y z`` or
      ``TRI i j k``.
    """

    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".json":
        return json.loads(p.read_text())
    if suffix in {".step", ".stp", ".iges", ".igs"}:
        if meshio is not None:
            try:  # pragma: no cover - exercised when meshio is available
                m = meshio.read(p)
                elements: List[List[int]] = []
                for block in m.cells:
                    if block.type in {"triangle", "quad"}:
                        elements.extend(block.data.tolist())
                if not elements and m.cells:
                    elements = m.cells[0].data.tolist()
                return {"nodes": m.points.tolist(), "elements": elements}
            except Exception:
                pass
        # fallback simple text representation
        lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
        return _parse_step_like(lines)
    raise ValueError(f"Unsupported CAD format: {suffix}")


def load_unstructured_mesh(path: Path) -> Dict[str, Any]:
    """Load a very small subset of an unstructured mesh format.

    If ``path`` has a ``.json`` suffix the same format as :func:`load_cad_geometry`
    is assumed.  Otherwise a simple text format is parsed where the first line is
    the number of nodes followed by that many lines of ``x y z`` coordinates.  The
    next line gives the number of elements followed by index triples.
    """
    p = Path(path)
    if p.suffix.lower() == ".json":
        return json.loads(p.read_text())

    lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    n_nodes = int(lines[0])
    nodes = [list(map(float, ln.split())) for ln in lines[1 : 1 + n_nodes]]
    n_elem = int(lines[1 + n_nodes])
    elements = [list(map(int, ln.split())) for ln in lines[2 + n_nodes : 2 + n_nodes + n_elem]]
    return {"nodes": nodes, "elements": elements}


def load_axisymmetric_mesh(path: Path) -> Dict[str, Any]:
    """Load a minimal axisymmetric mesh description.

    The file format is intentionally lightweight for tests.  If the file has a
    ``.json`` suffix it should contain ``r`` and ``z`` coordinate arrays.  A
    plain text format is also accepted where the first line lists ``nr nz``
    followed by ``nr`` lines of radial coordinates and ``nz`` lines of axial
    coordinates.  The result is returned as a dictionary with ``r`` and ``z``
    entries.
    """

    p = Path(path)
    if p.suffix.lower() == ".json":
        return json.loads(p.read_text())

    lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    nr, nz = map(int, lines[0].split())
    r = [float(lines[i + 1]) for i in range(nr)]
    z = [float(lines[1 + nr + i]) for i in range(nz)]
    return {"r": r, "z": z}
