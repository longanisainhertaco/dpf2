from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Any, List, Sequence

try:  # pragma: no cover - optional dependency
    import meshio  # type: ignore
except Exception:  # pragma: no cover - meshio may not be installed
    meshio = None


def _parse_step_like(lines: List[str]) -> Dict[str, Any]:
    """Parse a tiny subset of STEP/IGES style geometry.

    The parser understands lines beginning with ``NODE`` to define points and
    ``TRI`` to define triangular faces.  ``TRI`` statements may include an
    optional fourth field specifying a material tag for the element.  Indices in
    ``TRI`` statements are one-based to mimic common CAD conventions.
    """

    nodes: List[List[float]] = []
    elements: List[List[int]] = []
    materials: List[Any] = []
    for ln in lines:
        parts = ln.split()
        if not parts:
            continue
        tag, *rest = parts
        if tag.upper() == "NODE" and len(rest) == 3:
            nodes.append([float(v) for v in rest])
        elif tag.upper() == "TRI" and len(rest) in {3, 4}:
            elements.append([int(v) for v in rest[:3]])
            if len(rest) == 4:
                mat = rest[3]
                try:
                    materials.append(int(mat))
                except ValueError:
                    materials.append(mat)
    out: Dict[str, Any] = {"nodes": nodes, "elements": elements}
    if materials:
        out["materials"] = materials
    return out


def load_cad_geometry(path: Path) -> Dict[str, Any]:
    """Load a minimal CAD style geometry description from ``path``.

    The loader understands a few simple formats used in tests:

    * ``.json`` files containing ``nodes`` and ``elements`` lists (and optional
      ``materials``).
    * ``.step``/``.stp`` and ``.iges``/``.igs`` files.  If :mod:`meshio` is
      available it is used to parse these files.  Cell based material tags are
      extracted when present.  Otherwise a tiny custom text format is supported
      where each line is either ``NODE x y z`` or ``TRI i j k [mat]``.
    """

    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".json":
        return json.loads(p.read_text())
    if suffix in {".step", ".stp", ".iges", ".igs", ".stl", ".vtk"}:
        if meshio is not None:
            try:  # pragma: no cover - exercised when meshio is available
                m = meshio.read(p)
                elements: List[List[int]] = []
                materials: List[Any] = []
                for idx, block in enumerate(m.cells):
                    if block.type in {"triangle", "quad"}:
                        data = block.data.tolist()
                        elements.extend(data)
                        tag = None
                        if m.cell_data:
                            for key in ("material", "gmsh:physical", "cell_tags"):
                                vals = m.cell_data.get(key)
                                if vals and len(vals) > idx:
                                    tag = vals[idx]
                                    break
                        if tag is not None:
                            materials.extend(
                                tag.tolist() if hasattr(tag, "tolist") else list(tag)
                            )
                if not elements and m.cells:
                    elements = m.cells[0].data.tolist()
                result: Dict[str, Any] = {"nodes": m.points.tolist(), "elements": elements}
                if materials:
                    result["materials"] = materials
                return result
            except Exception:
                pass
        # fallback simple text representation for a tiny subset of the formats
        lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
        if suffix in {".step", ".stp", ".iges", ".igs"}:
            return _parse_step_like(lines)
        if suffix == ".stl":
            nodes: List[List[float]] = []
            elements: List[List[int]] = []
            materials: List[Any] = []
            node_idx: Dict[tuple[float, float, float], int] = {}
            current: List[int] = []
            current_mat: Any | None = None
            for ln in lines:
                parts = ln.split()
                if not parts:
                    continue
                tag = parts[0].lower()
                if tag == "solid":
                    current_mat = parts[1] if len(parts) > 1 else None
                elif tag == "facet":
                    current = []
                elif tag == "vertex" and len(parts) == 4:
                    pt = tuple(float(v) for v in parts[1:])
                    idx = node_idx.get(pt)
                    if idx is None:
                        idx = len(nodes)
                        nodes.append(list(pt))
                        node_idx[pt] = idx
                    current.append(idx)
                elif tag == "endfacet" and len(current) == 3:
                    elements.append(current[:3])
                    if current_mat is not None:
                        materials.append(current_mat)
            result: Dict[str, Any] = {"nodes": nodes, "elements": elements}
            if materials:
                result["materials"] = materials
            return result
        if suffix == ".vtk":
            nodes: List[List[float]] = []
            elements: List[List[int]] = []
            materials: List[Any] = []
            it = iter(lines)
            for ln in it:
                up = ln.upper()
                if up.startswith("POINTS"):
                    parts = ln.split()
                    npts = int(parts[1])
                    for _ in range(npts):
                        x, y, z = next(it).split()[:3]
                        nodes.append([float(x), float(y), float(z)])
                elif up.startswith("POLYGONS") or up.startswith("TRIANGLE") or up.startswith("TRIANGLES"):
                    parts = ln.split()
                    ntri = int(parts[1])
                    for _ in range(ntri):
                        vals = next(it).split()
                        if int(vals[0]) >= 3:
                            elements.append([int(vals[1]), int(vals[2]), int(vals[3])])
                elif up.startswith("CELL_DATA"):
                    ncell = int(ln.split()[1])
                    # expect "SCALARS" followed by lookup table and values
                    ln = next(it).strip()
                    if ln.upper().startswith("SCALARS"):
                        name = ln.split()[1].lower()
                        if name in {"material", "materials", "gmsh:physical", "cell_tags"}:
                            ln = next(it).strip()
                            vals: List[str] = []
                            if not ln.upper().startswith("LOOKUP_TABLE"):
                                vals.extend(ln.split())
                            while len(vals) < ncell:
                                vals.extend(next(it).split())
                            for v in vals[:ncell]:
                                try:
                                    materials.append(int(v))
                                except ValueError:
                                    materials.append(v)
            result: Dict[str, Any] = {"nodes": nodes, "elements": elements}
            if materials:
                result["materials"] = materials
            return result
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
    suffix = p.suffix.lower()
    if suffix == ".json":
        return json.loads(p.read_text())

    if suffix in {".stl", ".vtk"}:
        pts: Sequence[Sequence[float]] | None = None
        if meshio is not None:  # pragma: no cover - exercised when meshio available
            try:
                pts = meshio.read(p).points  # type: ignore[assignment]
            except Exception:
                pts = None
        if pts is None:
            if suffix == ".stl":
                pts = []
                for ln in p.read_text().splitlines():
                    ln = ln.strip()
                    if ln.lower().startswith("vertex"):
                        _, x, y, z = ln.split()
                        pts.append([float(x), float(y), float(z)])
            elif suffix == ".vtk":
                pts = []
                lines = p.read_text().splitlines()
                read = False
                for ln in lines:
                    ln = ln.strip()
                    if not ln:
                        continue
                    if read:
                        parts = ln.split()
                        if len(parts) == 3:
                            try:
                                pts.append([float(v) for v in parts])
                                continue
                            except ValueError:
                                break
                        else:
                            break
                    if ln.upper().startswith("POINTS"):
                        read = True
        if not pts:
            raise ValueError(f"Unsupported axisymmetric mesh format: {suffix}")
        r = sorted({math.hypot(pt[0], pt[1]) for pt in pts})
        z = sorted({pt[2] for pt in pts})
        return {"r": r, "z": z}

    lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    nr, nz = map(int, lines[0].split())
    r = [float(lines[i + 1]) for i in range(nr)]
    z = [float(lines[1 + nr + i]) for i in range(nz)]
    return {"r": r, "z": z}
