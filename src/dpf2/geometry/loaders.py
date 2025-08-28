from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


def load_cad_geometry(path: Path) -> Dict[str, Any]:
    """Load a minimal CAD style geometry description from ``path``.

    The expected format is JSON containing ``nodes`` and ``elements`` lists.
    """
    return json.loads(Path(path).read_text())


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
