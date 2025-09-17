import json
from pathlib import Path

from dpf2.geometry import load_cad_geometry, load_unstructured_mesh


def test_load_cad_geometry(tmp_path):
    data = {"nodes": [[0, 0, 0], [1, 0, 0], [0, 1, 0]], "elements": [[0, 1, 2]]}
    path = tmp_path / "cad.json"
    path.write_text(json.dumps(data))
    loaded = load_cad_geometry(path)
    assert loaded == data


def test_load_unstructured_mesh(tmp_path):
    content = "\n".join(
        [
            "3",
            "0 0 0",
            "1 0 0",
            "0 1 0",
            "1",
            "0 1 2",
        ]
    )
    path = tmp_path / "mesh.txt"
    path.write_text(content)
    loaded = load_unstructured_mesh(path)
    assert loaded["nodes"][0] == [0.0, 0.0, 0.0]
    assert loaded["elements"][0] == [0, 1, 2]
