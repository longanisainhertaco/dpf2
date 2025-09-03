from pathlib import Path

from dpf2.geometry.loaders import load_cad_geometry, load_axisymmetric_mesh


def test_load_step_geometry():
    path = Path(__file__).with_name("sample.step")
    data = load_cad_geometry(path)
    assert len(data["nodes"]) == 3
    assert data["elements"][0] == [1, 2, 3]


def test_load_iges_geometry():
    path = Path(__file__).with_name("sample.igs")
    data = load_cad_geometry(path)
    assert len(data["elements"]) == 1


def test_load_axisymmetric_mesh():
    path = Path(__file__).with_name("axisymmetric.json")
    mesh = load_axisymmetric_mesh(path)
    assert mesh["r"][0] == 0.0
    assert mesh["z"][-1] == 2.0
