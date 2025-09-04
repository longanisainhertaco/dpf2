from pathlib import Path

from dpf2.geometry.loaders import load_cad_geometry, load_axisymmetric_mesh
from dpf2.geometry import AxisymmetricProfile


def test_load_step_geometry():
    path = Path(__file__).with_name("sample.step")
    data = load_cad_geometry(path)
    assert len(data["nodes"]) == 3
    assert data["elements"][0] == [1, 2, 3]
    assert data["materials"][0] == "steel"


def test_load_iges_geometry():
    path = Path(__file__).with_name("sample.igs")
    data = load_cad_geometry(path)
    assert len(data["elements"]) == 1
    assert data["materials"][0] == 7


def test_load_axisymmetric_mesh():
    path = Path(__file__).with_name("axisymmetric.json")
    mesh = load_axisymmetric_mesh(path)
    assert mesh["r"][0] == 0.0
    assert mesh["z"][-1] == 2.0


def test_load_axisymmetric_stl():
    path = Path(__file__).with_name("axisymmetric.stl")
    mesh = load_axisymmetric_mesh(path)
    assert mesh["r"][-1] == 1.0
    assert mesh["z"][-1] == 2.0


def test_load_axisymmetric_vtk():
    path = Path(__file__).with_name("axisymmetric.vtk")
    mesh = load_axisymmetric_mesh(path)
    assert mesh["r"][-1] == 1.0
    assert mesh["z"][-1] == 2.0


def test_axisymmetric_profile_from_file():
    path = Path(__file__).with_name("axisymmetric.stl")
    prof = AxisymmetricProfile.from_file(path)
    assert prof.r[-1] == 1.0
    assert prof.z[-1] == 2.0


def test_load_tapered_cad():
    path = Path(__file__).with_name("tapered.step")
    data = load_cad_geometry(path)
    assert data["materials"] == ["copper", "vacuum"]
    assert data["features"]["metal"] == [1]
    assert data["features"]["gap"] == [2]


def test_load_hollow_cad():
    path = Path(__file__).with_name("hollow.step")
    data = load_cad_geometry(path)
    assert data["materials"] == ["steel", "air"]
    assert data["features"]["outer"] == [1, 2]
    assert data["features"]["inner"] == [2]
