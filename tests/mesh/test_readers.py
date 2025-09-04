from pathlib import Path

import pytest

import dpf2.mesh.readers as readers


def test_read_stl_requires_meshio(monkeypatch):
    monkeypatch.setattr(readers, "meshio", None)
    with pytest.raises(RuntimeError):
        readers.read_stl(Path("dummy.stl"))


def test_read_vtk_requires_meshio(monkeypatch):
    monkeypatch.setattr(readers, "meshio", None)
    with pytest.raises(RuntimeError):
        readers.read_vtk(Path("dummy.vtk"))
