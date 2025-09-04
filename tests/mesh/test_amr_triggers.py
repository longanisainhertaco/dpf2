import numpy as np
from dpf2.mesh.amr import (
    debye_length_refinement,
    ion_inertial_length_refinement,
    pressure_gradient_refinement,
    current_density_refinement,
    AMRMesh,
)


def test_debye_length_trigger():
    ld = np.array([[0.1, 0.6], [0.4, 0.3]])
    mask = debye_length_refinement(ld, 0.5)
    assert float(np.sum(mask)) == 3


def test_ion_inertial_length_trigger():
    di = np.array([[1.0, 0.2], [0.7, 0.1]])
    mask = ion_inertial_length_refinement(di, 0.5)
    assert float(np.sum(mask)) == 2


def test_pressure_gradient_trigger():
    p = np.array([[1.0, 1.0], [2.0, 4.0]])
    mask = pressure_gradient_refinement(p, 1.0)
    assert np.any(mask)


def test_current_density_trigger():
    J = np.array([[[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]],
                  [[0.0, 0.0, 0.0], [0.0, 0.0, 6.0]]])
    mask = current_density_refinement(J, 4.5)
    assert float(np.sum(mask)) == 2


def test_amrmesh_combines_triggers():
    mesh = AMRMesh(shape=(2, 2), criteria={"lambda_D_threshold": 0.5, "current_density_threshold": 4.5})
    ld = np.array([[0.1, 0.6], [0.4, 0.2]])
    J = np.array([[[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]],
                  [[0.0, 0.0, 0.0], [0.0, 0.0, 6.0]]])
    stats = mesh.refine({"lambda_D": ld, "current": J})
    assert stats["tagged_cells"] == 4
    assert mesh.tagging_stats()["tagged_cells"] == 4
