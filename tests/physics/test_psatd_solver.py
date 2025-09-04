import pytest
import numpy as np

from dpf2.physics.pic import SimplePIC
from dpf2.fields.psatd_solver import PSATDSolver


@pytest.mark.parametrize("deposition", ["Esirkepov", "EZ"])
def test_psatd_solver_divergence(deposition):
    pic = SimplePIC(
        charge=1.0,
        mass=1.0,
        length=1.0,
        positions=[0.25, 0.75],
        velocities=[0.1, -0.1],
        field_solver="PSATD",
        deposition=deposition,
        num_cells=32,
    )
    for _ in range(20):
        pic.step(None, dt=0.01, current=0.0, voltage=0.0)
    assert pic.divergence_error < 0.01


def test_psatd_pml_reflection():
    solver = PSATDSolver(num_cells=64, length=1.0, boundary="PML", pml_cells=4, pml_sigma=2.0)
    rho = np.zeros(64)
    rho[0] = 1.0
    E, _ = solver.solve(rho)
    values = [float(v) for v in E]
    reflection = abs(values[0]) / max(abs(v) for v in values)
    assert reflection < 0.01
