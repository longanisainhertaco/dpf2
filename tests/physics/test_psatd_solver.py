import pytest

from dpf2.physics.pic import SimplePIC


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
