from dpf2.physics.radiation import RadiationTransport
from dpf2.radiation.multigroup import MultiGroupDiffusion


def test_radiation_step_couples():
    diffusion = MultiGroupDiffusion([0.1, 0.1])
    solver = RadiationTransport(diffusion, dx=1.0)
    fluid, radiation = solver.step([1.0], dt=0.1)
    assert fluid[0] < 1.0
    assert len(radiation) == 2
