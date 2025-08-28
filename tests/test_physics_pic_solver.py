from dpf2.physics.pic import SimplePIC
from dpf2.core.bases import CouplingState


def test_pic_updates_and_couples():
    solver = SimplePIC(
        charge=-1.0,
        mass=1.0,
        length=1.0,
        positions=[0.0, 0.5],
        velocities=[0.0, 0.0],
    )
    solver.step(None, dt=1.0, current=0.0, voltage=1.0)
    assert any(v != 0.0 for v in solver.velocities)
    feedback = solver.coupling_interface()
    assert isinstance(feedback, CouplingState)
    assert feedback.back_reaction != 0.0
