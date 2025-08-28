import pytest

from dpf2.physics.pic import SimplePIC, HybridPIC


def test_simple_pic_coupling():
    pic = SimplePIC(
        charge=1.0,
        mass=1.0,
        length=1.0,
        positions=[0.0, 0.5],
        velocities=[0.0, 0.0],
    )
    pic.step(None, dt=1.0, current=0.0, voltage=1.0)
    iface = pic.coupling_interface()
    assert iface.back_reaction == pytest.approx(2.0)


def test_hybrid_pic_coupling():
    pic = HybridPIC(
        charge=1.0,
        mass=1.0,
        length=1.0,
        positions=[0.0, 0.5],
        velocities=[0.0, 0.0],
        fluid_fraction=0.5,
    )
    pic.step(None, dt=1.0, current=2.0, voltage=1.0)
    iface = pic.coupling_interface()
    assert iface.back_reaction == pytest.approx(3.0)
