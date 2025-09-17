import importlib.util
from pathlib import Path

"""Unit tests for the :mod:`dpf2.simulation.warp_piclibrary` helpers.

These tests use light–weight mock objects to emulate the parts of the WarpX
API that the collision handler interacts with.  This allows verification of
the collision logic without requiring a full WarpX installation.
"""


import math
import random

import pytest

module_path = (
    Path(__file__).resolve().parent.parent / "src/dpf2/simulation/warp_piclibrary.py"
)
spec = importlib.util.spec_from_file_location("warp_pic", module_path)
warp_pic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(warp_pic)
PICCollisionHandler = warp_pic.PICCollisionHandler


class SimpleParticleContainer:
    def __init__(self, velocities, mass=1.0):
        self._vel = [list(vec) for vec in velocities]
        self.mass = mass

    def get_velocities(self):
        return [vec[:] for vec in self._vel]

    def set_velocities(self, v):
        self._vel = [list(vec) for vec in v]


class SimpleWarpX:
    def __init__(self):
        self._species = {
            "e": SimpleParticleContainer([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], mass=1.0),
            "i": SimpleParticleContainer([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], mass=1.0),
        }
        self.registered_ops = []
        self.volume = 1.0

    def get_particle_container(self, name):
        return self._species[name]

    def add_collision_operator(self, s1, s2, freq_func, kwargs):
        self.registered_ops.append((s1, s2, freq_func, kwargs))


def _vec_sum(vs):
    return [sum(comp) for comp in zip(*vs)]


def _arrays_close(a, b, tol=1e-12):
    for va, vb in zip(a, b):
        for xa, xb in zip(va, vb):
            if not math.isclose(xa, xb, rel_tol=1e-9, abs_tol=tol):
                return False
    return True


def test_apply_collisions_scatter_velocities():
    random.seed(1)
    warp = SimpleWarpX()
    handler = PICCollisionHandler(
        lambda ne, Te, Z=1.0, **k: 100.0, species_pairs=[("e", "i")]
    )

    v_e_before = warp.get_particle_container("e").get_velocities()
    v_i_before = warp.get_particle_container("i").get_velocities()

    handler.apply_collisions(warp, dt=0.1)

    v_e_after = warp.get_particle_container("e").get_velocities()
    v_i_after = warp.get_particle_container("i").get_velocities()

    assert not _arrays_close(v_e_before, v_e_after)
    assert not _arrays_close(v_i_before, v_i_after)

    before_total = [a + b for a, b in zip(_vec_sum(v_e_before), _vec_sum(v_i_before))]
    after_total = [a + b for a, b in zip(_vec_sum(v_e_after), _vec_sum(v_i_after))]
    assert _arrays_close([before_total], [after_total])


def test_no_collisions_when_zero_probability():
    """Collisions should leave velocities unchanged when the rate is zero."""
    random.seed(1)
    warp = SimpleWarpX()
    handler = PICCollisionHandler(
        lambda ne, Te, Z=1.0, **k: 0.0, species_pairs=[("e", "i")]
    )

    v_e_before = warp.get_particle_container("e").get_velocities()
    v_i_before = warp.get_particle_container("i").get_velocities()

    handler.apply_collisions(warp, dt=0.1)

    assert _arrays_close(v_e_before, warp.get_particle_container("e").get_velocities())
    assert _arrays_close(v_i_before, warp.get_particle_container("i").get_velocities())


def test_apply_collisions_unknown_species():
    warp = SimpleWarpX()
    handler = PICCollisionHandler(lambda ne, Te, **k: 1.0, species_pairs=[("e", "x")])
    with pytest.raises(ValueError):
        handler.apply_collisions(warp, dt=0.1)


def test_apply_collisions_missing_hook():
    class BrokenWarp:
        """WarpX-like object missing required collision interface."""

    warp = BrokenWarp()
    handler = PICCollisionHandler(lambda ne, Te, **k: 1.0, species_pairs=[("e", "i")])
    with pytest.raises(AttributeError):
        handler.apply_collisions(warp, dt=0.1)


def test_uses_native_warp_api_when_available():
    class NativeWarp(SimpleWarpX):
        def __init__(self):
            super().__init__()
            self.called = []

        def do_mcc_collisions(self, s1, s2, dt, freq, **kwargs):
            self.called.append((s1, s2, dt))

    warp = NativeWarp()
    handler = PICCollisionHandler(lambda ne, Te, **k: 1.0, species_pairs=[("e", "i")])
    handler.apply_collisions(warp, dt=0.1)
    assert warp.called == [("e", "i", 0.1)]


def test_setup_warpx_collisions_registers_ops():
    warp = SimpleWarpX()
    freq = lambda ne, Te, Z=1.0, **k: 1.0
    handler = PICCollisionHandler(freq)
    handler.setup_warpx_collisions(warp, [("e", "i")])
    assert warp.registered_ops[0][0:2] == ("e", "i")
    assert warp.registered_ops[0][2] is freq
    assert warp.registered_ops[0][3] == {}
