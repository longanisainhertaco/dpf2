import types
import sys
import numpy as np

sys.modules.setdefault("h5py", types.ModuleType("h5py"))
sys.modules.setdefault("picmi", types.ModuleType("picmi"))
sys.modules.setdefault("adios2", types.ModuleType("adios2"))
amrex_stub = types.ModuleType("amrex")


class EBIndexSpace:
    pass


amrex_stub.EBIndexSpace = EBIndexSpace
sys.modules.setdefault("amrex", amrex_stub)
collision_stub = types.ModuleType("dpf2.simulation.collision_model")


class CollisionModel:
    pass


collision_stub.CollisionModel = CollisionModel
sys.modules.setdefault("dpf2.simulation.collision_model", collision_stub)

from dpf2.simulation.warpx_wrapper import _compute_kinetic_energy, WarpXWrapper, c


def test_relativistic_energy_conservation():
    vel = np.array([[0.01 * c, 0.0, 0.0]])
    e_nr = _compute_kinetic_energy(vel, 1.0, relativistic=False)[0]
    e_rel = _compute_kinetic_energy(vel, 1.0, relativistic=True)[0]
    assert abs(e_rel - e_nr) / e_nr < 1e-4
    vel2 = np.array([[0.9 * c, 0.0, 0.0]])
    e_rel2 = _compute_kinetic_energy(vel2, 1.0, relativistic=True)[0]
    gamma = 1.0 / (1 - 0.9**2) ** 0.5
    assert abs(e_rel2 - (gamma - 1) * c**2) / ((gamma - 1) * c**2) < 1e-4


def test_time_dependent_boundary_injection():
    grid_shape = (2, 2, 2)

    class DummyWarp:
        def __init__(self):
            self.fields = {}
            self._t = 0.0

        def set_boundary_field(self, arr, comp):
            self.fields[comp] = arr

        def get_time(self):
            return self._t

    def callback():
        return {"boundary_fields": {"Ex": lambda t: np.full(grid_shape, t)}}

    wrapper = WarpXWrapper.__new__(WarpXWrapper)
    wrapper.fluid_callback = callback
    wrapper.grid_shape = grid_shape
    wrapper.interp_method = "linear"
    wrapper.time_dependent_boundaries = True
    wrapper.warp = DummyWarp()

    wrapper.inject_boundary_fields()
    arr0 = wrapper.warp.fields["Ex"]
    data0 = arr0.data if hasattr(arr0, "data") else arr0
    assert all(v == 0.0 for plane in data0 for row in plane for v in row)
    wrapper.warp._t = 1.0
    wrapper.inject_boundary_fields()
    arr1 = wrapper.warp.fields["Ex"]
    data1 = arr1.data if hasattr(arr1, "data") else arr1
    assert all(v == 1.0 for plane in data1 for row in plane for v in row)
