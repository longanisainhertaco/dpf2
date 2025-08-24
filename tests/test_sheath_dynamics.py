import sys
import types
import contextlib
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest

# Stub out heavy dependencies before importing HybridController
sys.modules.setdefault("fluid_solver_high_order", types.SimpleNamespace(FluidSolverHighOrder=object))
sys.modules.setdefault("warpx_wrapper", types.SimpleNamespace(WarpXWrapper=object))
sys.modules.setdefault("radiation_model", types.SimpleNamespace(RadiationModel=object))
sys.modules.setdefault("collision_model", types.SimpleNamespace(CollisionModel=object))


class _SheathConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class _HybridConfig:
    """Placeholder for HybridConfig used in tests."""

sys.modules.setdefault("config_schema", types.SimpleNamespace(SheathConfig=_SheathConfig, HybridConfig=_HybridConfig))

from dpf2.simulation.sheath_model import PlasmaSheathFormation
from dpf2.simulation.utils import FieldManager, SimulationState
from dpf2.simulation.hybrid_controller import HybridController


class DummySheath(PlasmaSheathFormation):
    """Lightweight sheath model used for tests to avoid heavy calculations."""

    def compute_sheath_thickness(self):
        self.sheath_thickness = self.dx
        return self.sheath_thickness

    def _poisson_equation(self):
        self.x_grid = np.array([0.0, self.sheath_thickness])
        self.electric_field = np.array([0.0, 1.0])
        self.electric_potential = np.zeros(2)
        return self.electric_potential

    def compute_density_profiles(self):
        self.ion_density_profile = np.ones(2) * self.ion_density
        self.electron_density_profile = np.ones(2) * self.electron_density
        self.bohm_velocity = 1.0


def make_state_and_sheath():
    fm = FieldManager(
        grid_shape=(4, 4, 4),
        dx=1.0,
        dy=1.0,
        dz=1.0,
        domain_lo=(0.0, 0.0, 0.0),
        boundary_conditions={
            "x_lo": "periodic",
            "x_hi": "periodic",
            "y_lo": "periodic",
            "y_hi": "periodic",
            "z_lo": "periodic",
            "z_hi": "periodic",
        },
    )
    state = SimulationState(
        grid_shape=(4, 4, 4),
        dx=1.0,
        dy=1.0,
        dz=1.0,
        domain_lo=(0.0, 0.0, 0.0),
        boundary_conditions={},
        field_manager=fm,
    )
    cfg = SimpleNamespace(
        ion_density=1e18,
        electron_density=1e18,
        sheath_voltage=10.0,
        ion_temperature=1.0,
        electron_temperature=1.0,
        ion_mass=1.67e-27,
        dx=1e-4,
        max_sheath_thickness=1e-3,
        num_grid_points=2,
        plasma_edge_potential=0.0,
    )
    sheath = DummySheath(cfg)
    return state, sheath, fm


def test_sheath_model_updates_e_field():
    state, sheath, fm = make_state_and_sheath()
    sheath.apply(state, dt=1e-9)
    assert np.allclose(fm.get_E()[0, :, :, -1], 1.0)


def test_hybrid_controller_applies_sheath_bc():
    state, sheath, fm = make_state_and_sheath()
    config = SimpleNamespace(
        criteria=SimpleNamespace(grad_thr=1.0, knud_thr=1.0, hall_thr=1.0, non_max_fac=1.0),
        coupling=SimpleNamespace(
            buffer_cells=1,
            filter_sigma=1.0,
            blend_width=1,
            max_iters=1,
            coupling_tol=1e-3,
            target_vol_frac=0.1,
        ),
    )
    dummy = SimpleNamespace()
    controller = HybridController(
        config=config,
        fluid_solver=dummy,
        pic_solver=None,
        circuit_model=dummy,
        radiation_model=dummy,
        collision_model=dummy,
        sheath_model=sheath,
        field_manager=fm,
    )
    controller.apply_boundary_conditions(state, dt=1e-9)
    assert np.allclose(fm.get_E()[0, :, :, -1], 1.0)
