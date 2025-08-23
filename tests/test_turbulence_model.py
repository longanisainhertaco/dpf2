import os
import sys
import numpy as np
import pytest
from dataclasses import dataclass


# Make Simulation modules importable
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "Simulation"))

from turbulence_model import TurbulenceModel
from utils import SimulationState


@dataclass
class TurbulenceConfig:
    """Minimal configuration stand-in used for testing."""

    wall_function_type: str = "none"
    C_mu: float = 0.09
    C_epsilon1: float = 1.44
    C_epsilon2: float = 1.92
    sigma_k: float = 1.0
    sigma_epsilon: float = 1.3
    initial_k: float = 1e-6
    initial_epsilon: float = 1e-8
    compressibility_correction: bool = False
    wall_function_kappa: float = 0.41
    wall_function_E: float = 9.8
    compressibility_alpha: float = 1.0
    compressibility_beta: float = 1.0


def _basic_state(grid_shape=(5, 1, 1), dx=0.01):
    """Create a minimal SimulationState with required fields."""

    density = np.ones(grid_shape)
    velocity = np.zeros(grid_shape + (3,))
    viscosity = np.ones(grid_shape) * 1e-5
    bc = {"x_lo": "wall", "x_hi": "wall"}
    state = SimulationState(
        grid_shape, dx, dx, dx, (0.0, 0.0, 0.0), bc,
        density=density, velocity=velocity, viscosity=viscosity, ghost=1
    )
    return state


def test_log_law_wall_function_adjusts_velocity_and_fields():
    cfg = TurbulenceConfig(wall_function_type="log_law")
    model = TurbulenceModel(cfg)
    state = _basic_state()

    model.k = np.ones_like(state.density) * 0.1
    model.epsilon = np.ones_like(state.density) * 0.01

    g = state.ghost
    k0 = model.k[g, 0, 0]
    nu = state.viscosity[g, 0, 0] / state.density[g, 0, 0]
    u_tau = (model.C_mu ** 0.25) * np.sqrt(k0)
    y = 0.5 * state.dx
    y_plus = y * u_tau / nu
    u_plus = (1.0 / model.wall_function_kappa) * np.log(model.wall_function_E * y_plus)
    expected_velocity = u_plus * u_tau
    expected_k = u_tau ** 2 / np.sqrt(model.C_mu)
    expected_epsilon = (u_tau ** 3) / (model.wall_function_kappa * y)

    model._apply_wall_functions(state)

    assert np.isclose(state.velocity[g, 0, 0, 0], expected_velocity)
    assert np.isclose(model.k[g, 0, 0], expected_k)
    assert np.isclose(model.epsilon[g, 0, 0], expected_epsilon)


def test_power_law_wall_function_adjusts_velocity():
    cfg = TurbulenceConfig(wall_function_type="power_law")
    model = TurbulenceModel(cfg)
    state = _basic_state()

    model.k = np.ones_like(state.density) * 0.1
    model.epsilon = np.ones_like(state.density) * 0.01

    g = state.ghost
    state.velocity[g + 1, 0, 0, 0] = 5.0
    expected = 5.0 * (0.5 * state.dx / state.dx) ** (1.0 / 7.0)

    model._apply_wall_functions(state)

    assert np.isclose(state.velocity[g, 0, 0, 0], expected)


def test_wall_function_validation():
    cfg = TurbulenceConfig(wall_function_type="log_law")
    model = TurbulenceModel(cfg)
    state = _basic_state()
    state.viscosity = None

    model.k = np.ones_like(state.density) * 0.1
    model.epsilon = np.ones_like(state.density) * 0.01

    with pytest.raises(ValueError):
        model._apply_wall_functions(state)


def test_unrecognized_wall_function_type_raises():
    """Ensure an error is raised for unsupported wall function options."""
    cfg = TurbulenceConfig(wall_function_type="invalid_option")

    with pytest.raises(ValueError):
        TurbulenceModel(cfg)

