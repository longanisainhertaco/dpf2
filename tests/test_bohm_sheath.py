import sys
from pathlib import Path
import numpy as np

# Make Simulation modules importable
sim_path = Path(__file__).resolve().parents[1] / 'Simulation'
sys.path.append(str(sim_path))

from sheath_model import BohmSheath, e_charge, m_e
from utils import FieldManager, SimulationState


def make_state():
    fm = FieldManager(
        grid_shape=(4, 4, 4),
        dx=1.0,
        dy=1.0,
        dz=1.0,
        domain_lo=(0.0, 0.0, 0.0),
        boundary_conditions={
            'x_lo': 'periodic',
            'x_hi': 'periodic',
            'y_lo': 'periodic',
            'y_hi': 'periodic',
            'z_lo': 'periodic',
            'z_hi': 'periodic',
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
    return state, fm


def test_apply_updates_field_manager():
    state, fm = make_state()
    sheath = BohmSheath(electron_temperature=5.0, ion_mass=1.67e-27)
    sheath.apply(state)
    mass_ratio = 1.67e-27 / (2 * np.pi * m_e)
    sheath_potential = 5.0 * np.log(np.sqrt(mass_ratio))
    expected_field = sheath_potential / fm.dz
    assert np.allclose(fm.get_E()[2, :, :, -1], expected_field)


def test_apply_updates_state_velocity_and_potential():
    state, _ = make_state()
    sheath = BohmSheath(electron_temperature=5.0, ion_mass=1.67e-27)
    sheath.apply(state)
    v_bohm = np.sqrt(e_charge * 5.0 / 1.67e-27)
    mass_ratio = 1.67e-27 / (2 * np.pi * m_e)
    sheath_potential = 5.0 * np.log(np.sqrt(mass_ratio))
    assert np.allclose(state.velocity[2, :, :, -1], v_bohm)
    assert np.allclose(state.potential[:, :, -1], sheath_potential)


def test_apply_preserves_interior_state():
    """Applying the sheath should only modify the boundary-adjacent cells."""
    state, _ = make_state()
    state.velocity = np.zeros((3, 4, 4, 4))
    state.potential = np.zeros((4, 4, 4))
    sheath = BohmSheath(electron_temperature=5.0, ion_mass=1.67e-27)
    sheath.apply(state)
    # Interior (second-to-last) cells remain untouched
    assert np.all(state.velocity[:, :, :, -2] == 0.0)
    assert np.all(state.potential[:, :, -2] == 0.0)


def test_apply_direct_field_manager():
    """The sheath can operate on a FieldManager without a SimulationState."""
    _, fm = make_state()
    sheath = BohmSheath(electron_temperature=3.0, ion_mass=1.67e-27)
    sheath.apply(fm)
    mass_ratio = 1.67e-27 / (2 * np.pi * m_e)
    phi_s = 3.0 * np.log(np.sqrt(mass_ratio))
    expected_field = phi_s / fm.dz
    assert np.allclose(fm.get_E()[2, :, :, -1], expected_field)


def test_apply_updates_momentum_array():
    density = np.ones((4, 4, 4))
    momentum = np.zeros((3, 4, 4, 4))
    sheath = BohmSheath(electron_temperature=2.0, ion_mass=1.67e-27)
    sheath.apply(density, momentum)
    v_bohm = np.sqrt(e_charge * 2.0 / 1.67e-27)
    assert np.allclose(momentum[2, :, :, -1], v_bohm)
    assert np.all(momentum[2, :, :, -2] == 0.0)


def test_apply_handles_all_axes():
    """The sheath orientation can be changed via the ``axis`` parameter."""
    mass_ratio = 1.67e-27 / (2 * np.pi * m_e)
    phi_s = 5.0 * np.log(np.sqrt(mass_ratio))
    v_bohm = np.sqrt(e_charge * 5.0 / 1.67e-27)

    # X-directed sheath
    state_x, fm_x = make_state()
    sheath_x = BohmSheath(electron_temperature=5.0, ion_mass=1.67e-27, axis=0)
    sheath_x.apply(state_x)
    assert np.allclose(state_x.velocity[0, -1, :, :], v_bohm)
    assert np.allclose(state_x.potential[-1, :, :], phi_s)
    assert np.allclose(fm_x.get_E()[0, -1, :, :], phi_s / fm_x.dx)

    # Y-directed sheath
    state_y, fm_y = make_state()
    sheath_y = BohmSheath(electron_temperature=5.0, ion_mass=1.67e-27, axis=1)
    sheath_y.apply(state_y)
    assert np.allclose(state_y.velocity[1, :, -1, :], v_bohm)
    assert np.allclose(state_y.potential[:, -1, :], phi_s)
    assert np.allclose(fm_y.get_E()[1, :, -1, :], phi_s / fm_y.dy)
