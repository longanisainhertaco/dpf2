import sys
from pathlib import Path
import numpy as np

# Make Simulation modules importable
sim_path = Path(__file__).resolve().parents[1] / 'Simulation'
sys.path.append(str(sim_path))

from sheath_model import BohmSheath, e_charge
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
    v_bohm = np.sqrt(e_charge * 5.0 / 1.67e-27)
    assert np.allclose(fm.get_E()[2, :, :, -1], v_bohm)


def test_apply_updates_momentum_array():
    density = np.ones((4, 4, 4))
    momentum = np.zeros((3, 4, 4, 4))
    sheath = BohmSheath(electron_temperature=2.0, ion_mass=1.67e-27)
    sheath.apply(density, momentum)
    v_bohm = np.sqrt(e_charge * 2.0 / 1.67e-27)
    assert np.allclose(momentum[2, :, :, -1], v_bohm)
    assert np.all(momentum[2, :, :, -2] == 0.0)
