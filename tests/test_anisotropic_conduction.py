import numpy as np
import importlib.util
from pathlib import Path

from physics_models import PhysicsModels
from core_schema import EOSModel

# Load ResistiveMHD directly to avoid package import side effects
_mhd_spec = importlib.util.spec_from_file_location(
    "mhd", Path(__file__).resolve().parent.parent / "src" / "dpf2" / "physics" / "mhd.py"
)
mhd = importlib.util.module_from_spec(_mhd_spec)
_mhd_spec.loader.exec_module(mhd)  # type: ignore
ResistiveMHD = mhd.ResistiveMHD


def test_anisotropic_conductivity_fields():
    cfg = PhysicsModels(
        eos_model=EOSModel.IDEAL,
        gamma=1.4,
        anisotropic_conductivity_enabled=True,
        conductivity_parallel=10.0,
        conductivity_perpendicular=1.0,
        hall_parameter=0.5,
    )
    assert cfg.anisotropic_conductivity_enabled is True
    assert cfg.conductivity_parallel == 10.0
    assert cfg.conductivity_perpendicular == 1.0
    assert cfg.hall_parameter == 0.5


def test_cross_field_conduction_and_hall_current():
    model = ResistiveMHD(sigma_parallel=10.0, sigma_perp=1.0, hall_param=0.5)
    grad_T = np.array([1.0, 2.0, 3.0])
    B = np.array([0.0, 0.0, 1.0])
    J = np.array([1.0, 0.0, 0.0])

    q = model.cross_field_conduction(grad_T, B)
    assert np.allclose(q, np.array([-1.0, -2.0, -30.0]))

    hall_J = model.hall_current(J, B)
    assert np.allclose(hall_J, np.array([1.0, -0.5, 0.0]))
