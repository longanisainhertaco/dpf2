import numpy as np
import sys
import pydantic_stub
import types

sys.modules.setdefault("pydantic", pydantic_stub)
sys.modules.setdefault("pydantic.dataclasses", pydantic_stub.dataclasses)
import numpy as _real_np
sys.modules["numpy"] = _real_np
scipy_constants = types.SimpleNamespace(mu_0=1.0)
sys.modules.setdefault("scipy", types.SimpleNamespace(constants=scipy_constants))
sys.modules.setdefault("scipy.constants", scipy_constants)
chem_stub = types.ModuleType("dpf2.chemistry")
class _Saha:
    def ionization_state(self, rho, T):
        return np.zeros_like(rho)
chem_stub.ChemistryModel = object
chem_stub.SahaEquilibrium = _Saha
sys.modules.setdefault("dpf2.chemistry", chem_stub)
core_stub = types.ModuleType("dpf2.core_schema")
core_stub.IonizationModel = object
core_stub.ConfigSectionBase = object
core_stub.to_camel_case = lambda s: s
sys.modules.setdefault("dpf2.core_schema", core_stub)
rad_stub = types.ModuleType("dpf2.radiation")
class _Rad: ...
rad_stub.RadiationBase = _Rad
sys.modules.setdefault("dpf2.radiation", rad_stub)
diag_stub = types.ModuleType("dpf2.diagnostics")
class _Output: ...
diag_stub.OutputField = _Output
sys.modules.setdefault("dpf2.diagnostics", diag_stub)

from dpf2.boundary_conditions import KineticSheath
from dpf2.hall_mhd_solver import HallMHDSolver, MHDState
sys.modules["numpy"] = _real_np


def _state(shape=(4, 4, 4)):
    import numpy as np  # ensure real numpy
    rho = np.ones(shape)
    mom = np.zeros(shape + (3,))
    B = np.zeros(shape + (3,))
    p = 1.0
    gamma = 5.0 / 3.0
    energy = np.full(shape, p / (gamma - 1.0))
    return MHDState(rho=rho, mom=mom, energy=energy, B=B)


def test_sheath_limits_current():
    sheath = KineticSheath()
    solver = HallMHDSolver()
    solver.sheath = sheath
    state = _state()
    solver.step(state, dt=1.0, current=10.0)
    assert solver.current <= sheath.last_ion_flux


def test_impurity_radiation_spike():
    sheath = KineticSheath(impurity_fraction=0.5)
    solver = HallMHDSolver(rad_coeff=1.0)
    solver.sheath = sheath
    state = _state()
    energy_before = state.energy.copy()
    new_state = solver.step(state, dt=1.0)
    loss = energy_before - new_state.energy
    assert np.allclose(loss.mean(), sheath.last_impurity_flux * solver.rad_coeff)
