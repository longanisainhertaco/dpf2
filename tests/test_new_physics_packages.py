import numpy as np

from dpf2.eos import TabulatedEOS, create_eos
from dpf2.chemistry import FlychkTable
from dpf2.radiation import BremsstrahlungModel, MonteCarloRadiation
from dpf2.hall_mhd_solver import HallMHDSolver, MHDState
from core_schema import EOSModel


def test_tabulated_eos_pressure():
    eos = TabulatedEOS("tests/data/sesame_dummy.csv")
    rho = np.array([1.5])
    e = np.array([450.0])
    p = eos.pressure(rho, e)
    assert np.allclose(p, rho * 450.0)


def test_flychk_interpolation():
    chem = FlychkTable("tests/data/flychk_dummy.csv")
    T = np.array([5.0, 50.0])
    z = chem.ionization_state(np.ones_like(T), T)
    assert z[0] > 0.1 and z[1] < 2.0


def test_monte_carlo_radiation_loss():
    base = BremsstrahlungModel(coeff=1e-6)
    mc = MonteCarloRadiation(base, rng_seed=1)
    rho = np.ones((4,))
    T = np.ones((4,)) * 1e3
    loss = mc.loss(rho, T)
    assert loss.min() >= 0


def test_hall_mhd_solver_with_models():
    eos = create_eos(EOSModel.TABULATED, table_path="tests/data/sesame_dummy.csv")
    chem = FlychkTable("tests/data/flychk_dummy.csv")
    rad = MonteCarloRadiation(BremsstrahlungModel(coeff=1e-6), rng_seed=0)
    solver = HallMHDSolver(eos=eos, chemistry=chem, radiation=rad)
    shape = (2, 2, 2)
    rho = np.ones(shape)
    mom = np.zeros(shape + (3,))
    energy = np.ones(shape) * 10.0
    B = np.zeros(shape + (3,))
    state = MHDState(rho=rho, mom=mom, energy=energy, B=B)
    new_state = solver.step(state, 1e-9)
    assert np.all(solver.last_pressure > 0)
    assert np.all(solver.last_ionization >= 0)
    assert np.all(new_state.energy <= energy)
