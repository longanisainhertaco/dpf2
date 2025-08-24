import numpy as np
import h5py

from dpf2.eos import TabulatedEOS, create_eos
from dpf2.chemistry import FlychkTable
from dpf2.radiation import BremsstrahlungModel, MonteCarloRadiation
from dpf2.hall_mhd_solver import HallMHDSolver, MHDState
from dpf2.core_schema import EOSModel


def _make_test_table(path):
    rho = np.array([1.0, 2.0])
    T = np.array([100.0, 200.0])
    p = rho[:, None] * T[None, :]
    e = T[None, :] / rho[:, None]
    with h5py.File(path, "w") as f:
        f.create_dataset("rho", data=rho)
        f.create_dataset("T", data=T)
        f.create_dataset("p", data=p)
        f.create_dataset("e", data=e)


def test_tabulated_eos_pressure(tmp_path):
    file_path = tmp_path / "eos.h5"
    _make_test_table(file_path)
    eos = TabulatedEOS(file_path)
    rho = np.array([1.5])
    T = np.array([150.0])
    p = eos.pressure(rho, T)
    e = eos.energy(rho, T)
    assert np.allclose(p, rho * T)
    assert np.allclose(e, T / rho)


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


def test_hall_mhd_solver_with_models(tmp_path):
    eos = create_eos(EOSModel.IDEAL)
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
