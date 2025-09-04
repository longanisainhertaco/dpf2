import numpy as np
import pytest

from dpf2.hall_mhd_solver import HallMHDSolver, MHDState, _divergence
from dpf2.core.bases import CircuitSolverBase, CouplingState


def _uniform_state(shape):
    rho = np.ones(shape)
    mom = np.zeros(shape + (3,))
    mom[..., 0] = 0.1
    mom[..., 1] = -0.2
    mom[..., 2] = 0.05
    B = np.array(
        [
            [
                [[0.3, -0.1, 0.2] for _ in range(shape[2])]
                for _ in range(shape[1])
            ]
            for _ in range(shape[0])
        ]
    )
    p = 1.0
    gamma = 5.0 / 3.0
    kinetic = 0.5 * (0.1 ** 2 + (-0.2) ** 2 + 0.05 ** 2)
    magnetic = 0.5 * (0.3 ** 2 + (-0.1) ** 2 + 0.2 ** 2)
    energy_val = p / (gamma - 1.0) + kinetic + magnetic
    energy = np.full(shape, energy_val)
    return MHDState(rho=rho, mom=mom, energy=energy, B=B)


def test_conservation_and_divergence():
    shape = (4, 4, 4)
    state = _uniform_state(shape)
    solver = HallMHDSolver()
    new_state = solver.step(state, 0.1)
    assert np.allclose(new_state.rho, state.rho)
    assert np.allclose(new_state.mom, state.mom)
    assert np.allclose(new_state.energy, state.energy)
    assert np.allclose(new_state.B, state.B)
    assert np.max(np.abs(_divergence(new_state.B))) < 1e-12


def test_divergence_cleaning():
    shape = (4, 4, 4)
    rng = np.random.default_rng(0)
    rho = np.ones(shape)
    mom = np.zeros(shape + (3,))
    B = rng.random(shape + (3,)) - 0.5
    energy = np.ones(shape)
    state = MHDState(rho=rho, mom=mom, energy=energy, B=B)
    solver = HallMHDSolver()
    new_state = solver.step(state, 0.0)
    initial_div = np.max(np.abs(_divergence(B)))
    final_div = np.max(np.abs(_divergence(new_state.B)))
    assert final_div < initial_div


class DummyCircuit(CircuitSolverBase):
    def __init__(self) -> None:
        self.last_back_emf = 0.0

    def step(self, coupling: CouplingState, back_emf: float, dt: float) -> CouplingState:  # pragma: no cover - simple stub
        self.last_back_emf = back_emf
        return CouplingState(current=coupling.current, voltage=coupling.voltage)


def test_instability_modules_coupling_and_impedance():
    J = np.zeros((1, 3))

    def lhd(J):
        eta = 0.1 * np.ones(J.shape[:-1])
        Ez = np.zeros(J.shape)
        Ez[0][2] = 0.02
        return eta, Ez

    def m0(J):
        eta = 0.05 * np.ones(J.shape[:-1])
        Ez = np.zeros(J.shape)
        Ez[0][2] = 0.03
        return eta, Ez

    circuit = DummyCircuit()
    solver = HallMHDSolver(lower_hybrid_drift=lhd, m0_instability=m0, circuit=circuit)

    solver.compute_anomalous_resistivity(J)
    assert solver.last_E_anom[0][2] == pytest.approx(0.05)
    assert solver.last_voltage_spike == pytest.approx(0.05)

    circuit.step(CouplingState(current=1.0, voltage=0.0), solver.last_voltage_spike, 0.1)
    assert circuit.last_back_emf == pytest.approx(0.05)

    solver.current = 1.0
    solver.impedance_growth.append(solver.last_voltage_spike / (abs(solver.current) + 1e-30))
    assert solver.impedance_growth[-1] == pytest.approx(0.05)
