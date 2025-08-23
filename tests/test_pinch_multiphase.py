import numpy as np

from dpf2.solvers.pinch_multiphase import PinchMultiphaseSolver


def test_phase_transitions_and_conservation():
    solver = PinchMultiphaseSolver()
    t = np.linspace(0.0, 1e-6, 200)
    current = np.full_like(t, 1e5)
    res = solver.run(t, current)

    # verify phase order
    transitions = [res.phase[0]]
    for ph in res.phase[1:]:
        if ph != transitions[-1]:
            transitions.append(ph)
    assert transitions == ["axial", "radial", "stagnation", "rebound"]

    # mass conservation
    mass_series = np.full_like(res.time, res.mass)
    assert np.allclose(np.diff(mass_series), 0.0)

    # momentum continuity at phase boundaries
    radial_momentum = res.mass * res.radial_velocity
    axial_momentum = res.mass * res.axial_velocity
    idx = [i for i in range(1, res.phase.size) if res.phase[i] != res.phase[i - 1]]
    for i in idx:
        assert np.isclose(radial_momentum[i - 1], radial_momentum[i])
        assert np.isclose(axial_momentum[i - 1], axial_momentum[i])
