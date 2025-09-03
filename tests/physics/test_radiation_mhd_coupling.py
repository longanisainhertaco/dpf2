
import cmath
import math
import numpy as np



from dpf2.circuit.distributed import TransmissionLineSegment
from dpf2.physics.radiation_mhd import RadiationMHDSolver
from dpf2.rlc_solver import solve_distributed_circuit


def _sum(arr):
    data = getattr(arr, "data", arr)
    if isinstance(data, list):
        return sum(_sum(x) for x in data)
    return data


def test_frequency_domain_coupling_zero_back_reaction():
    seg = TransmissionLineSegment(
        from_node=0,
        to_node=1,
        length=1.0,
        L_per_m=1e-6,
        R_per_m=0.0,
        C_per_m=1e-10,
    )
    solver = RadiationMHDSolver()
    freq = 1e7
    dt = 1e-9
    t_end = 1e-8
    res = solve_distributed_circuit(
        [seg],
        [],
        V0=0.0,
        t_end=t_end,
        dt=dt,
        frequency=freq,
        em_solver=solver,
    )
    ref = solve_distributed_circuit([seg], [], V0=0.0, t_end=t_end, dt=dt, frequency=freq)
    assert np.allclose(res.current, ref.current)
    assert np.allclose(res.voltage, ref.voltage)
    iface = solver.coupling_interface()
    assert iface.back_reaction == 0.0


def _amp_phase(signal, t, w):
    """Return amplitude and phase of ``signal`` sampled at times ``t``."""

    s = np.sin(w * t)
    c = np.sin(w * t + np.pi / 2)  # cosine via phase shift
    n = len(t)
    if n > 1:
        s = s[:-1]
        c = c[:-1]
        signal = signal[:-1]
        n -= 1
    a = 2 / n * _sum(signal * s)
    b = 2 / n * _sum(signal * c)
    amp = float(np.sqrt(a * a + b * b))
    phase = float(math.atan2(b, a))
    return amp, phase


def test_wave_propagation_matches_analytic_solution():
    seg = TransmissionLineSegment(
        from_node=0,
        to_node=1,
        length=1.0,
        L_per_m=1e-6,
        R_per_m=0.0,
        C_per_m=1e-10,
    )
    freq = 1e7
    dt = 5e-9  # 20 steps per period
    t_end = 2e-7  # two periods
    Z0 = seg.characteristic_impedance(freq)
    res = solve_distributed_circuit(
        [seg],
        [],
        V0=1.0,
        t_end=t_end,
        dt=dt,
        frequency=freq,
        Z_load=Z0,
    )
    w = 2 * np.pi * freq
    t = res.t
    amp_I, phase_I = _amp_phase(res.branch_currents[:, 0], t, w)
    gamma = seg.propagation_constant(freq) * seg.length
    expected = (cmath.cosh(gamma) / cmath.sinh(gamma)) / Z0
    assert np.isclose(amp_I, abs(expected), rtol=1e-3)
    assert np.isclose(phase_I, cmath.phase(expected), atol=1e-3)


def test_energy_conservation_closed_system():
    solver = RadiationMHDSolver()
    state = solver.allocate_state((2, 2, 2))
    state.energy = state.energy + 1.0
    state.magnetic = state.magnetic + 0.1
    dt = 1e-6
    current = 2.0
    voltage = 3.0
    def total_energy(st):
        e = _sum(st.energy)
        B2 = _sum(st.magnetic * st.magnetic)
        return float(e + 0.5 * B2)

    initial = total_energy(state)
    radiated = 0.0
    totals: list[float] = []
    for _ in range(5):
        before = total_energy(state)
        state = solver.step(state, dt, current=current, voltage=voltage)
        after = total_energy(state)
        radiated += before - after
        totals.append(after + radiated)
    for tot in totals:
        assert np.isclose(tot, initial, rtol=1e-12)

