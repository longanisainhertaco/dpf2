"""Simple solver for a distributed series RLC circuit.

The real project contains a comprehensive circuit model.  For the unit tests in
this kata we implement only the behaviour that is required to demonstrate the
interaction between :class:`~dpf2.circuit.distributed.TransmissionLineSegment`
and :class:`~dpf2.circuit.distributed.TriggeredSwitch` objects.  The solver
supports time varying parameters and switch triggering albeit in a very small
subset of the features of the full application.

The module still exposes ``run_circuit_simulation`` from the original solver for
backwards compatibility.  The new :func:`solve_distributed_circuit` function is
used by the tests introduced in this exercise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Any
from concurrent.futures import ThreadPoolExecutor

# ``numpy`` may be replaced by a light‑weight stub in the test environment but
# it provides the minimal functionality used below (``array``).
from .gpu_utils import xp, solve_linear, to_cpu
import cmath
import numpy as np

try:  # pragma: no cover - cupy optional
    import cupy as cp  # type: ignore
    _currents_kernel = cp.ElementwiseKernel(
        "float64 prev, float64 vi, float64 vj, float64 R, float64 L, float64 dt",
        "float64 out",
        "out = prev + ((vi - vj - R * prev) / L) * dt;",
        "rlc_update_currents",
    )
except Exception:  # pragma: no cover - fallback when cupy unavailable
    cp = None  # type: ignore
    _currents_kernel = None

try:  # pragma: no cover - MPI is optional
    from mpi4py import MPI  # type: ignore
except Exception:  # pragma: no cover - graceful fallback when mpi4py missing
    MPI = None

try:  # pragma: no cover - GPU backend optional
    from numba import cuda  # type: ignore
except Exception:  # pragma: no cover - fallback when numba unavailable
    cuda = None

from .circuit.distributed import TransmissionLineSegment, TriggeredSwitch, assemble_matrices
from .core.bases import PlasmaSolverBase

# Re-export for legacy tests
# ``circuit_solver`` is a heavy dependency in the full project.  The tests in
# this kata only require that ``run_circuit_simulation`` exists so we try to
# import it lazily and fall back to a stub when unavailable.
try:  # pragma: no cover - imported in real application
    from dpf2.circuit_solver import run_circuit_simulation  # type: ignore
except Exception:  # pragma: no cover - simplified test environment
    def run_circuit_simulation(*args, **kwargs):  # type: ignore
        raise RuntimeError("circuit_solver not available in minimal test environment")

__all__ = ["run_circuit_simulation", "solve_distributed_circuit", "DistributedRLCSolution"]


@dataclass
class DistributedRLCSolution:
    """Container returned by :func:`solve_distributed_circuit`.


    The arrays may originate from either :mod:`numpy` or :mod:`cupy`
    depending on the active backend.
    """

    t: Any
    current: Any
    voltage: Any
    branch_currents: Any
    node_voltages: Any
    reflections: Any | None = None



# ---------------------------------------------------------------------------
# Solver


def solve_distributed_circuit(
    segments: Sequence[TransmissionLineSegment],
    switches: Sequence[TriggeredSwitch] | None,
    V0: float,
    t_end: float,
    dt: float,
    I0: float = 0.0,
    frequency: float | None = None,
    Z_load: float | complex | None = None,
    n_threads: int = 1,
    em_solver: PlasmaSolverBase | None = None,
) -> DistributedRLCSolution:
    """Integrate an RLC network using a very small nodal analysis scheme.

    The implementation is intentionally compact and supports only the features
    required by the unit tests.  All capacitors are aggregated into a single
    equivalent element between the first and last nodes.  Inductive branches are
    solved via a linear system enforcing Kirchhoff's current law at each node.
    When ``frequency`` is provided the solver evaluates the telegrapher
    equations in the frequency domain for cascaded transmission line segments
    with an optional terminating ``Z_load``.
    """

    switches = list(switches or [])
    comm = MPI.COMM_WORLD if MPI else None
    rank = comm.Get_rank() if comm is not None else 0
    size = comm.Get_size() if comm is not None else 1

    # Frequency domain telegrapher solution with branch-aware nodal analysis.
    if frequency is not None and segments:
        w = 2.0 * xp.pi * frequency
        n_steps = int(t_end / dt) + 1

        t = xp.array([i * dt for i in range(n_steps)])
        vin = xp.array([xp.sin(w * ti) for ti in t]) * V0

        # Build node list and admittance matrix
        nodes: set[int] = set()
        for seg in segments:
            nodes.add(seg.from_node)
            nodes.add(seg.to_node)
        node_list = sorted(nodes)
        src = node_list[0]
        node_index = {n: i for i, n in enumerate(node_list)}
        n_nodes = len(node_list)

        Y = np.zeros((n_nodes, n_nodes)) + 0j
        for seg in segments:
            gamma = seg.propagation_constant(frequency) * seg.length
            Z0 = seg.characteristic_impedance(frequency)
            sinh_gl = cmath.sinh(gamma)
            cosh_gl = cmath.cosh(gamma)
            if sinh_gl == 0:
                # Treat as simple impedance
                Y_self = 1.0 / Z0
                Y_off = -1.0 / Z0
            else:
                Y_self = (1.0 / Z0) * (cosh_gl / sinh_gl)  # coth
                Y_off = -(1.0 / Z0) * (1.0 / sinh_gl)      # -csch
            i = node_index[seg.from_node]
            j = node_index[seg.to_node]
            Y[i, i] += Y_self
            Y[j, j] += Y_self
            Y[i, j] += Y_off
            Y[j, i] += Y_off

        if Z_load is None:
            ZL = segments[-1].characteristic_impedance(frequency)
        else:
            ZL = np.inf if Z_load == np.inf else complex(Z_load)
        if ZL != np.inf:
            load_idx = node_index[segments[-1].to_node]
            Y[load_idx, load_idx] += 1.0 / ZL

        # Solve for unknown node voltages
        unknown_nodes = [n for n in node_list if n != src]
        unk_idx = [node_index[n] for n in unknown_nodes]
        if unk_idx:
            Y_mat = [[Y[i][j] for j in unk_idx] for i in unk_idx]
            rhs = [-Y[i][node_index[src]] * V0 for i in unk_idx]

            def _solve_complex(A, b):
                n = len(b)
                A = [row[:] for row in A]
                b = b[:]
                for i in range(n):
                    pivot = A[i][i]
                    for j in range(i, n):
                        A[i][j] /= pivot
                    b[i] /= pivot
                    for k in range(n):
                        if k == i:
                            continue
                        factor = A[k][i]
                        for j in range(i, n):
                            A[k][j] -= factor * A[i][j]
                        b[k] -= factor * b[i]
                return b

            v_unknown = _solve_complex(Y_mat, rhs)
        else:
            v_unknown = []

        V_full = [0j] * n_nodes
        V_full[node_index[src]] = V0
        for n, val in zip(unknown_nodes, v_unknown):
            V_full[node_index[n]] = val

        # Branch currents phasor values
        branch_phasors: list[complex] = []
        for seg in segments:
            gamma = seg.propagation_constant(frequency) * seg.length
            Z0 = seg.characteristic_impedance(frequency)
            sinh_gl = cmath.sinh(gamma)
            cosh_gl = cmath.cosh(gamma)
            if sinh_gl == 0:
                Y_self = 1.0 / Z0
                Y_off = -1.0 / Z0
            else:
                Y_self = (1.0 / Z0) * (cosh_gl / sinh_gl)
                Y_off = -(1.0 / Z0) * (1.0 / sinh_gl)
            i = node_index[seg.from_node]
            j = node_index[seg.to_node]
            I = Y_self * V_full[i] + Y_off * V_full[j]
            branch_phasors.append(I)

        # Total current from source
        I_src = 0.0 + 0.0j
        for seg, I in zip(segments, branch_phasors):
            if seg.from_node == src:
                I_src += I
            elif seg.to_node == src:
                I_src -= I

        # Generate time series for source and load nodes
        load_node = segments[-1].to_node
        node_voltages = xp.zeros((n_steps, 2))
        amp_src = abs(V_full[node_index[src]])
        phase_src = cmath.phase(V_full[node_index[src]])
        amp_load = abs(V_full[node_index[load_node]])
        phase_load = cmath.phase(V_full[node_index[load_node]])
        node_voltages[:, 0] = xp.sin(w * t + phase_src) * amp_src
        node_voltages[:, 1] = xp.sin(w * t + phase_load) * amp_load

        branch_currents = xp.zeros((n_steps, len(branch_phasors)))
        for b, ph in enumerate(branch_phasors):
            amp = abs(ph)
            phase = cmath.phase(ph)
            branch_currents[:, b] = xp.sin(w * t + phase) * amp

        amp_I = abs(I_src)
        phase_I = cmath.phase(I_src)
        total_I = xp.array([xp.sin(w * ti + phase_I) for ti in t]) * amp_I

        # Couple to optional EM solver using the generated time series
        if em_solver is not None:
            em_state: Any | None = None
            for idx in range(n_steps):
                I_src_t = float(to_cpu(total_I[idx]))
                V_in_t = float(to_cpu(vin[idx]))
                em_state = em_solver.step(em_state, dt, I_src_t, V_in_t)
                fb = em_solver.coupling_interface()
                # Apply feedback to source signals and ensure all time-series
                # arrays remain consistent.  A full network recomputation would
                # be expensive, so the feedback is applied uniformly across all
                # nodes and branches.
                total_I[idx] += fb.back_reaction
                vin[idx] += fb.voltage
                node_voltages[idx, :] += fb.voltage
                branch_currents[idx, :] += fb.back_reaction

        # Reflection coefficients for backward compatibility
        reflections: list[complex] = []
        Z_eff = ZL
        for seg in reversed(segments):
            refl = seg.reflection_coefficient(frequency, Z_eff)
            reflections.insert(0, refl)
            Z0 = seg.characteristic_impedance(frequency)
            gamma = seg.propagation_constant(frequency)
            tanh_gl = cmath.tanh(gamma * seg.length)
            Z_eff = Z0 * (Z_eff + Z0 * tanh_gl) / (Z0 + Z_eff * tanh_gl)

        return DistributedRLCSolution(
            t=to_cpu(t),
            current=to_cpu(total_I),
            voltage=to_cpu(vin),
            branch_currents=to_cpu(branch_currents),
            node_voltages=to_cpu(node_voltages),
            reflections=reflections,
        )

    # ------------------------------------------------------------------
    # Determine topology
    nodes: set[int] = set()
    for seg in segments:
        nodes.add(seg.from_node)
        nodes.add(seg.to_node)
    for sw in switches:
        nodes.add(sw.from_node)
        nodes.add(sw.to_node)
    if not nodes:
        return DistributedRLCSolution(
            t=to_cpu(xp.zeros(0)),
            current=to_cpu(xp.zeros(0)),
            voltage=to_cpu(xp.zeros(0)),
            branch_currents=to_cpu(xp.zeros((0, 0))),
            node_voltages=to_cpu(xp.zeros((0, 0))),
        )

    node_list = sorted(nodes)
    src = node_list[0]
    ground = node_list[-1]
    node_index = {n: i for i, n in enumerate(node_list)}

    # Branch definition helper -------------------------------------------------
    class _Branch:
        __slots__ = ("from_node", "to_node", "L", "R", "delay_steps")

        def __init__(self, from_node: int, to_node: int, L: float, R: float, delay_steps: int = 0):
            self.from_node = from_node
            self.to_node = to_node
            self.L = L
            self.R = R
            self.delay_steps = delay_steps

    branches: list[_Branch] = []
    plasma_L = 0.0
    if em_solver is not None:
        try:
            plasma_L = float(getattr(em_solver.coupling_interface(), "Lp", 0.0))
        except Exception:
            plasma_L = 0.0

    def _update_branch_lists(t: float) -> float:
        branches.clear()

        def build_branch(seg: TransmissionLineSegment):
            L, R, _ = seg.totals(t, frequency)
            if L == 0.0 and R == 0.0:
                return None  # pure capacitive branch handled via C matrix
            delay_steps = int(round(seg.delay() / dt)) if hasattr(seg, "delay") else 0
            return _Branch(seg.from_node, seg.to_node, L or 1e-12, R, delay_steps)

        if n_threads > 1:
            with ThreadPoolExecutor(max_workers=n_threads) as ex:
                for br in ex.map(build_branch, segments):
                    if br is not None:
                        branches.append(br)
        else:
            for seg in segments:
                br = build_branch(seg)
                if br is not None:
                    branches.append(br)
        for sw in switches:
            branches.append(
                _Branch(sw.from_node, sw.to_node, sw.L_parasitic or 1e-12, sw.resistance(t), 0)
            )
        if plasma_L != 0.0:
            branches.append(_Branch(src, ground, plasma_L or 1e-12, 0.0, 0))
        # Use the matrix assembly helper to determine the total capacitance
        _, _, C_mat = assemble_matrices(segments, switches, t)
        size = C_mat.shape[0] if hasattr(C_mat, "shape") else 0
        if not size:
            return 0.0
        diag_sum = 0.0
        for i in range(size):
            diag_sum += C_mat[i][i]
        return float(diag_sum) / 2.0

    # Initial branch list and total capacitance
    C_total = _update_branch_lists(0.0)

    n_nodes = len(node_list)
    n_branches = len(branches)

    n_steps = int(t_end / dt) + 1
    t = [i * dt for i in range(n_steps)]

    currents = xp.zeros((n_steps, n_branches))
    node_voltages = xp.zeros((n_steps, n_nodes))
    total_I = xp.zeros(n_steps)
    V_cap = xp.zeros(n_steps)

    total_I[0] = I0
    V_cap[0] = V0
    node_voltages[0, node_index[src]] = V0

    # ------------------------------------------------------------------
    def _solve(M, b):
        """Solve ``M x = b`` using the accelerated backend."""

        return solve_linear(M, b)

    # ------------------------------------------------------------------
    em_state: Any | None = None

    for k in range(1, n_steps):
        tk = t[k - 1]

        # Update branch parameters and total capacitance (allows time dependence)
        C_total = _update_branch_lists(tk)

        unknown_nodes = [n for n in node_list if n not in (src, ground)]
        n_unknown = len(unknown_nodes)
        unk_index = {n: i for i, n in enumerate(unknown_nodes)}

        M = xp.zeros((n_unknown, n_unknown))
        rhs = xp.zeros(n_unknown)

        # Pre-compute constants for each branch
        a_vals = [0.0] * n_branches
        b_vals = [0.0] * n_branches
        for idx_b, br in enumerate(branches):
            I_prev = currents[k - 1, idx_b]
            L = br.L
            R = br.R
            a = I_prev - dt * (R * I_prev) / L
            b = dt / L
            a_vals[idx_b] = a
            b_vals[idx_b] = b

            i, j = br.from_node, br.to_node

            if i not in (src, ground):
                ii = unk_index[i]
                rhs[ii] -= a
                M[ii, ii] += b
                if j not in (src, ground):
                    jj = unk_index[j]
                    M[ii, jj] -= b
                else:
                    vj = V_cap[k - 1] if j == src else 0.0
                    rhs[ii] += b * vj
            if j not in (src, ground):
                jj = unk_index[j]
                rhs[jj] += a
                M[jj, jj] += b
                if i not in (src, ground):
                    ii = unk_index[i]
                    M[jj, ii] -= b
                else:
                    vi = V_cap[k - 1] if i == src else 0.0
                    rhs[jj] += b * vi

        if n_unknown:
            v_unknown = _solve(M, rhs)
        else:
            v_unknown = xp.zeros(0)

        # Compose full node voltage vector
        v_full = xp.zeros(n_nodes)
        v_full[node_index[src]] = V_cap[k - 1]
        v_full[node_index[ground]] = 0.0
        for n in unknown_nodes:
            v_full[node_index[n]] = v_unknown[unk_index[n]] if n_unknown else 0.0

        # Update branch currents with solved voltages
        if _currents_kernel is not None and n_branches:
            vi_arr = xp.zeros(n_branches)
            vj_arr = xp.zeros(n_branches)
            R_arr = xp.array([br.R for br in branches])
            L_arr = xp.array([br.L for br in branches])
            for idx_b, br in enumerate(branches):
                i = node_index[br.from_node]
                j = node_index[br.to_node]
                if br.delay_steps > 0 and k - br.delay_steps >= 0:
                    vi = node_voltages[k - br.delay_steps, i]
                    vj = node_voltages[k - br.delay_steps, j]
                elif br.delay_steps > 0:
                    vi = node_voltages[0, i]
                    vj = node_voltages[0, j]
                else:
                    vi = v_full[i]
                    vj = v_full[j]
                vi_arr[idx_b] = vi
                vj_arr[idx_b] = vj
            currents[k] = _currents_kernel(
                currents[k - 1], vi_arr, vj_arr, R_arr, L_arr, dt
            )
        else:
            for idx_b, br in enumerate(branches):
                i = node_index[br.from_node]
                j = node_index[br.to_node]
                if br.delay_steps > 0 and k - br.delay_steps >= 0:
                    vi = node_voltages[k - br.delay_steps, i]
                    vj = node_voltages[k - br.delay_steps, j]
                elif br.delay_steps > 0:
                    vi = node_voltages[0, i]
                    vj = node_voltages[0, j]
                else:
                    vi = v_full[i]
                    vj = v_full[j]
                dIdt = (vi - vj - br.R * currents[k - 1, idx_b]) / br.L
                currents[k, idx_b] = currents[k - 1, idx_b] + dIdt * dt

        node_voltages[k] = v_full

        # Total current leaving the source node
        tot = 0.0
        for idx_b, br in enumerate(branches):
            if br.from_node == src:
                tot += currents[k, idx_b]
            elif br.to_node == src:
                tot -= currents[k, idx_b]

        # Couple to optional EM solver
        if em_solver is not None:
            em_state = em_solver.step(em_state, dt, tot, V_cap[k - 1])
            fb = em_solver.coupling_interface()
            # Apply feedback uniformly so the precomputed branch currents and
            # node voltages remain consistent with the adjusted source signals.
            tot += fb.back_reaction
            currents[k, :] += fb.back_reaction
            node_voltages[k, :] += fb.voltage
            plasma_L = getattr(fb, "Lp", plasma_L)
            if hasattr(em_solver, "plasma_inductance"):
                try:
                    plasma_L = float(em_solver.plasma_inductance(em_state))
                except Exception:
                    pass

        total_I[k] = tot

        # Capacitor voltage update
        if C_total > 0.0:
            V_cap[k] = V_cap[k - 1] - dt * tot / C_total
        else:
            V_cap[k] = V_cap[k - 1]

        for sw in switches:
            sw.update(tk + dt)

    return DistributedRLCSolution(
        t=to_cpu(xp.array(t)),
        current=to_cpu(xp.array(total_I)),
        voltage=to_cpu(xp.array(V_cap)),
        branch_currents=to_cpu(xp.array(currents)),
        node_voltages=to_cpu(xp.array(node_voltages)),
    )
