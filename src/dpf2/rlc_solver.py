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
from typing import Sequence

# ``numpy`` may be replaced by a light‑weight stub in the test environment but
# it provides the minimal functionality used below (``array``).
import numpy as np

from .circuit.distributed import TransmissionLineSegment, TriggeredSwitch, assemble_matrices

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
    """Container returned by :func:`solve_distributed_circuit`."""

    t: np.ndarray
    current: np.ndarray
    voltage: np.ndarray
    branch_currents: np.ndarray
    node_voltages: np.ndarray


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
) -> DistributedRLCSolution:
    """Integrate an RLC network using a very small nodal analysis scheme.

    The implementation is intentionally compact and supports only the features
    required by the unit tests.  All capacitors are aggregated into a single
    equivalent element between the first and last nodes.  Inductive branches are
    solved via a linear system enforcing Kirchhoff's current law at each node.
    """

    switches = list(switches or [])

    # Simplified analytic handling for pure transmission line problems.  If a
    # ``frequency`` is supplied and all segments define a propagation velocity we
    # treat the network as a cascade of delay lines with optional attenuation due
    # to a very small skin‑effect resistance model.  This path is primarily used
    # in regression tests and bypasses the general time domain solver.
    if frequency is not None and segments and all(seg.propagation_velocity for seg in segments):
        w = 2.0 * np.pi * frequency
        n_steps = int(t_end / dt) + 1
        t = np.array([i * dt for i in range(n_steps)])
        vin = np.array([np.sin(w * ti) for ti in t]) * V0
        total_delay = sum(seg.delay() for seg in segments)
        attenuation = 1.0
        total_R = 0.0
        for seg in segments:
            if seg.skin_effect_coeff:
                attenuation *= np.exp(-seg.skin_effect_coeff * seg.length * np.sqrt(frequency))
            total_R += seg.R_per_m * seg.length + seg.R_parasitic
            if seg.skin_effect_coeff:
                total_R += seg.skin_effect_coeff * seg.length * np.sqrt(frequency)
        vout = np.array([np.sin(w * (ti - total_delay)) for ti in t]) * (attenuation * V0)
        node_voltages = np.zeros((len(t), 2))
        node_voltages[:, 0] = vin
        node_voltages[:, 1] = vout
        current = (vin - vout) / (total_R if total_R != 0 else 1e-12)
        branch_currents = current[:, None]
        return DistributedRLCSolution(
            t=t,
            current=current,
            voltage=vin,
            branch_currents=branch_currents,
            node_voltages=node_voltages,
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
            t=np.zeros(0),
            current=np.zeros(0),
            voltage=np.zeros(0),
            branch_currents=np.zeros((0, 0)),
            node_voltages=np.zeros((0, 0)),
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

    def _update_branch_lists(t: float) -> float:
        branches.clear()
        for seg in segments:
            L, R, _ = seg.totals(t, frequency)
            if L == 0.0 and R == 0.0:
                continue  # pure capacitive branch handled via C matrix
            delay_steps = int(round(seg.delay() / dt)) if hasattr(seg, "delay") else 0
            branches.append(_Branch(seg.from_node, seg.to_node, L or 1e-12, R, delay_steps))
        for sw in switches:
            branches.append(
                _Branch(sw.from_node, sw.to_node, sw.L_parasitic or 1e-12, sw.resistance(t), 0)
            )
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

    currents = np.zeros((n_steps, n_branches))
    node_voltages = np.zeros((n_steps, n_nodes))
    total_I = np.zeros(n_steps)
    V_cap = np.zeros(n_steps)

    total_I[0] = I0
    V_cap[0] = V0
    node_voltages[0, node_index[src]] = V0

    # ------------------------------------------------------------------
    def _solve(M, b):
        """Solve ``M x = b`` using ``numpy.linalg.solve`` if available."""

        try:  # pragma: no cover - prefer real numpy implementation
            return np.linalg.solve(M, b)  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - lightweight fallback
            # Very small dense Gaussian elimination suitable for the tests
            M = [[float(M[i][j]) for j in range(len(b))] for i in range(len(b))]
            b = [float(bb) for bb in b]
            n = len(b)
            for i in range(n):
                pivot = M[i][i]
                if pivot == 0.0:
                    for j in range(i + 1, n):
                        if M[j][i] != 0.0:
                            M[i], M[j] = M[j], M[i]
                            b[i], b[j] = b[j], b[i]
                            pivot = M[i][i]
                            break
                factor = pivot
                for j in range(i, n):
                    M[i][j] /= factor
                b[i] /= factor
                for k in range(n):
                    if k == i:
                        continue
                    factor = M[k][i]
                    for j in range(i, n):
                        M[k][j] -= factor * M[i][j]
                    b[k] -= factor * b[i]
            return np.array(b)

    # ------------------------------------------------------------------
    for k in range(1, n_steps):
        tk = t[k - 1]

        # Update branch parameters and total capacitance (allows time dependence)
        C_total = _update_branch_lists(tk)

        unknown_nodes = [n for n in node_list if n not in (src, ground)]
        n_unknown = len(unknown_nodes)
        unk_index = {n: i for i, n in enumerate(unknown_nodes)}

        M = np.zeros((n_unknown, n_unknown))
        rhs = np.zeros(n_unknown)

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
            v_unknown = np.zeros(0)

        # Compose full node voltage vector
        v_full = np.zeros(n_nodes)
        v_full[node_index[src]] = V_cap[k - 1]
        v_full[node_index[ground]] = 0.0
        for n in unknown_nodes:
            v_full[node_index[n]] = v_unknown[unk_index[n]] if n_unknown else 0.0

        # Update branch currents with solved voltages
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
        total_I[k] = tot

        # Capacitor voltage update
        if C_total > 0.0:
            V_cap[k] = V_cap[k - 1] - dt * tot / C_total
        else:
            V_cap[k] = V_cap[k - 1]

        for sw in switches:
            sw.update(tk + dt)

    return DistributedRLCSolution(
        t=np.array(t),
        current=np.array(total_I),
        voltage=np.array(V_cap),
        branch_currents=np.array(currents),
        node_voltages=np.array(node_voltages),
    )
