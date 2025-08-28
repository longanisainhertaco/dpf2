from __future__ import annotations

# Backward compatibility wrapper
from typing import Any, Sequence
from concurrent.futures import ThreadPoolExecutor

from .pinch_models import (
    AnalyticPinchModel,
    PinchResult,
    SemiAnalyticPinchModel,
    PinchModelBase,
)
from .ablation import ablation_mass_energy_source, insulator_sleeve_area
from .core.bases import CouplingState, PlasmaSolverBase
from .geometry.inductance import loop_mutual_inductance


def advance_plasma_with_circuit(
    plasma: PlasmaSolverBase, state: Any, coupling: CouplingState, dt: float
) -> CouplingState:
    """Advance a plasma solver and return updated coupling terms.

    The helper hides the common ceremony required when coupling a
    :class:`PlasmaSolverBase` implementation to an external circuit.  The
    supplied ``coupling`` provides the instantaneous circuit current and
    voltage which are fed into :meth:`PlasmaSolverBase.step`.  After the
    plasma is advanced, the new inductance and EMF terms are retrieved via
    :meth:`PlasmaSolverBase.coupling_interface` and returned as a fresh
    :class:`CouplingState` instance.
    """

    plasma.step(state, dt, coupling.current, coupling.voltage)
    fb = plasma.coupling_interface()

    # ------------------------------------------------------------------
    # Geometry based mutual inductance
    # ------------------------------------------------------------------
    # Attempt to extract simple geometric information from the plasma solver
    # and state objects.  ``coil_radius`` is expected on the plasma solver
    # instance while the instantaneous plasma ``radius`` and optional
    # ``axial_position`` may be provided on the state.  Missing attributes
    # simply result in zero mutual inductance which retains backwards
    # compatibility with very small test solvers.
    coil_radius = getattr(plasma, "coil_radius", 0.0)
    plasma_radius = 0.0
    axial = 0.0
    if isinstance(state, dict):
        plasma_radius = float(state.get("radius", 0.0))
        axial = float(state.get("axial_position", 0.0))
    else:
        plasma_radius = float(getattr(state, "radius", 0.0))
        axial = float(getattr(state, "axial_position", 0.0))

    if coil_radius > 0.0 and plasma_radius > 0.0:
        M_new = loop_mutual_inductance(coil_radius, plasma_radius, axial)
    else:
        M_new = fb.mutual_inductance

    dMdt = (M_new - coupling.mutual_inductance) / dt if dt > 0 else 0.0
    back_reaction = fb.back_reaction + coupling.current * dMdt

    return CouplingState(
        Lp=fb.Lp,
        emf=fb.emf,
        current=coupling.current,
        voltage=coupling.voltage,
        mutual_inductance=M_new,
        back_reaction=back_reaction,
    )


def advance_plasmas_with_circuit(
    plasmas: Sequence[PlasmaSolverBase],
    states: Sequence[Any],
    couplings: Sequence[CouplingState],
    dt: float,
    n_threads: int = 1,
) -> list[CouplingState]:
    """Advance multiple plasma solvers in parallel.

    Parameters
    ----------
    plasmas, states, couplings:
        Sequences of plasma solvers, their corresponding state objects and
        coupling states.
    dt:
        Timestep in seconds applied to all solvers.
    n_threads:
        Number of worker threads to use.  A value of ``1`` executes the
        updates serially.
    """

    tasks = list(zip(plasmas, states, couplings))

    def _run(args: tuple[PlasmaSolverBase, Any, CouplingState]) -> CouplingState:
        p, s, c = args
        return advance_plasma_with_circuit(p, s, c, dt)

    if n_threads > 1:
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            return list(ex.map(_run, tasks))
    return [_run(t) for t in tasks]

__all__ = [
    "AnalyticPinchModel",
    "PinchResult",
    "SemiAnalyticPinchModel",
    "PinchModelBase",
    "ablation_mass_energy_source",
    "insulator_sleeve_area",
    "advance_plasma_with_circuit",
    "advance_plasmas_with_circuit",
]
