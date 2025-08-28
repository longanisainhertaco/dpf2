from __future__ import annotations

# Backward compatibility wrapper
from typing import Any

from .pinch_models import (
    AnalyticPinchModel,
    PinchResult,
    SemiAnalyticPinchModel,
    PinchModelBase,
)
from .ablation import ablation_mass_energy_source, insulator_sleeve_area
from .core.bases import CouplingState, PlasmaSolverBase


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
    return CouplingState(
        Lp=fb.Lp,
        emf=fb.emf,
        current=coupling.current,
        voltage=coupling.voltage,
        mutual_inductance=fb.mutual_inductance,
        back_reaction=fb.back_reaction,
    )

__all__ = [
    "AnalyticPinchModel",
    "PinchResult",
    "SemiAnalyticPinchModel",
    "PinchModelBase",
    "ablation_mass_energy_source",
    "insulator_sleeve_area",
    "advance_plasma_with_circuit",
]
