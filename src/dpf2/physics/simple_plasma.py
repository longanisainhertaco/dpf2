from __future__ import annotations

"""Toy plasma solver providing inductance/EMF feedback to the circuit."""

from dataclasses import dataclass, field
from typing import Any, Callable

# ``MultiGroupDiffusion`` is an optional dependency; provide a lightweight
# fallback to keep this module importable in stripped-down environments.
try:  # pragma: no cover - exercised when radiation package is present
    from ..radiation.multigroup import MultiGroupDiffusion  # type: ignore
except Exception:  # pragma: no cover - fallback for test environment
    class MultiGroupDiffusion:  # type: ignore
        def couple(self, energies, dt):
            return energies

from ..core.bases import PlasmaSolverBase, CouplingState


@dataclass
class ZeroDPlasma(PlasmaSolverBase):
    """Very small plasma model used for tests and examples.

    The solver keeps a dummy state and uses a user supplied ``inductance``
    function to provide coupling terms back to the circuit.  The function is
    expected to return ``(Lp, emf)`` when called with the current simulation
    time, current and voltage.
    """

    inductance_model: Callable[[float, float, float], tuple[float, float]]
    time: float = 0.0
    circuit_feedback: CouplingState = field(init=False, default_factory=CouplingState)
    inductance: float = 0.0
    back_emf: float = 0.0

    def step(self, state: Any, dt: float, current: float, voltage: float) -> Any:
        """Advance the dummy plasma state and compute circuit feedback."""

        self.time += dt
        Lp, emf = self.inductance_model(self.time, current, voltage)
        self.inductance = Lp
        self.back_emf = emf
        self.circuit_feedback = CouplingState(
            Lp=Lp,
            emf=emf,
            current=current,
            voltage=voltage,
            mutual_inductance=0.0,
            back_reaction=0.0,
        )
        return state

    # ------------------------------------------------------------------
    def coupling_interface(self) -> CouplingState:  # pragma: no cover - simple
        """Expose the latest circuit coupling terms."""

        return CouplingState(
            Lp=self.circuit_feedback.Lp,
            emf=self.circuit_feedback.emf,
            mutual_inductance=0.0,
            back_reaction=0.0,
        )

    # ------------------------------------------------------------------
    # Radiation coupling
    # ------------------------------------------------------------------
    def apply_radiation(
        self, energy: float, radiation: MultiGroupDiffusion, dt: float
    ) -> float:
        """Couple a single-cell fluid energy to a multi-group radiation model."""

        updated = radiation.couple([energy], dt)
        return updated[0]


__all__ = ["ZeroDPlasma"]
