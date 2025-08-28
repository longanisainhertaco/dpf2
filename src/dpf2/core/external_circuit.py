from __future__ import annotations

from dataclasses import dataclass

from .bases import CircuitSolverBase, CouplingState


@dataclass
class ExternalCircuit(CircuitSolverBase):
    """Minimal external circuit model.

    The circuit stores a time-varying inductance and integrates the current in
    response to an applied voltage using the simple relation ``V = L dI/dt``.
    This is intended solely for unit tests and does not represent a complete
    circuit model.
    """

    inductance: float = 1.0
    current: float = 0.0

    def step(
        self,
        coupling: CouplingState,
        back_emf: float,
        dt: float,
    ) -> CouplingState:
        """Advance the circuit state by ``dt`` seconds."""

        Lp = coupling.Lp
        emf = coupling.emf
        current = coupling.current
        M = coupling.mutual_inductance
        back_reaction = coupling.back_reaction

        Ltot = self.inductance + Lp
        dI = (back_emf - emf) * dt / max(Ltot, 1.0e-30)
        self.current = current + dI
        self.inductance = Ltot
        return CouplingState(
            Lp=Lp,
            emf=emf,
            current=self.current,
            voltage=back_emf,
            mutual_inductance=M,
            back_reaction=back_reaction,
        )


__all__ = ["ExternalCircuit"]
