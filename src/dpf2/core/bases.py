"""This file provides canonical interfaces for plasma, circuit and
diagnostics modules. Other copies in the repository should import
from here to avoid duplication."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class PlasmaSolverBase(ABC):
    """Interface for plasma solvers coupled to an external circuit."""

    @abstractmethod
    def step(self, state: Any, dt: float, current: float, voltage: float) -> Any:
        """Advance the plasma state by ``dt`` seconds.

        Parameters
        ----------
        state:
            Current plasma state object.
        dt:
            Time step in seconds.
        current, voltage:
            Instantaneous circuit current and capacitor voltage supplied by
            the circuit solver for coupling.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    def coupling_interface(self) -> dict[str, float]:
        """Return circuit coupling terms for the current plasma state.

        The mapping should contain at least the plasma inductance ``Lp`` in
        Henries and the induced electromotive force ``emf`` in Volts.  Plasma
        solvers that do not actively couple to the circuit may rely on this
        default implementation which returns zero for both quantities.
        """

        return {"Lp": 0.0, "emf": 0.0}


class CircuitSolverBase(ABC):
    """Interface for external circuit solvers."""

    @abstractmethod
    def step(
        self,
        current: float,
        back_emf: float,
        dt: float,
        plasma_feedback: dict[str, float] | None = None,
    ) -> tuple[float, float]:
        """Return updated ``(current, voltage)`` after ``dt`` seconds."""
        raise NotImplementedError


class DiagnosticsBase(ABC):
    """Interface for simulation diagnostics."""

    @abstractmethod
    def record(self, state: Any, time: float) -> None:
        """Record the simulation ``state`` at ``time``."""
        raise NotImplementedError


__all__ = [
    "PlasmaSolverBase",
    "CircuitSolverBase",
    "DiagnosticsBase",
]
