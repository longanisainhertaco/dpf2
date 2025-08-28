"""This file provides canonical interfaces for plasma, circuit and
diagnostics modules. Other copies in the repository should import
from here to avoid duplication."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class CouplingState:
    """Coupling information exchanged between plasma and circuit solvers.

    Attributes
    ----------
    Lp, emf:
        Plasma inductance in Henries and induced electromotive force in
        Volts.
    current, voltage:
        Circuit current and capacitor voltage supplied to the plasma
        solver for advancing its state.
    """

    Lp: float = 0.0
    emf: float = 0.0
    current: float = 0.0
    voltage: float = 0.0


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
    def coupling_interface(self) -> CouplingState:
        """Return circuit coupling terms for the current plasma state.

        Plasma solvers that do not actively couple to the circuit may rely on
        this default implementation which returns zeros for all quantities.
        """

        return CouplingState()


class CircuitSolverBase(ABC):
    """Interface for external circuit solvers."""

    @abstractmethod
    def step(
        self,
        coupling: CouplingState,
        back_emf: float,
        dt: float,
    ) -> CouplingState:
        """Advance the circuit state by ``dt`` seconds."""
        raise NotImplementedError


class DiagnosticsBase(ABC):
    """Interface for simulation diagnostics."""

    @abstractmethod
    def record(self, state: Any, time: float) -> None:
        """Record the simulation ``state`` at ``time``."""
        raise NotImplementedError


__all__ = [
    "CouplingState",
    "PlasmaSolverBase",
    "CircuitSolverBase",
    "DiagnosticsBase",
]
