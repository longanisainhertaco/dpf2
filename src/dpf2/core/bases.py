"""Abstract base classes for solver components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class PlasmaSolverBase(ABC):
    """Interface for plasma solvers."""

    @abstractmethod
    def step(self, state: Any, dt: float) -> Any:
        """Advance the plasma state by ``dt`` seconds."""
        raise NotImplementedError


class CircuitSolverBase(ABC):
    """Interface for external circuit solvers."""

    @abstractmethod
    def step(self, current: float, voltage: float, dt: float) -> tuple[float, float]:
        """Return updated (current, voltage) after ``dt`` seconds."""
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
