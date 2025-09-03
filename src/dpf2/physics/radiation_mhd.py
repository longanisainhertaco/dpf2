from __future__ import annotations

"""Minimal 3-D radiation magnetohydrodynamics solver."""

from dataclasses import dataclass, field
from typing import Callable, Tuple, Any

import numpy as np

try:  # pragma: no cover - GPU backend optional
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover
    cp = None  # type: ignore

from dpf2.core.bases import PlasmaSolverBase, CouplingState
from ..eos import EOSBase, IdealGasEOS

Array = Any


@dataclass
class AMRGrid:
    """Placeholder structure representing an AMR hierarchy.

    The ``refine`` method examines a dummy array with the same shape as the
    grid.  Whenever the user supplied ``criterion`` evaluates ``True`` a single
    child grid with doubled resolution is attached.  This provides a very small
    stand‑in for a real AMR hierarchy used purely in tests.
    """

    level: int
    shape: Tuple[int, int, int]
    parent: "AMRGrid | None" = None
    children: list["AMRGrid"] = field(default_factory=list)

    def refine(self, criterion: Callable[[Array], Array]) -> None:
        """Refine cells according to ``criterion``."""

        dummy = np.zeros(self.shape)
        mask = criterion(dummy)
        if isinstance(mask, bool):
            cond = mask
        else:
            cond = bool(np.any(mask))
        if cond:
            child_shape = tuple(s * 2 for s in self.shape)
            child = AMRGrid(level=self.level + 1, shape=child_shape, parent=self)
            self.children.append(child)


@dataclass
class RadiationMHDState:
    """Container holding the solver state variables."""

    density: Array
    velocity: Array
    magnetic: Array
    energy: Array
    electron_temp: Array | None = None
    grid: AMRGrid | None = None


class RadiationMHDSolver(PlasmaSolverBase):
    """Light‑weight 3‑D radiation‑MHD solver interface.

    The implementation is intentionally compact and targets the unit tests.  It
    exposes a minimal feature set exercising AMR, Hall physics and two
    temperature evolution together with a simple radiation loss model that feeds
    back to the external circuit through :class:`~dpf2.core.bases.CouplingState`.
    """

    def __init__(
        self,
        eos: EOSBase | None = None,
        *,
        use_hall: bool = False,
        two_temperature: bool = False,
        use_gpu: bool = False,
        refine_criterion: Callable[[Array], Array] | None = None,
    ) -> None:
        self.eos = eos or IdealGasEOS()
        self.use_hall = use_hall
        self.two_temperature = two_temperature
        self.use_gpu = bool(use_gpu and cp is not None)
        self.xp = cp if self.use_gpu else np
        self.refine_criterion = refine_criterion
        self.circuit_feedback = CouplingState()

    # ------------------------------------------------------------------
    def allocate_state(self, shape: Tuple[int, int, int]) -> RadiationMHDState:
        """Allocate a fresh state filled with zeros."""

        xp = self.xp
        rho = xp.zeros(shape)
        vel = xp.zeros(shape + (3,))
        B = xp.zeros(shape + (3,))
        e = xp.zeros(shape)
        Te = xp.zeros(shape) if self.two_temperature else None
        grid = AMRGrid(level=0, shape=shape)
        return RadiationMHDState(rho, vel, B, e, Te, grid)

    # ------------------------------------------------------------------
    def step(
        self,
        state: RadiationMHDState | None,
        dt: float,
        current: float,
        voltage: float,
    ) -> RadiationMHDState:
        """Advance the state by ``dt`` seconds (placeholder update)."""

        if state is None:
            state = self.allocate_state((1, 1, 1))

        xp = self.xp
        size = 1
        for s in getattr(state.energy, "shape", []):
            size *= s
        total_before = float(xp.sum(state.energy))

        # Two-temperature energy evolution (placeholder coupling).
        if self.two_temperature and state.electron_temp is not None:
            state.electron_temp = state.electron_temp + 0.1 * dt

        # Hall term placeholder acting on the magnetic field.
        if self.use_hall:
            state.magnetic = state.magnetic + dt * current

        # Adaptive mesh refinement.
        if self.refine_criterion and state.grid is not None:
            state.grid.refine(self.refine_criterion)

        # Radiation losses proportional to the supplied electrical power.
        power = abs(current * voltage)
        radiation_loss = 0.1 * power * dt
        if size:
            loss_density = radiation_loss / size
            state.energy = state.energy - loss_density
            if self.two_temperature and state.electron_temp is not None:
                state.electron_temp = state.electron_temp - loss_density

        total_after = float(xp.sum(state.energy))
        delta = total_after - total_before  # negative when energy is lost

        # Convert energy change to an effective back‑reaction voltage.
        if current != 0.0:
            back_reaction = -delta / (dt * current)
        else:  # Fallback when current is zero – interpret power as voltage
            back_reaction = -delta / dt

        self.circuit_feedback = CouplingState(
            current=current, voltage=voltage, back_reaction=back_reaction
        )
        return state

    # ------------------------------------------------------------------
    def coupling_interface(self) -> CouplingState:  # pragma: no cover - trivial
        """Return circuit coupling terms."""

        return CouplingState(back_reaction=self.circuit_feedback.back_reaction)


__all__ = ["AMRGrid", "RadiationMHDState", "RadiationMHDSolver"]
