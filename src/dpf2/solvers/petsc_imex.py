"""Lightweight PETSc-backed IMEX integrator for Hall-MHD closures.

The helper is deliberately minimal: it accepts explicit and implicit
residual callbacks operating on flattened NumPy arrays and either delegates
to ``petsc4py``'s ARKIMEX time-stepper or falls back to a single first-order
IMEX update when PETSc is unavailable.  The class is designed so solvers in
``src/dpf2`` can supply small closures for stiff Braginskii terms and
transport-driven source terms without re-implementing TS wiring.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import logging
import numpy as np

try:  # pragma: no cover - petsc4py optional in CI
    from petsc4py import PETSc
except Exception:  # pragma: no cover
    PETSc = None

logger = logging.getLogger(__name__)

ExplicitFunc = Callable[[float, np.ndarray], np.ndarray]
ImplicitFunc = Callable[[float, np.ndarray], np.ndarray]


@dataclass
class PetscIMEXStepper:
    """Wrapper around PETSc ARKIMEX with a graceful pure-Python fallback."""

    use_petsc: bool = True
    options_prefix: str | None = None

    def __post_init__(self) -> None:
        self._available = bool(self.use_petsc and PETSc is not None)
        self._ts: PETSc.TS | None = None
        if not self._available:
            logger.info("petsc4py not available; falling back to explicit IMEX update")
            return

        self._ts = PETSc.TS().create()
        self._ts.setType(PETSc.TS.Type.ARKIMEX)
        self._ts.setProblemType(PETSc.TS.ProblemType.NONLINEAR)
        self._ts.setMaxSteps(1)
        if self.options_prefix:
            self._ts.setOptionsPrefix(self.options_prefix)
        self._ts.setFromOptions()

    @property
    def available(self) -> bool:
        """Return ``True`` when PETSc backing is active."""

        return self._available

    def advance(
        self,
        state: np.ndarray,
        explicit_rhs: ExplicitFunc,
        implicit_rhs: ImplicitFunc,
        dt: float,
    ) -> np.ndarray:
        """Advance ``state`` by ``dt`` using IMEX splitting.

        Parameters
        ----------
        state:
            Flattenable state vector.
        explicit_rhs, implicit_rhs:
            Callables returning ``dstate/dt`` for the explicit and implicit
            partitions.  They receive ``(time, state)`` and must return arrays
            with the same shape as ``state``.
        dt:
            Time step size in seconds.
        """

        flat_state = np.asarray(state, dtype=float).ravel()

        if not self._available or self._ts is None:
            return flat_state + dt * (
                np.asarray(explicit_rhs(0.0, flat_state))
                + np.asarray(implicit_rhs(0.0, flat_state))
            )

        ts = self._ts

        def _ifunc(ts_obj, t, x, xdot, f) -> None:  # pragma: no cover - petsc only
            x_arr = x.getArray(readonly=True)
            xdot_arr = xdot.getArray(readonly=True)
            f_arr = f.getArray()
            f_arr[:] = xdot_arr - implicit_rhs(t, x_arr)

        def _rhs(ts_obj, t, x, f) -> None:  # pragma: no cover - petsc only
            x_arr = x.getArray(readonly=True)
            f_arr = f.getArray()
            f_arr[:] = explicit_rhs(t, x_arr)

        vec = PETSc.Vec().createWithArray(flat_state, bsize=flat_state.size)
        ts.setIFunction(_ifunc)
        ts.setRHSFunction(_rhs)
        ts.setSolution(vec)
        ts.setTimeStep(dt)
        ts.setMaxTime(dt)
        ts.setExactFinalTime(PETSc.TS.ExactFinalTime.STEPOVER)

        ts.solve(None)
        result = ts.getSolution().getArray().copy()
        return result.reshape(state.shape)


__all__ = ["PetscIMEXStepper"]
