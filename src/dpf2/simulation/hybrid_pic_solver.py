"""Hybrid 2D/3D Particle-in-Cell module.

This solver couples a fluid description with a particle description to
capture kinetic effects while retaining a coarse fluid background.  The
implementation focuses on two physical processes relevant to DPF
simulations:

* ``m=0`` interchange growth that can rapidly disrupt the plasma column.
* Mechanism-based anomalous resistivity leading to voltage spikes.

The solver exposes a :class:`~dpf2.core.bases.CouplingState` interface so
that it may be connected to the existing external circuit model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Optional, Callable, Dict

import numpy as np
import logging

from ..core.bases import CouplingState, PlasmaSolverBase


logger = logging.getLogger(__name__)


class _FluidSolver(Protocol):
    """Minimal protocol for fluid solvers used by :class:`HybridPICSolver`."""

    def step(self, state: Any, dt: float, current: float, voltage: float) -> Any:
        ...


class _ParticleSolver(Protocol):
    """Minimal protocol for particle solvers used by :class:`HybridPICSolver`."""

    def step(self, state: Any, dt: float, current: float, voltage: float) -> Any:
        ...

    def beam_current(self) -> float:  # pragma: no cover - trivial protocol
        """Return the instantaneous beam current in Amperes."""
        ...


@dataclass
class HybridPICSolver(PlasmaSolverBase):
    """Hybrid fluid/PIC plasma solver.

    Parameters
    ----------
    fluid:
        Instance implementing the minimal fluid solver protocol.
    particles:
        Instance implementing the minimal particle solver protocol.
    dim:
        Problem dimensionality, either 2 or 3.  The default is three
        dimensions.  Only basic geometry factors depend on ``dim``.
    base_resistivity:
        Background resistivity (Ohm·m) used in the anomalous resistivity
        model.
    J_crit:
        Critical current density (A/m²) above which the anomalous
        resistivity mechanism activates.
    radius:
        Characteristic plasma radius (m) used for ``m=0`` growth estimates.
    """

    fluid: _FluidSolver
    particles: _ParticleSolver
    dim: int = 3
    base_resistivity: float = 1e-4
    J_crit: float = 1e6
    radius: float = 1e-2
    circuit_feedback: CouplingState = field(init=False, default_factory=CouplingState)
    last_voltage_spike: float = field(init=False, default=0.0)
    last_beam_current: float = field(init=False, default=0.0)

    def m0_growth(self, current: float, density: float) -> float:
        """Estimate the linear ``m=0`` interchange growth rate.

        The model is intentionally simplified but retains the expected
        proportionality ``gamma ∝ I / (r * sqrt{ρ})``.
        """

        if density <= 0:
            return 0.0
        return abs(current) / (self.radius * np.sqrt(density))

    def compute_anomalous_resistivity(self, J: float) -> tuple[float, float]:
        """Return (eta, spike) for the given current density ``J``.

        The anomalous resistivity follows a quadratic scaling once a
        threshold current density is exceeded.  The corresponding voltage
        spike ``eta * J`` is also returned.
        """

        if abs(J) <= self.J_crit:
            return self.base_resistivity, 0.0
        excess = abs(J) / self.J_crit
        eta = self.base_resistivity * excess * excess
        spike = eta * abs(J)
        return eta, spike

    def step(self, state: Any, dt: float, current: float, voltage: float, refinement_cb: Optional[Callable[[Dict[str, Any]], Dict[str, int]]] = None) -> Any:
        """Advance both fluid and particle descriptions.

        The method updates the coupled solvers, evaluates ``m=0`` growth
        and anomalous resistivity, and records the beam current reported by
        the particle sub-system.  Circuit coupling is provided through
        :class:`~dpf2.core.bases.CouplingState`.
        """

        # Advance constituent solvers.  State objects are forwarded
        # verbatim to keep the interface lightweight.
        fluid_state = self.fluid.step(state, dt, current, voltage)
        particle_state = self.particles.step(state, dt, current, voltage)

        # Simple cylindrical current density estimate.
        area = np.pi * self.radius ** (2 if self.dim == 3 else 1)
        J = current / max(area, 1e-12)

        eta, spike = self.compute_anomalous_resistivity(J)
        self.last_voltage_spike = spike
        self.last_beam_current = self.particles.beam_current()

        if refinement_cb is not None:
            stats = refinement_cb({"current": J})
            if stats:
                logger.info(f"AMR callback stats: {stats}")

        # A crude inductance model incorporating m=0 growth.  In practice
        # this would be replaced by a full EM field solution.
        density = 1.0  # kg/m^3 placeholder
        growth = self.m0_growth(current, density)
        Lp = self.radius * (1.0 + 0.1 * growth)
        emf = eta * current

        self.circuit_feedback = CouplingState(
            Lp=Lp,
            emf=emf,
            current=current,
            voltage=voltage,
            back_reaction=spike,
        )

        # The solver does not maintain its own state; forward the tuple of
        # the sub-solvers for potential external use.
        return fluid_state, particle_state

    def coupling_interface(self) -> CouplingState:  # pragma: no cover - simple
        return self.circuit_feedback


__all__ = ["HybridPICSolver"]
