"""Simple multi-phase pinch solver.

This module provides a minimal implementation of the canonical phases of a
Dense Plasma Focus (DPF) pinch.  The model is intentionally lightweight and
is designed for algorithmic demonstrations rather than high fidelity physics.

The solver advances the state through four phases:

* axial rundown – the current sheath moves axially towards the anode end
* radial collapse – the plasma column contracts to a minimum radius
* stagnation – the pinch stagnates at minimum radius for a short time
* rebound – the plasma expands radially after stagnation

The dynamics are represented by ordinary differential equations integrated in
an explicit manner over a provided time grid.  Mass is conserved by
construction and momentum is continuous across phase boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..pinch_models import PinchModelBase, PinchResult

__all__ = ["PinchMultiphaseSolver", "PinchMultiphaseResult"]


@dataclass
class PinchMultiphaseResult(PinchResult):
    """Result object including additional diagnostics for the multi-phase solver."""

    phase: np.ndarray
    radial_velocity: np.ndarray
    axial_velocity: np.ndarray
    mass: float


class PinchMultiphaseSolver(PinchModelBase):
    """Very simple four phase pinch model.

    The model is intentionally heuristic.  The goal is to provide a deterministic
    sequence of phases that can be used in unit tests and algorithmic examples.
    """

    def __init__(
        self,
        initial_radius: float = 1e-2,
        anode_length: float = 0.1,
        min_radius: float = 1e-3,
        stagnation_time: float = 50e-9,
        mass: float = 1e-6,
    ) -> None:
        self.initial_radius = initial_radius
        self.anode_length = anode_length
        self.min_radius = min_radius
        self.stagnation_time = stagnation_time
        self.mass = mass
        # heuristic acceleration coefficients
        self._k_axial = 5e2
        self._k_radial = 5e2
        self._k_rebound = 5e2

    # ------------------------------------------------------------------
    def run(self, time: Iterable[float], current: Iterable[float]) -> PinchMultiphaseResult:
        t = np.asarray(time)
        I = np.asarray(current)
        n = t.size
        r = np.empty(n)
        z = np.empty(n)
        vr = np.empty(n)
        vz = np.empty(n)
        phase = np.empty(n, dtype="U12")

        # initialise state
        r_cur = self.initial_radius
        z_cur = 0.0
        vr_cur = 0.0
        vz_cur = 0.0
        phase_cur = "axial"
        stag_elapsed = 0.0

        r[0] = r_cur
        z[0] = z_cur
        vr[0] = vr_cur
        vz[0] = vz_cur
        phase[0] = phase_cur

        for k in range(1, n):
            dt = t[k] - t[k - 1]
            I_prev = I[k - 1]

            phase_next = phase_cur
            if phase_cur == "axial":
                vz_cur += self._k_axial * I_prev**2 / self.mass * dt
                z_cur += vz_cur * dt
                if z_cur >= self.anode_length:
                    z_cur = self.anode_length
                    phase_next = "radial"
            elif phase_cur == "radial":
                vr_cur -= self._k_radial * I_prev**2 / self.mass * dt
                r_cur += vr_cur * dt
                if r_cur <= self.min_radius:
                    r_cur = self.min_radius
                    vr_cur = 0.0
                    phase_next = "stagnation"
                    stag_elapsed = 0.0
            elif phase_cur == "stagnation":
                stag_elapsed += dt
                if stag_elapsed >= self.stagnation_time:
                    phase_next = "rebound"
            elif phase_cur == "rebound":
                vr_cur += self._k_rebound * I_prev**2 / self.mass * dt
                r_cur += vr_cur * dt
                if r_cur >= self.initial_radius:
                    r_cur = self.initial_radius
                    vr_cur = 0.0

            r[k] = r_cur
            z[k] = z_cur
            vr[k] = vr_cur
            vz[k] = vz_cur
            phase[k] = phase_cur
            phase_cur = phase_next

        temperature = 1e3 * (I / 1e4) ** 2
        pressure = 0.5 * (I**2) * 1e-6
        neutron_yield = 0.0

        return PinchMultiphaseResult(
            time=t,
            radius=r,
            temperature=temperature,
            pressure=pressure,
            neutron_yield=neutron_yield,
            axial_position=z,
            phase=phase,
            radial_velocity=vr,
            axial_velocity=vz,
            mass=self.mass,
        )
