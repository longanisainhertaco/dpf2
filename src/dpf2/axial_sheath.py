from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import warnings

from .diagnostics import magnetic_reynolds_number

mu0 = 4e-7 * np.pi


@dataclass
class SheathResult:
    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    jxb_force: np.ndarray
    swept_mass: np.ndarray
    end_index: int
    magnetic_reynolds: np.ndarray | None = None


class AxialSheathModel:
    """Evolve axial sheath motion using :math:`J\times B` forcing."""

    def __init__(
        self,
        area: float,
        mass: float,
        length: float,
        initial_position: float = 0.0,
        initial_velocity: float = 0.0,
        upstream_density: float = 0.0,
        upstream_pressure: float = 0.0,
    ) -> None:
        self.area = area
        self.mass = mass
        self.length = length
        self.initial_position = initial_position
        self.initial_velocity = initial_velocity
        self.upstream_density = upstream_density
        self.upstream_pressure = upstream_pressure
        self.radius = np.sqrt(area / np.pi)

    def run(
        self, time: Iterable[float], current: Iterable[float], start_index: int = 0
    ) -> SheathResult:
        t = np.array(list(time))
        I = np.array(list(current))
        J = I / self.area
        B = mu0 * I / (2 * np.pi * self.radius)
        jxb = J * B
        net_force = (jxb - self.upstream_pressure) * self.area

        pos = [self.initial_position]
        vel = [self.initial_velocity]
        mass = self.mass
        mass_history = [mass]
        p = self.initial_position
        v = self.initial_velocity
        end_idx = len(t) - 1

        rms: list[float] = []
        for k in range(start_index, len(t) - 1):
            dt = t[k + 1] - t[k]
            F = net_force[k]
            a = F / mass
            v += a * dt
            p += v * dt
            mass += self.upstream_density * self.area * v * dt
            pos.append(p)
            vel.append(v)

            mass_history.append(mass)

            if p >= self.length:
                end_idx = k + 1
                break
        else:
            end_idx = len(t) - 1

        times = t[start_index : end_idx + 1]

        forces = net_force[start_index : end_idx + 1]

        return SheathResult(
            time=times,
            position=np.array(pos),
            velocity=np.array(vel),
            jxb_force=forces,
            swept_mass=np.array(mass_history),
            end_index=end_idx,
        )
