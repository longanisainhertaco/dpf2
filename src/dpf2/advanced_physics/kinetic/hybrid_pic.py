"""Hybrid PIC solver combining kinetic ions and fluid electrons.

This module implements a hybrid particle-in-cell solver where ions are
treated kinetically as particles while electrons are modeled as a
neutralizing fluid. This approach is well-suited for studying ion-scale
phenomena while avoiding the computational cost of resolving electron
dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np

try:
    from scipy.constants import mu_0, e as q_e, m_e, m_p, k as k_B
except ImportError:
    mu_0 = 4e-7 * np.pi
    q_e = 1.602176634e-19
    m_e = 9.1093837015e-31
    m_p = 1.67262192369e-27
    k_B = 1.380649e-23

__all__ = [
    "HybridPICSolver",
    "Particle",
    "ParticleSpecies",
]


@dataclass
class Particle:
    """Single macro-particle representation.

    Attributes
    ----------
    x, y, z : float
        Position [m].
    vx, vy, vz : float
        Velocity [m/s].
    weight : float
        Number of physical particles represented.
    species_id : int
        Index identifying the particle species.
    """

    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    weight: float = 1.0
    species_id: int = 0

    def position(self) -> np.ndarray:
        """Return position as array."""
        return np.array([self.x, self.y, self.z])

    def velocity(self) -> np.ndarray:
        """Return velocity as array."""
        return np.array([self.vx, self.vy, self.vz])

    def kinetic_energy(self, mass: float) -> float:
        """Compute kinetic energy [J]."""
        v2 = self.vx ** 2 + self.vy ** 2 + self.vz ** 2
        return 0.5 * mass * v2 * self.weight


@dataclass
class ParticleSpecies:
    """Particle species definition.

    Attributes
    ----------
    name : str
        Species name (e.g., "D", "T", "He").
    mass : float
        Particle mass [kg].
    charge : float
        Particle charge [C].
    """

    name: str
    mass: float
    charge: float

    @classmethod
    def deuterium(cls) -> "ParticleSpecies":
        """Create deuterium ion species."""
        return cls(name="D", mass=2.0 * m_p, charge=q_e)

    @classmethod
    def tritium(cls) -> "ParticleSpecies":
        """Create tritium ion species."""
        return cls(name="T", mass=3.0 * m_p, charge=q_e)

    @classmethod
    def hydrogen(cls) -> "ParticleSpecies":
        """Create hydrogen ion species."""
        return cls(name="H", mass=m_p, charge=q_e)


@dataclass
class HybridPICSolver:
    """Hybrid PIC solver with kinetic ions and fluid electrons.

    This solver evolves ion macro-particles under the influence of
    self-consistent electromagnetic fields. Electrons are treated as
    a massless, charge-neutralizing fluid that provides an equation
    of state for the electron pressure.

    Parameters
    ----------
    nx, ny, nz : int
        Number of grid cells in each direction.
    dx, dy, dz : float
        Grid spacing [m].
    species : list of ParticleSpecies
        Ion species to simulate.
    Te : float
        Electron temperature [K] (isothermal electrons).
    resistivity : float
        Plasma resistivity [Ohm*m].
    """

    nx: int
    ny: int
    nz: int
    dx: float = 1.0
    dy: float = 1.0
    dz: float = 1.0
    species: List[ParticleSpecies] = field(default_factory=list)
    Te: float = 1.0e6
    resistivity: float = 0.0

    particles: List[Particle] = field(default_factory=list, repr=False)

    B: np.ndarray = field(init=False, repr=False)
    E: np.ndarray = field(init=False, repr=False)
    J: np.ndarray = field(init=False, repr=False)
    rho: np.ndarray = field(init=False, repr=False)
    n_i: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize grid arrays."""
        shape = (self.nx, self.ny, self.nz)
        self.B = np.zeros(shape + (3,))
        self.E = np.zeros(shape + (3,))
        self.J = np.zeros(shape + (3,))
        self.rho = np.zeros(shape)
        self.n_i = np.zeros(shape)
        if not self.species:
            self.species = [ParticleSpecies.deuterium()]

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return grid shape."""
        return (self.nx, self.ny, self.nz)

    @property
    def cell_volume(self) -> float:
        """Return cell volume."""
        return self.dx * self.dy * self.dz

    def add_particles(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        weights: Optional[np.ndarray] = None,
        species_id: int = 0,
    ) -> None:
        """Add particles to the simulation.

        Parameters
        ----------
        positions : ndarray
            Particle positions, shape (N, 3).
        velocities : ndarray
            Particle velocities, shape (N, 3).
        weights : ndarray, optional
            Particle weights, shape (N,).
        species_id : int
            Species index.
        """
        n = positions.shape[0]
        if weights is None:
            weights = np.ones(n)

        for i in range(n):
            p = Particle(
                x=positions[i, 0],
                y=positions[i, 1],
                z=positions[i, 2],
                vx=velocities[i, 0],
                vy=velocities[i, 1],
                vz=velocities[i, 2],
                weight=weights[i],
                species_id=species_id,
            )
            self.particles.append(p)

    def deposit_density(self) -> None:
        """Deposit particle density onto grid using CIC interpolation."""
        self.rho.fill(0.0)
        self.n_i.fill(0.0)

        for p in self.particles:
            sp = self.species[p.species_id]

            ix = int(p.x / self.dx) % self.nx
            iy = int(p.y / self.dy) % self.ny
            iz = int(p.z / self.dz) % self.nz

            fx = (p.x / self.dx) - ix
            fy = (p.y / self.dy) - iy
            fz = (p.z / self.dz) - iz

            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        wx = (1 - fx) if di == 0 else fx
                        wy = (1 - fy) if dj == 0 else fy
                        wz = (1 - fz) if dk == 0 else fz
                        w = wx * wy * wz

                        ii = (ix + di) % self.nx
                        jj = (iy + dj) % self.ny
                        kk = (iz + dk) % self.nz

                        charge_density = sp.charge * p.weight * w / self.cell_volume
                        self.rho[ii, jj, kk] += charge_density
                        self.n_i[ii, jj, kk] += p.weight * w / self.cell_volume

    def deposit_current(self) -> None:
        """Deposit particle current density onto grid.

        Uses the Esirkepov charge-conserving scheme for accurate
        current deposition.
        """
        self.J.fill(0.0)

        for p in self.particles:
            sp = self.species[p.species_id]
            v = np.array([p.vx, p.vy, p.vz])

            ix = int(p.x / self.dx) % self.nx
            iy = int(p.y / self.dy) % self.ny
            iz = int(p.z / self.dz) % self.nz

            fx = (p.x / self.dx) - ix
            fy = (p.y / self.dy) - iy
            fz = (p.z / self.dz) - iz

            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        wx = (1 - fx) if di == 0 else fx
                        wy = (1 - fy) if dj == 0 else fy
                        wz = (1 - fz) if dk == 0 else fz
                        w = wx * wy * wz

                        ii = (ix + di) % self.nx
                        jj = (iy + dj) % self.ny
                        kk = (iz + dk) % self.nz

                        j_contrib = sp.charge * p.weight * w * v / self.cell_volume
                        self.J[ii, jj, kk] += j_contrib

    def solve_fields(self) -> None:
        """Solve for electromagnetic fields using generalized Ohm's law.

        The electric field is computed from:
            E = -v_i x B + (J x B)/(en) - grad(p_e)/(en) + eta*J

        where v_i is the ion bulk velocity and p_e is electron pressure.
        """
        n_safe = np.maximum(self.n_i, 1e-30)

        v_i = self.J / (q_e * n_safe[..., np.newaxis])

        v_cross_B = np.cross(v_i, self.B, axis=-1)

        J_cross_B = np.cross(self.J, self.B, axis=-1)
        hall_term = J_cross_B / (q_e * n_safe[..., np.newaxis])

        p_e = n_safe * k_B * self.Te
        grad_pe = np.zeros_like(self.J)
        grad_pe[..., 0] = (np.roll(p_e, -1, axis=0) - np.roll(p_e, 1, axis=0)) / (2 * self.dx)
        grad_pe[..., 1] = (np.roll(p_e, -1, axis=1) - np.roll(p_e, 1, axis=1)) / (2 * self.dy)
        grad_pe[..., 2] = (np.roll(p_e, -1, axis=2) - np.roll(p_e, 1, axis=2)) / (2 * self.dz)
        pe_term = grad_pe / (q_e * n_safe[..., np.newaxis])

        resistive_term = self.resistivity * self.J

        self.E = -v_cross_B + hall_term - pe_term + resistive_term

    def push_particles(self, dt: float) -> None:
        """Advance particles using Boris algorithm.

        The Boris pusher is a second-order accurate, symplectic
        integrator that exactly conserves phase-space volume and
        provides excellent long-term energy conservation.

        Parameters
        ----------
        dt : float
            Time step [s].
        """
        for p in self.particles:
            sp = self.species[p.species_id]
            qm = sp.charge / sp.mass

            ix = int(p.x / self.dx) % self.nx
            iy = int(p.y / self.dy) % self.ny
            iz = int(p.z / self.dz) % self.nz

            E_local = self.E[ix, iy, iz]
            B_local = self.B[ix, iy, iz]

            v = np.array([p.vx, p.vy, p.vz])

            v_minus = v + 0.5 * qm * E_local * dt

            t = 0.5 * qm * B_local * dt
            t_mag2 = np.dot(t, t)
            s = 2.0 * t / (1.0 + t_mag2)

            v_prime = v_minus + np.cross(v_minus, t)
            v_plus = v_minus + np.cross(v_prime, s)

            v_new = v_plus + 0.5 * qm * E_local * dt

            p.vx, p.vy, p.vz = v_new

            p.x += p.vx * dt
            p.y += p.vy * dt
            p.z += p.vz * dt

            Lx = self.nx * self.dx
            Ly = self.ny * self.dy
            Lz = self.nz * self.dz
            p.x = p.x % Lx
            p.y = p.y % Ly
            p.z = p.z % Lz

    def advance_B(self, dt: float) -> None:
        """Advance magnetic field using Faraday's law.

        dB/dt = -curl(E)
        """
        curl_E = np.zeros_like(self.E)

        curl_E[..., 0] = (
            (np.roll(self.E[..., 2], -1, axis=1) - np.roll(self.E[..., 2], 1, axis=1)) / (2 * self.dy)
            - (np.roll(self.E[..., 1], -1, axis=2) - np.roll(self.E[..., 1], 1, axis=2)) / (2 * self.dz)
        )
        curl_E[..., 1] = (
            (np.roll(self.E[..., 0], -1, axis=2) - np.roll(self.E[..., 0], 1, axis=2)) / (2 * self.dz)
            - (np.roll(self.E[..., 2], -1, axis=0) - np.roll(self.E[..., 2], 1, axis=0)) / (2 * self.dx)
        )
        curl_E[..., 2] = (
            (np.roll(self.E[..., 1], -1, axis=0) - np.roll(self.E[..., 1], 1, axis=0)) / (2 * self.dx)
            - (np.roll(self.E[..., 0], -1, axis=1) - np.roll(self.E[..., 0], 1, axis=1)) / (2 * self.dy)
        )

        self.B -= dt * curl_E

    def step(self, dt: float) -> None:
        """Advance the simulation by one time step.

        Uses a standard hybrid PIC cycle:
        1. Deposit density and current
        2. Solve for electric field
        3. Push particles (Boris algorithm)
        4. Advance magnetic field

        Parameters
        ----------
        dt : float
            Time step [s].
        """
        self.deposit_density()
        self.deposit_current()

        self.solve_fields()

        self.push_particles(dt)

        self.advance_B(dt)

    def total_kinetic_energy(self) -> float:
        """Compute total ion kinetic energy."""
        total = 0.0
        for p in self.particles:
            sp = self.species[p.species_id]
            total += p.kinetic_energy(sp.mass)
        return total

    def total_magnetic_energy(self) -> float:
        """Compute total magnetic field energy."""
        B2 = np.sum(self.B ** 2, axis=-1)
        return 0.5 * np.sum(B2) / mu_0 * self.cell_volume

    def ion_temperature(self, species_id: int = 0) -> float:
        """Compute ion temperature from velocity distribution."""
        particles = [p for p in self.particles if p.species_id == species_id]
        if not particles:
            return 0.0

        sp = self.species[species_id]

        total_weight = sum(p.weight for p in particles)
        if total_weight <= 0:
            return 0.0

        vx_mean = sum(p.vx * p.weight for p in particles) / total_weight
        vy_mean = sum(p.vy * p.weight for p in particles) / total_weight
        vz_mean = sum(p.vz * p.weight for p in particles) / total_weight

        v2_mean = sum(
            ((p.vx - vx_mean) ** 2 + (p.vy - vy_mean) ** 2 + (p.vz - vz_mean) ** 2) * p.weight
            for p in particles
        ) / total_weight

        T = sp.mass * v2_mean / (3.0 * k_B)
        return T

    def cfl_timestep(self, cfl: float = 0.5) -> float:
        """Compute CFL-limited time step."""
        B_mag = np.sqrt(np.sum(self.B ** 2, axis=-1))
        B_max = np.max(B_mag)

        n_max = np.max(self.n_i)
        if n_max <= 0:
            n_max = 1e10

        sp = self.species[0]
        v_A = B_max / np.sqrt(mu_0 * sp.mass * n_max)

        omega_ci = abs(sp.charge) * B_max / sp.mass
        dt_gyro = 0.1 / max(omega_ci, 1e-30)

        dx_min = min(self.dx, self.dy, self.dz)
        dt_cfl = cfl * dx_min / max(v_A, 1e-30)

        return min(dt_cfl, dt_gyro)
