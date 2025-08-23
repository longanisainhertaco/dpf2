"""Resistive MHD equations in 2D cylindrical geometry."""
from __future__ import annotations

import numpy as np


class ResistiveMHD:
    """Minimal representation of resistive MHD equations.

    The formulation implemented here follows the ideal MHD equations with an
    additional simple resistive (Ohmic) contribution.  Only the radial and
    axial momenta are evolved which is sufficient for the 2D ``r-z`` geometry
    used throughout the code base.  The azimuthal magnetic field ``B_phi`` is
    retained as it plays an important role in plasma pinches.
    """

    def __init__(self, gamma: float = 5 / 3, eta: float = 0.0) -> None:
        """Create a new resistive MHD system.

        Parameters
        ----------
        gamma:
            Ratio of specific heats.
        eta:
            Scalar resistivity used for Ohmic heating and magnetic field decay.
        """

        self.gamma = gamma
        self.eta = eta
        self.equations = [
            "density",
            "momentum_r",
            "momentum_z",
            "energy",
            "B_r",
            "B_z",
            "B_phi",
        ]

    def conservative_variables(self, primitives: np.ndarray) -> np.ndarray:
        """Convert primitive variables to conservative form.

        The primitive state is expected to be ``[rho, v_r, v_z, p, B_r, B_z,
        B_phi]``.  The returned conservative vector is
        ``[rho, rho*v_r, rho*v_z, E, B_r, B_z, B_phi]`` where ``E`` is the
        total energy density.
        """

        rho, v_r, v_z, p, B_r, B_z, B_phi = primitives
        kinetic = 0.5 * rho * (v_r ** 2 + v_z ** 2)
        magnetic = 0.5 * (B_r ** 2 + B_z ** 2 + B_phi ** 2)
        energy = p / (self.gamma - 1.0) + kinetic + magnetic
        return np.array([rho, rho * v_r, rho * v_z, energy, B_r, B_z, B_phi])

    def _pressure(self, U: np.ndarray) -> float:
        """Return the thermal pressure from conservative variables."""

        rho, m_r, m_z, E, B_r, B_z, B_phi = U
        v_r = m_r / rho
        v_z = m_z / rho
        v2 = v_r ** 2 + v_z ** 2
        B2 = B_r ** 2 + B_z ** 2 + B_phi ** 2
        return (E - 0.5 * rho * v2 - 0.5 * B2) * (self.gamma - 1.0)

    def flux_function(self, U: np.ndarray, direction: str) -> np.ndarray:
        """Compute ideal MHD fluxes in the given ``direction``.

        Parameters
        ----------
        U:
            Conservative state vector.
        direction:
            Either ``"r"`` or ``"z"`` selecting the spatial direction of the
            flux.
        """

        rho, m_r, m_z, E, B_r, B_z, B_phi = U
        v_r = m_r / rho
        v_z = m_z / rho
        B2 = B_r ** 2 + B_z ** 2 + B_phi ** 2
        p = self._pressure(U)
        total_p = p + 0.5 * B2
        Bdotv = B_r * v_r + B_z * v_z

        if direction == "r":
            return np.array(
                [
                    m_r,
                    m_r * v_r + total_p - B_r ** 2,
                    m_z * v_r - B_r * B_z,
                    (E + total_p) * v_r - B_r * Bdotv,
                    0.0,
                    v_z * B_r - v_r * B_z,
                    B_phi * v_r,
                ]
            )

        if direction == "z":
            return np.array(
                [
                    m_z,
                    m_r * v_z - B_r * B_z,
                    m_z * v_z + total_p - B_z ** 2,
                    (E + total_p) * v_z - B_z * Bdotv,
                    v_r * B_z - v_z * B_r,
                    0.0,
                    B_phi * v_z,
                ]
            )

        raise ValueError("direction must be 'r' or 'z'")

    def source_terms(self, U: np.ndarray) -> np.ndarray:
        """Return resistive source terms.

        A simple Ohmic model is employed where magnetic fields decay and the
        lost magnetic energy appears as thermal energy.  Geometric source terms
        are neglected for this minimal implementation.
        """

        _, _, _, _, B_r, B_z, B_phi = U
        B2 = B_r ** 2 + B_z ** 2 + B_phi ** 2
        return np.array(
            [
                0.0,
                0.0,
                0.0,
                self.eta * B2,
                -self.eta * B_r,
                -self.eta * B_z,
                -self.eta * B_phi,
            ]
        )
