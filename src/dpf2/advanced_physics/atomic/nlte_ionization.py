"""Non-LTE ionization model for collisional-radiative equilibrium.

This module implements a collisional-radiative (CR) model for computing
ionization balance in plasmas that are not in local thermodynamic
equilibrium (LTE). The model solves rate equations for the population
of each ionization state considering:

- Electron impact ionization
- Three-body recombination
- Radiative recombination
- Dielectronic recombination
- Autoionization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.constants import e as q_e, m_e, k as k_B, h as h_planck
    from scipy.integrate import solve_ivp
except ImportError:
    q_e = 1.602176634e-19
    m_e = 9.1093837015e-31
    k_B = 1.380649e-23
    h_planck = 6.62607015e-34
    solve_ivp = None

__all__ = [
    "NLTEIonization",
    "IonizationState",
    "AtomicData",
]

RY_EV = 13.6057


@dataclass
class AtomicData:
    """Atomic data for a single element.

    Attributes
    ----------
    Z : int
        Atomic number.
    symbol : str
        Element symbol.
    ionization_energies : list of float
        Ionization energies for each charge state [eV].
    statistical_weights : list of float
        Statistical weights g for ground states.
    """

    Z: int
    symbol: str
    ionization_energies: List[float] = field(default_factory=list)
    statistical_weights: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize with hydrogenic data if not provided."""
        if not self.ionization_energies:
            self.ionization_energies = [
                RY_EV * (i + 1) ** 2 for i in range(self.Z)
            ]
        if not self.statistical_weights:
            self.statistical_weights = [2.0] * (self.Z + 1)

    @classmethod
    def hydrogen(cls) -> "AtomicData":
        """Create hydrogen atomic data."""
        return cls(Z=1, symbol="H", ionization_energies=[13.6])

    @classmethod
    def deuterium(cls) -> "AtomicData":
        """Create deuterium atomic data."""
        return cls(Z=1, symbol="D", ionization_energies=[13.6])

    @classmethod
    def helium(cls) -> "AtomicData":
        """Create helium atomic data."""
        return cls(Z=2, symbol="He", ionization_energies=[24.59, 54.42])

    @classmethod
    def argon(cls) -> "AtomicData":
        """Create argon atomic data."""
        I_energies = [
            15.76, 27.63, 40.74, 59.81, 75.02, 91.01, 124.32, 143.46,
            422.45, 478.69, 538.96, 618.26, 686.10, 755.74, 854.77,
            918.03, 4120.89, 4426.23
        ]
        return cls(Z=18, symbol="Ar", ionization_energies=I_energies)


@dataclass
class IonizationState:
    """Container for ionization state populations.

    Attributes
    ----------
    populations : ndarray
        Population fractions for each charge state,
        shape (n_cells, Z+1) or (Z+1,) for single point.
    Z_mean : ndarray
        Mean ionization state.
    """

    populations: np.ndarray
    Z_mean: np.ndarray

    @classmethod
    def neutral(cls, Z: int, shape: Tuple = ()) -> "IonizationState":
        """Create fully neutral state."""
        if shape:
            pops = np.zeros(shape + (Z + 1,))
            pops[..., 0] = 1.0
            Z_mean = np.zeros(shape)
        else:
            pops = np.zeros(Z + 1)
            pops[0] = 1.0
            Z_mean = 0.0
        return cls(populations=pops, Z_mean=np.asarray(Z_mean))

    @classmethod
    def fully_ionized(cls, Z: int, shape: Tuple = ()) -> "IonizationState":
        """Create fully ionized state."""
        if shape:
            pops = np.zeros(shape + (Z + 1,))
            pops[..., -1] = 1.0
            Z_mean = np.full(shape, float(Z))
        else:
            pops = np.zeros(Z + 1)
            pops[-1] = 1.0
            Z_mean = float(Z)
        return cls(populations=pops, Z_mean=np.asarray(Z_mean))


@dataclass
class NLTEIonization:
    """Non-LTE ionization model using collisional-radiative equations.

    This model solves for the steady-state or time-dependent ionization
    balance by computing rates for all relevant atomic processes and
    solving the resulting rate equations.

    Parameters
    ----------
    atomic_data : AtomicData
        Atomic properties for the element.
    include_dielectronic : bool
        Include dielectronic recombination.
    include_autoionization : bool
        Include autoionization.
    use_coronal_limit : bool
        Use coronal equilibrium (neglect three-body recombination).
    """

    atomic_data: AtomicData
    include_dielectronic: bool = True
    include_autoionization: bool = True
    use_coronal_limit: bool = False

    def compute_ionization_rates(
        self,
        ne: float,
        Te: float,
    ) -> np.ndarray:
        """Compute electron impact ionization rate coefficients.

        Uses the Lotz formula for collisional ionization:
            S = sum_i a_i * ln(u) / (u * I_i^2) * exp(-I_i/kT)

        Parameters
        ----------
        ne : float
            Electron density [m^-3].
        Te : float
            Electron temperature [K].

        Returns
        -------
        ndarray
            Ionization rate coefficients S_z [m^3/s] for z -> z+1.
        """
        Z = self.atomic_data.Z
        S = np.zeros(Z)

        kT_eV = k_B * Te / q_e

        for z in range(Z):
            I_z = self.atomic_data.ionization_energies[z]
            u = kT_eV / I_z

            if u < 0.01:
                S[z] = 0.0
                continue

            a = 4.5e-14

            S[z] = a * np.log(1.0 + u) / (u * I_z ** 2) * np.exp(-I_z / kT_eV)

        return S

    def compute_recombination_rates(
        self,
        ne: float,
        Te: float,
    ) -> np.ndarray:
        """Compute total recombination rate coefficients.

        Includes:
        - Radiative recombination (alpha_rr)
        - Dielectronic recombination (alpha_dr)
        - Three-body recombination (alpha_3b)

        Parameters
        ----------
        ne : float
            Electron density [m^-3].
        Te : float
            Electron temperature [K].

        Returns
        -------
        ndarray
            Recombination rate coefficients alpha_z [m^3/s] for z -> z-1.
        """
        Z = self.atomic_data.Z
        alpha = np.zeros(Z)

        kT_eV = k_B * Te / q_e
        Te_eV = kT_eV

        for z in range(Z):
            z_eff = z + 1

            alpha_rr = 2.6e-19 * z_eff ** 2 / np.sqrt(Te_eV)

            if self.include_dielectronic and z < Z - 1:
                I_z = self.atomic_data.ionization_energies[z] if z < len(self.atomic_data.ionization_energies) else 0
                E_exc = 0.8 * I_z
                alpha_dr = (
                    1.9e-9 * z_eff ** 0.5 * Te_eV ** (-1.5)
                    * np.exp(-E_exc / Te_eV)
                )
            else:
                alpha_dr = 0.0

            if not self.use_coronal_limit:
                I_z = self.atomic_data.ionization_energies[z]
                lambda_dB = h_planck / np.sqrt(2 * np.pi * m_e * k_B * Te)
                g_ratio = 2.0

                alpha_3b = (
                    g_ratio * lambda_dB ** 3 * ne
                    * self.compute_ionization_rates(ne, Te)[z]
                    * np.exp(I_z * q_e / (k_B * Te))
                ) if z < Z else 0.0
            else:
                alpha_3b = 0.0

            alpha[z] = alpha_rr + alpha_dr + alpha_3b

        return alpha

    def compute_rate_matrix(
        self,
        ne: float,
        Te: float,
    ) -> np.ndarray:
        """Compute the rate matrix for population evolution.

        The rate matrix A satisfies dn/dt = A * n where n is the
        vector of population fractions.

        Parameters
        ----------
        ne : float
            Electron density [m^-3].
        Te : float
            Electron temperature [K].

        Returns
        -------
        ndarray
            Rate matrix, shape (Z+1, Z+1).
        """
        Z = self.atomic_data.Z
        S = self.compute_ionization_rates(ne, Te)
        alpha = self.compute_recombination_rates(ne, Te)

        A = np.zeros((Z + 1, Z + 1))

        for z in range(Z):
            A[z, z] -= ne * S[z]
            A[z + 1, z] += ne * S[z]

        for z in range(1, Z + 1):
            A[z, z] -= ne * alpha[z - 1]
            A[z - 1, z] += ne * alpha[z - 1]

        return A

    def solve_rate_equations(
        self,
        ne: float,
        Te: float,
        n_initial: Optional[np.ndarray] = None,
        t_final: float = 1e-6,
        method: str = "steady",
    ) -> IonizationState:
        """Solve the ionization rate equations.

        Parameters
        ----------
        ne : float
            Electron density [m^-3].
        Te : float
            Electron temperature [K].
        n_initial : ndarray, optional
            Initial population fractions. If None, starts neutral.
        t_final : float
            Final time for time-dependent solution [s].
        method : str
            Solution method: "steady" or "transient".

        Returns
        -------
        IonizationState
            Solution populations and mean charge.
        """
        Z = self.atomic_data.Z

        if n_initial is None:
            n = np.zeros(Z + 1)
            n[0] = 1.0
        else:
            n = np.asarray(n_initial).copy()

        A = self.compute_rate_matrix(ne, Te)

        if method == "steady":
            A_modified = A.copy()
            A_modified[-1, :] = 1.0
            b = np.zeros(Z + 1)
            b[-1] = 1.0

            try:
                n_ss = np.linalg.solve(A_modified, b)
            except np.linalg.LinAlgError:
                n_ss = n

            n_ss = np.maximum(n_ss, 0.0)
            n_ss /= np.sum(n_ss)

            Z_mean = np.sum(np.arange(Z + 1) * n_ss)

            return IonizationState(populations=n_ss, Z_mean=np.asarray(Z_mean))

        else:
            if solve_ivp is None:
                dt = t_final / 1000
                for _ in range(1000):
                    dn = A @ n
                    n = n + dt * dn
                    n = np.maximum(n, 0.0)
                    n /= np.sum(n)
            else:
                def rhs(t, y):
                    return A @ y

                sol = solve_ivp(
                    rhs,
                    (0, t_final),
                    n,
                    method="BDF",
                    dense_output=True,
                )
                n = sol.y[:, -1]
                n = np.maximum(n, 0.0)
                n /= np.sum(n)

            Z_mean = np.sum(np.arange(Z + 1) * n)

            return IonizationState(populations=n, Z_mean=np.asarray(Z_mean))

    def coronal_equilibrium(
        self,
        Te: float,
    ) -> IonizationState:
        """Compute coronal equilibrium ionization balance.

        In the coronal limit (low density), the ionization balance
        depends only on temperature and the ratio S/alpha.

        Parameters
        ----------
        Te : float
            Electron temperature [K].

        Returns
        -------
        IonizationState
            Coronal equilibrium populations.
        """
        ne_dummy = 1e20
        S = self.compute_ionization_rates(ne_dummy, Te)

        old_limit = self.use_coronal_limit
        self.use_coronal_limit = True
        alpha = self.compute_recombination_rates(ne_dummy, Te)
        self.use_coronal_limit = old_limit

        Z = self.atomic_data.Z
        n = np.zeros(Z + 1)
        n[0] = 1.0

        for z in range(Z):
            if alpha[z] > 0:
                n[z + 1] = n[z] * S[z] / alpha[z]
            else:
                n[z + 1] = 0.0

        n /= np.sum(n)
        Z_mean = np.sum(np.arange(Z + 1) * n)

        return IonizationState(populations=n, Z_mean=np.asarray(Z_mean))

    def saha_equilibrium(
        self,
        ne: float,
        Te: float,
    ) -> IonizationState:
        """Compute Saha (LTE) ionization balance.

        In LTE, detailed balance gives the Saha equation:
            n_{z+1}/n_z = (2/n_e) * (g_{z+1}/g_z) * (2*pi*m_e*kT/h^2)^(3/2)
                          * exp(-I_z/kT)

        Parameters
        ----------
        ne : float
            Electron density [m^-3].
        Te : float
            Electron temperature [K].

        Returns
        -------
        IonizationState
            Saha equilibrium populations.
        """
        Z = self.atomic_data.Z
        n = np.zeros(Z + 1)
        n[0] = 1.0

        lambda_dB = h_planck / np.sqrt(2 * np.pi * m_e * k_B * Te)

        for z in range(Z):
            I_z = self.atomic_data.ionization_energies[z]
            g_z = self.atomic_data.statistical_weights[z]
            g_zp1 = self.atomic_data.statistical_weights[z + 1]

            saha_ratio = (
                2.0 / ne * (g_zp1 / g_z) / lambda_dB ** 3
                * np.exp(-I_z * q_e / (k_B * Te))
            )

            n[z + 1] = n[z] * saha_ratio

        n /= np.sum(n)
        Z_mean = np.sum(np.arange(Z + 1) * n)

        return IonizationState(populations=n, Z_mean=np.asarray(Z_mean))

    def ionization_time(
        self,
        ne: float,
        Te: float,
        z: int = 0,
    ) -> float:
        """Compute characteristic ionization time.

        Parameters
        ----------
        ne : float
            Electron density [m^-3].
        Te : float
            Electron temperature [K].
        z : int
            Initial charge state.

        Returns
        -------
        float
            Ionization time [s].
        """
        S = self.compute_ionization_rates(ne, Te)
        if z < len(S) and S[z] > 0:
            return 1.0 / (ne * S[z])
        return np.inf

    def recombination_time(
        self,
        ne: float,
        Te: float,
        z: int = 1,
    ) -> float:
        """Compute characteristic recombination time.

        Parameters
        ----------
        ne : float
            Electron density [m^-3].
        Te : float
            Electron temperature [K].
        z : int
            Initial charge state.

        Returns
        -------
        float
            Recombination time [s].
        """
        alpha = self.compute_recombination_rates(ne, Te)
        if z > 0 and z <= len(alpha) and alpha[z - 1] > 0:
            return 1.0 / (ne * alpha[z - 1])
        return np.inf

    def radiative_loss_rate(
        self,
        ne: float,
        Te: float,
        ni: float,
    ) -> float:
        """Compute radiative power loss from ionization/recombination.

        Parameters
        ----------
        ne : float
            Electron density [m^-3].
        Te : float
            Electron temperature [K].
        ni : float
            Ion density [m^-3].

        Returns
        -------
        float
            Radiative power loss [W/m^3].
        """
        state = self.solve_rate_equations(ne, Te)
        Z = self.atomic_data.Z

        power_loss = 0.0

        for z in range(Z):
            n_z = ni * state.populations[z]
            I_z = self.atomic_data.ionization_energies[z]

            alpha = self.compute_recombination_rates(ne, Te)
            power_loss += n_z * ne * alpha[z] * I_z * q_e

        return power_loss
