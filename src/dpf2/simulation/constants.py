# constants.py
"""
Central repository for physical constants.

Note: Many constants are also defined locally in other modules or imported
directly from scipy.constants. This file primarily serves to satisfy specific
imports like the one in circuit.py. Consider standardizing constant usage
across the project (e.g., consistently using scipy.constants).
"""

import scipy.constants as const
import logging

logger = logging.getLogger(__name__)

# Elementary charge (Coulombs)
e = const.e
e_charge = const.e # Alias sometimes used

# Electron mass (kg)
me = const.m_e
m_e = const.m_e # Alias sometimes used

# Vacuum permittivity (F/m)
epsilon0 = const.epsilon_0
epsilon_0 = const.epsilon_0 # Alias sometimes used

# Vacuum permeability (H/m)
mu0 = const.mu_0
mu_0 = const.mu_0 # Alias sometimes used

# Speed of light (m/s)
c = const.c

# Boltzmann constant (J/K)
kB = const.k
k_B = const.k # Alias sometimes used

# Planck constant (J*s)
h_planck = const.h

# Proton mass (kg)
m_p = const.m_p

# Neutron mass (kg)
m_n = const.m_n

# Pi
pi = const.pi

logger.debug("Physical constants loaded from constants.py")

# Example of a derived constant (if needed)
# alpha = e**2 / (4 * pi * epsilon0 * const.hbar * c) # Fine structure constant

# Add other frequently used constants as needed