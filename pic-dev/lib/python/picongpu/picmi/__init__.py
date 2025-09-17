"""
PICMI for PIConGPU
"""

from .simulation import Simulation
from .grid import Cartesian3DGrid
from .solver import ElectromagneticSolver
from .lasers import DispersivePulseLaser, GaussianLaser, PlaneWaveLaser, FromOpenPMDPulseLaser
from .species import Species
from .layout import PseudoRandomLayout, GriddedLayout
from . import constants

from . import diagnostics

from .distribution import (
    FoilDistribution,
    UniformDistribution,
    GaussianDistribution,
    CylindricalDistribution,
    AnalyticDistribution,
)

from .interaction import Interaction
from .interaction.ionization.fieldionization import (
    ADK,
    ADKVariant,
    BSI,
    BSIExtension,
    Keldysh,
)
from .interaction.ionization.electroniccollisionalequilibrium import ThomasFermi

import picmistandard

import sys

assert sys.version_info.major > 3 or sys.version_info.minor >= 10, "Python 3.10 is required for PIConGPU PICMI"

__all__ = [
    "Simulation",
    "Cartesian3DGrid",
    "ElectromagneticSolver",
    "DispersivePulseLaser",
    "FromOpenPMDPulseLaser",
    "GaussianLaser",
    "PlaneWaveLaser",
    "Species",
    "PseudoRandomLayout",
    "GriddedLayout",
    "constants",
    "FoilDistribution",
    "UniformDistribution",
    "GaussianDistribution",
    "AnalyticDistribution",
    "ADK",
    "ADKVariant",
    "BSI",
    "BSIExtension",
    "Keldysh",
    "ThomasFermi",
    "Interaction",
    "diagnostics",
    "CylindricalDistribution",
]


codename = "picongpu"
"""
name of this PICMI implementation
required by PICMI interface
"""

picmistandard.register_constants(constants)
