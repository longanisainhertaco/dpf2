"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari, Julian Lenz
License: GPLv3+
"""

import typeguard

from picongpu.picmi.diagnostics.util import diagnostic_converts_to

from ...pypicongpu.output.macro_particle_count import (
    MacroParticleCount as PyPIConGPUMacroParticleCount,
)
from ...pypicongpu.species.species import Species as PyPIConGPUSpecies
from ..species import Species as PICMISpecies
from .timestepspec import TimeStepSpec


@diagnostic_converts_to(PyPIConGPUMacroParticleCount)
@typeguard.typechecked
class MacroParticleCount:
    """
    Specifies the parameters for counting the total number of macro particles of a given species.

    This plugin counts the total number of macro particles in the simulation,
    useful for tracking particle statistics and population dynamics.

    Parameters
    ----------
    species: string
        Name of the particle species to count (e.g., "electron", "proton").

    period: int
        Number of simulation steps between consecutive counts.
        Unit: steps (simulation time steps).

    name: string, optional
        Optional name for the macro particle count plugin.
    """

    def __init__(self, species: PICMISpecies, period: TimeStepSpec):
        self.species = species
        self.period = period

    def check(self, dict_species_picmi_to_pypicongpu: dict[PICMISpecies, PyPIConGPUSpecies], *args, **kwargs):
        if self.species not in dict_species_picmi_to_pypicongpu.keys():
            raise ValueError(f"Species {self.species} is not known to Simulation")
        pypicongpu_species = dict_species_picmi_to_pypicongpu.get(self.species)
        if pypicongpu_species is None:
            raise ValueError(f"Species {self.species} is not mapped to a PyPIConGPUSpecies.")
