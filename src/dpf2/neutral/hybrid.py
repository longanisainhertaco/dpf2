from __future__ import annotations

"""Hybrid neutral gas module coupling DSMC and fluid estimates to sheath motion."""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dpf2.neutral.dsmc import DSMC, load_lxcat_table
from dpf2.physics.neutral_gas import NeutralGasFluid
from dpf2.physics.neutral_gas.swarm import (
    SwarmParameters,
    compute_swarm_parameters,
    validate_swarm_parameters,
)
from dpf2.axial_sheath import AxialSheathModel


@dataclass
class HybridNeutralModel:
    """Couple a DSMC solver, fluid surrogate and sheath entry conditions."""

    dsmc: DSMC
    fluid: NeutralGasFluid
    swarm: SwarmParameters
    table_path: Path

    @classmethod
    def from_lxcat(
        cls,
        table: Path | str,
        *,
        knudsen_number: float = 1.0,
        volume: float = 1.0,
        puff_start: float = 0.0,
        puff_end: float = 0.0,
        puff_rate: float = 0.0,
    ) -> "HybridNeutralModel":
        """Build a hybrid model validated against an LXCat style table."""

        path = Path(table)
        table_data = load_lxcat_table(path)
        swarm = compute_swarm_parameters(table_data)
        dsmc = DSMC(
            table_data,
            knudsen_number=knudsen_number,
            puff_start=puff_start,
            puff_end=puff_end,
            puff_rate=puff_rate,
        )
        fluid = NeutralGasFluid(
            rho=0.0,
            volume=volume,
            mass_flow_rate=puff_rate,
            puff_start=puff_start,
            puff_end=puff_end,
        )
        return cls(dsmc=dsmc, fluid=fluid, swarm=swarm, table_path=path)

    def step(self, dt: float, *, t: float = 0.0, plasma_density: float = 0.0) -> float:
        """Advance DSMC and fluid estimates and return a blended density."""

        effective_ionization = min(self.swarm.mobility, 1e-6)
        neutral_density = self.dsmc.run(
            dt,
            t=t,
            plasma_density=plasma_density,
            ionization_rate=effective_ionization,
        )
        fluid_density = self.fluid.step(dt, t, ionization_rate=effective_ionization)
        # Blend the kinetic and fluid estimates; DSMC density is treated as number
        # density, while the fluid surrogate acts as a conditioning buffer.
        blended = 0.5 * (neutral_density + fluid_density)
        self.dsmc.density = max(0.0, blended)
        return blended

    def couple_sheath(
        self, sheath: AxialSheathModel, dt: float, *, t: float = 0.0, plasma_density: float = 0.0
    ) -> float:
        """Update a sheath model's upstream density using the hybrid neutral state."""

        density = self.step(dt, t=t, plasma_density=plasma_density)
        sheath.upstream_density = density
        return density

    def validate_swarm(self, reference: Mapping[str, float]) -> SwarmParameters:
        """Validate swarm parameters against a reference map."""

        return validate_swarm_parameters(self.table_path, dict(reference))


__all__ = ["HybridNeutralModel"]
