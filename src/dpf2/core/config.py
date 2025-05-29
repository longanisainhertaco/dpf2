"""Configuration schema for DPF simulations."""
from __future__ import annotations
from pydantic.dataclasses import dataclass as pydantic_dataclass


@pydantic_dataclass
class DPFConfig:
    """Simulation configuration parameters."""

    cathode_radius: float = 0.015
    anode_radius: float = 0.025
    electrode_length: float = 0.10
    capacitance: float = 30e-6
    inductance: float = 20e-9
    resistance: float = 0.01
    charging_voltage: float = 15000.0
    gas_type: str = "deuterium"
    initial_pressure: float = 133.3
    nr_cells: int = 100
    nz_cells: int = 200
    cfl_number: float = 0.5
    end_time: float = 10e-6
