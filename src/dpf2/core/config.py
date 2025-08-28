"""Configuration schema for DPF simulations."""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError
from pydantic.dataclasses import dataclass as pydantic_dataclass

from ..exceptions import ConfigurationError


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

    @classmethod
    def from_file(cls, path: str | Path) -> "DPFConfig":
        """Load configuration parameters from a JSON file.

        Parameters
        ----------
        path:
            Path to a JSON file containing configuration data.

        Returns
        -------
        DPFConfig
            A new configuration instance populated with values from the file.

        Raises
        ------
        ConfigurationError
            If the file is missing, contains invalid JSON, or fails validation.
        """

        file_path = Path(path)
        try:
            raw = file_path.read_text()
        except FileNotFoundError as e:  # pragma: no cover - simple error path
            raise ConfigurationError(f"Configuration file not found: {file_path}") from e

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Error decoding JSON from {file_path}: {e}") from e

        if not isinstance(data, dict):
            raise ConfigurationError(
                f"Configuration file {file_path} did not contain a JSON object"
            )

        try:
            return cls(**data)
        except (TypeError, ValidationError) as e:
            fields: list[str] = []
            if isinstance(e, ValidationError):
                fields = [".".join(map(str, err["loc"])) for err in e.errors()]
            raise ConfigurationError(
                f"Error validating configuration: {e}", fields=fields
            ) from e
