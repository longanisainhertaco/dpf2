"""Configuration schema for DPF simulations."""
from __future__ import annotations

import json
from dataclasses import asdict
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
            hints: dict[str, str] = {}
            if isinstance(e, ValidationError):
                for err in e.errors():
                    field = ".".join(map(str, err["loc"]))
                    fields.append(field)
                    hints[field] = err["msg"]
                hint_msg = "; ".join(f"{f}: {m}" for f, m in hints.items())
            else:
                hint_msg = str(e)
            raise ConfigurationError(
                f"Error validating configuration: {hint_msg}", fields=fields, hints=hints
            ) from e

    def to_file(self, path: str | Path) -> Path:
        """Write configuration parameters to a JSON file.

        Parameters
        ----------
        path:
            Destination path for the JSON file.

        Returns
        -------
        Path
            The path where the configuration was written.

        Raises
        ------
        ConfigurationError
            If the configuration cannot be serialized or written to disk.
        """

        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            file_path.write_text(json.dumps(asdict(self), indent=2))
        except Exception as e:  # pragma: no cover - simple error path
            raise ConfigurationError(
                f"Error writing configuration to {file_path}: {e}"
            ) from e
        return file_path
