"""Configuration schema for DPF simulations."""
from __future__ import annotations

import json
from dataclasses import asdict, replace, field
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import numpy as np
from pydantic import ValidationError
from pydantic.dataclasses import dataclass as pydantic_dataclass

from ..exceptions import ConfigurationError


@pydantic_dataclass
class JitterDistribution:
    """Description of a stochastic perturbation on a scalar parameter."""

    distribution: Literal["normal", "uniform"] = "normal"
    std: float = 0.0

    def sample(self, rng: np.random.Generator, nominal: float, *, relative: bool = True) -> float:
        """Return a jittered value around ``nominal``."""
        import random as _random

        sigma = self.std * (nominal if relative else 1.0)
        if sigma == 0.0:
            return float(nominal)
        if self.distribution == "uniform":
            if hasattr(rng, "uniform"):
                return float(rng.uniform(nominal - sigma, nominal + sigma))
            return float(_random.uniform(nominal - sigma, nominal + sigma))
        if hasattr(rng, "normal"):
            return float(rng.normal(nominal, sigma))
        return float(_random.gauss(nominal, sigma))


@pydantic_dataclass
class JitterConfig:
    """Collections of jitter distributions for common parameters."""

    voltage: JitterDistribution = field(default_factory=JitterDistribution)
    pressure: JitterDistribution = field(default_factory=JitterDistribution)
    switch_timing: JitterDistribution = field(default_factory=JitterDistribution)


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
    geometry: Dict[str, Any] | None = None

    jitter: JitterConfig = field(default_factory=JitterConfig)
    datasets: Dict[str, Dict[str, Dict[str, object]]] | None = None


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

        # Pydantic's dataclass wrapper does not recursively coerce nested
        # dataclasses, so construct the jitter configuration explicitly when
        # loading from a dictionary.
        if "jitter" in data and isinstance(data["jitter"], dict):
            j = data["jitter"]
            data["jitter"] = JitterConfig(
                voltage=JitterDistribution(**j.get("voltage", {})),
                pressure=JitterDistribution(**j.get("pressure", {})),
                switch_timing=JitterDistribution(**j.get("switch_timing", {})),
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

    # ------------------------------------------------------------------
    def apply_jitter(
        self, rng: np.random.Generator | None = None
    ) -> Tuple["DPFConfig", Dict[str, float]]:
        """Return a new config with jittered parameters.

        The returned dictionary contains the sampled values for each jittered
        field (``voltage``, ``pressure`` and ``switch_timing``) so callers can
        record them in manifests.
        """

        rng = rng or np.random.default_rng()
        cfg = replace(self)

        jittered: Dict[str, float] = {}

        cfg.charging_voltage = self.jitter.voltage.sample(
            rng, self.charging_voltage
        )
        jittered["voltage"] = cfg.charging_voltage

        cfg.initial_pressure = self.jitter.pressure.sample(
            rng, self.initial_pressure
        )
        jittered["pressure"] = cfg.initial_pressure

        jittered["switch_timing"] = self.jitter.switch_timing.sample(
            rng, 0.0, relative=False
        ) * 1e-9
        return cfg, jittered
