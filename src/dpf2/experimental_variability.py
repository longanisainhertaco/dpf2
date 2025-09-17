from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Literal

from pydantic import BaseModel, ConfigDict, Field
from .utils.pydantic_compat import model_validator



# Local imports ---------------------------------------------------------------
from .core_schema import ConfigSectionBase, UnitsSystem, UNIT_SCALE_MAP, to_camel_case

logger = logging.getLogger(__name__)


class ExperimentalVariabilityModel(ConfigSectionBase):
    """Configuration of stochastic and environmental variability."""

    config_section_id: ClassVar[Literal["variability"]] = "variability"

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        validate_default=True,
    )

    # -- Global toggles ----------------------------------------------------
    disable_all_variability: bool = Field(
        False, description="Disables all variability behavior across the model"
    )
    variability_application_mode: Literal["single", "per_segment", "ensemble_only"] = "single"

    # -- Stochastic fields -------------------------------------------------
    pressure_jitter_pct: float = Field(
        0.0, metadata={"units": "%", "group": "Noise"}
    )
    trigger_jitter_ns: float = Field(
        0.0, metadata={"units": "ns", "group": "Noise"}
    )
    erosion_multiplier: float = Field(
        1.0, metadata={"units": "scale", "group": "Aging"}
    )
    stochastic_run_id: Optional[int] = Field(
        None, description="Seed for RNG-based reproducibility"
    )

    # -- Distribution settings --------------------------------------------
    distribution_model: Literal["uniform", "normal", "lognormal"] = "normal"
    distribution_params: Optional[Dict[str, float]] = None
    per_field_distributions: Optional[
        Dict[str, Literal["uniform", "normal", "lognormal"]]
    ] = None
    per_field_distribution_params: Optional[Dict[str, Dict[str, float]]] = None
    jitter_correlation_matrix: Optional[List[List[float]]] = None

    # -- Profile overrides -------------------------------------------------
    erosion_profile_from_file: Optional[Path] = None
    erosion_model_type: Literal["scalar", "energy_weighted"] = "scalar"

    time_varying_environment_model: Literal[
        "none", "custom_script", "from_file"
    ] = "none"
    time_varying_profile_path: Optional[Path] = None
    time_varying_profile_schema: Literal["flat", "multi-var", "tabular"] = "flat"
    expected_profile_columns: Optional[List[str]] = Field(
        default_factory=lambda: ["t_ns", "pressure", "temperature"]
    )
    time_column_unit: Literal["ns", "us", "ms"] = "ns"
    profile_conflict_policy: Literal[
        "prefer_file", "prefer_scalar", "error"
    ] = "prefer_file"

    # -- ML / ensemble flags ----------------------------------------------
    allow_ensemble_variation: bool = True
    apply_to_parameter_sweep: Optional[bool] = True
    tagged_fields: Dict[str, List[str]] = Field(default_factory=dict)

    # -- Fingerprinting ----------------------------------------------------
    variability_config_hash: Optional[str] = None

    # ------------------------------------------------------------------
    @classmethod
    def with_defaults(cls) -> "ExperimentalVariabilityModel":
        return cls()

    def resolve_defaults(self) -> "ExperimentalVariabilityModel":
        data = self.model_dump()
        return self.model_validate(data)

    def required_fields(self) -> List[str]:
        return [n for n, f in self.model_fields.items() if f.is_required()]

    def get_field_metadata(self) -> Dict[str, Dict[str, Any]]:
        return {n: field.json_schema_extra or {} for n, field in self.model_fields.items()}

    def normalize_units(self, base_units: UnitsSystem) -> "ExperimentalVariabilityModel":
        scale = UNIT_SCALE_MAP.get(base_units, 1.0)
        trig = self.trigger_jitter_ns * scale
        return self.model_copy(update={"trigger_jitter_ns": trig})

    def summarize(self) -> str:
        dist = self.per_field_distributions.get("pressure_jitter_pct", self.distribution_model) if self.per_field_distributions else self.distribution_model
        erosion_desc = f"{self.erosion_multiplier}x"
        if self.erosion_profile_from_file:
            erosion_desc += f" ({self.erosion_model_type} override: {self.erosion_profile_from_file})"
        env_desc = "disabled"
        if self.time_varying_environment_model != "none":
            path = self.time_varying_profile_path.name if self.time_varying_profile_path else "n/a"
            env_desc = f"enabled (env profile: {path}, schema = {self.time_varying_profile_schema})"
        ml_count = len(self.tagged_fields.get("ml", []))
        return (
            f"Stochastic: pressure = {self.pressure_jitter_pct}% ({dist}), trigger = {self.trigger_jitter_ns} ns\n"
            f"Erosion: {erosion_desc}\n"
            f"Time-varying: {env_desc}\n"
            f"Mode: {self.variability_application_mode}, ML-tagged fields: {ml_count}"
        )

    # Helper ---------------------------------------------------------------
    def hash_variability(self) -> str:
        data = self.model_dump(exclude={"variability_config_hash"}, by_alias=True)
        for key in ("erosion_profile_from_file", "time_varying_profile_path"):
            path = data.get(key)
            if path is not None:
                p = Path(path)
                try:
                    if p.exists():
                        data[key] = p.read_bytes().decode(errors="ignore")
                except Exception as e:
                    logger.warning("Failed to read %s: %s", p, e)
                    data[key] = str(path)
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def check_rules(cls, values: "ExperimentalVariabilityModel") -> "ExperimentalVariabilityModel":
        if values.disable_all_variability:
            values = values.model_copy(update={
                "pressure_jitter_pct": 0.0,
                "trigger_jitter_ns": 0.0,
                "erosion_multiplier": 1.0,
            })

        if not 0 <= values.pressure_jitter_pct <= 100:
            raise ValueError("pressure_jitter_pct must be between 0 and 100")
        if values.trigger_jitter_ns < 0:
            raise ValueError("trigger_jitter_ns must be >= 0")
        if not 0.0 <= values.erosion_multiplier <= 10.0:
            raise ValueError("erosion_multiplier must be between 0.0 and 10.0")

        if values.erosion_profile_from_file is not None:
            p = Path(values.erosion_profile_from_file)
            if not (p.exists() or str(p).startswith("file:")):
                raise ValueError("erosion_profile_from_file must exist or be URI-valid")

        if values.time_varying_environment_model == "from_file":
            if values.time_varying_profile_path is None:
                raise ValueError("time_varying_profile_path is required when environment model is 'from_file'")
            p = Path(values.time_varying_profile_path)
            if not (p.exists() or str(p).startswith("file:")):
                raise ValueError("time_varying_profile_path must exist or be URI-valid")

        if values.erosion_profile_from_file and values.erosion_multiplier != 1.0:
            if values.profile_conflict_policy == "prefer_file":
                values = values.model_copy(update={"erosion_multiplier": 1.0})
            elif values.profile_conflict_policy == "prefer_scalar":
                values = values.model_copy(update={"erosion_profile_from_file": None})
            else:
                raise ValueError("Conflict between scalar and profile erosion settings")

        if values.jitter_correlation_matrix is not None:
            mat = values.jitter_correlation_matrix
            n = len(mat)
            for row in mat:
                if len(row) != n:
                    raise ValueError("jitter_correlation_matrix must be square")
            for i in range(n):
                for j in range(i, n):
                    if mat[i][j] != mat[j][i]:
                        raise ValueError("jitter_correlation_matrix must be symmetric")
            if n != 3:
                raise ValueError("jitter_correlation_matrix size must match stochastic field count")

        if values.tagged_fields:
            valid_fields = set(values.model_fields.keys())
            for tag, fields in values.tagged_fields.items():
                for f in fields:
                    if f not in valid_fields:
                        raise ValueError(f"tagged field {f} not found")

        # Compute hash --------------------------------------------------
        values = values.model_copy(update={"variability_config_hash": values.hash_variability()})
        return values


import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from .dpf_config import DPFConfig


class MonteCarloVariability:
    """Utility for applying Monte-Carlo variations to a :class:`DPFConfig`.

    The sampler draws perturbations for capacitor voltage, fill pressure and
    selected geometric quantities using the distribution information stored in
    :class:`ExperimentalVariabilityModel`.  The resulting configurations can be
    fed to :class:`~dpf2.simulation_engine.SimulationEngine` to execute an
    ensemble of runs.
    """

    def __init__(self, model: ExperimentalVariabilityModel, realizations: int) -> None:
        self.model = model
        self.realizations = realizations
        self.rng = np.random.default_rng(model.stochastic_run_id)

    # ------------------------------------------------------------------
    def _distribution(self, field: str) -> tuple[str, float]:
        """Return distribution type and relative jitter for ``field``."""

        dist = (
            self.model.per_field_distributions.get(field, self.model.distribution_model)
            if self.model.per_field_distributions
            else self.model.distribution_model
        )
        params = self.model.distribution_params or {}
        if self.model.per_field_distribution_params and field in self.model.per_field_distribution_params:
            params = self.model.per_field_distribution_params[field]

        if field == "fill_pressure":
            pct = self.model.pressure_jitter_pct / 100.0
        else:
            pct = params.get("jitter_pct", params.get("std_pct", 0.0)) / 100.0
        return dist, pct

    # ------------------------------------------------------------------
    def _sample_scalar(self, field: str, nominal: float) -> np.ndarray:
        dist, pct = self._distribution(field)
        if pct == 0.0:
            return np.full(self.realizations, nominal)
        if dist == "uniform":
            span = nominal * pct
            return self.rng.uniform(nominal - span, nominal + span, self.realizations)
        if dist == "normal":
            sigma = nominal * pct
            return self.rng.normal(nominal, sigma, self.realizations)
        if dist == "lognormal":
            sigma = pct
            mu = np.log(nominal) - 0.5 * sigma**2
            return self.rng.lognormal(mean=mu, sigma=sigma, size=self.realizations)
        raise ValueError(f"Unknown distribution: {dist}")

    # Public API -------------------------------------------------------
    def sample_capacitor_voltage(self, nominal: float) -> np.ndarray:
        return self._sample_scalar("capacitor_voltage", nominal)

    def sample_fill_pressure(self, nominal: float) -> np.ndarray:
        return self._sample_scalar("fill_pressure", nominal)

    def sample_geometry_tolerances(self, geometry: Dict[str, float]) -> List[Dict[str, float]]:
        samples: List[Dict[str, float]] = []
        for _ in range(self.realizations):
            g_sample: Dict[str, float] = {}
            for name, value in geometry.items():
                g_sample[name] = self._sample_scalar(name, value)[0]
            samples.append(g_sample)
        return samples

    def generate_configurations(self, base_config: "DPFConfig") -> List["DPFConfig"]:
        from .dpf_config import DPFConfig  # Local import to avoid circular ref

        voltage_samples = self.sample_capacitor_voltage(base_config.circuit_config.V0)
        pressure_samples = self.sample_fill_pressure(
            base_config.initial_conditions.paschen_model.gas_pressure_torr
        )
        geom_base = getattr(base_config, "electrode_geometry", None)
        if geom_base is not None:
            geom_samples = self.sample_geometry_tolerances(
                {"cathode_gap_degrees": geom_base.cathode_gap_degrees}
            )
        else:
            geom_samples = [{} for _ in range(self.realizations)]

        configs: List[DPFConfig] = []
        for i in range(self.realizations):
            cfg = base_config.model_copy(deep=True)
            cfg = cfg.model_copy(
                update={
                    "circuit_config": cfg.circuit_config.model_copy(update={"V0": voltage_samples[i]}),
                    "initial_conditions": cfg.initial_conditions.model_copy(
                        update={
                            "paschen_model": cfg.initial_conditions.paschen_model.model_copy(
                                update={"gas_pressure_torr": pressure_samples[i]}
                            )
                        }
                    ),
                    **(
                        {
                            "electrode_geometry": cfg.electrode_geometry.model_copy(
                                update=geom_samples[i]
                            )
                        }
                        if geom_samples[i] and getattr(cfg, "electrode_geometry", None) is not None
                        else {}
                    ),
                }
            )
            configs.append(cfg)
        return configs


__all__ = ["ExperimentalVariabilityModel", "MonteCarloVariability"]

