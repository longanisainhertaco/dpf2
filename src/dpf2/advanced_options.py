from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Literal

from pydantic import BaseModel, ConfigDict, Field
from .utils.pydantic_compat import model_validator

# ---------------------------------------------------------------------------


# Local imports ---------------------------------------------------------------
from .core_schema import ConfigSectionBase, to_camel_case


class AdvancedOptions(ConfigSectionBase):
    """Developer and experimental toggles for DPF simulations."""

    config_section_id: ClassVar[Literal["advanced"]] = "advanced"

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        validate_default=True,
    )

    # ------------------------------------------------------------------
    # Core toggles
    simulate_breakdown_without_plasma: bool = Field(
        False,
        description="Force breakdown physics without plasma present",
        metadata={"group": "Debug", "reloadable": False},
    )

    enable_diagnostics_mock_mode: bool = Field(
        False,
        description="Stub diagnostics output for UI testing",
        metadata={"group": "Diagnostics", "reloadable": True},
    )

    disable_all_validators: bool = Field(
        False,
        description="Globally disable schema validation — dangerous",
        metadata={"group": "Danger", "reloadable": False},
    )

    disable_reason: Optional[str] = Field(
        None, description="Required if disable_all_validators is True"
    )

    use_legacy_energy_tracker: bool = Field(
        False, description="Force use of deprecated energy diagnostics"
    )

    export_diagnostics_stub: bool = Field(
        False, description="Write placeholder diagnostics for scripting or CI"
    )

    override_initial_conditions_check: bool = Field(
        False, description="Skip EOS compatibility checks for initial conditions"
    )

    # ------------------------------------------------------------------
    # Backend controls
    force_amrex_tile_size: Optional[Tuple[int, int, int]] = Field(
        None, metadata={"units": "cells", "group": "AMReX"}
    )

    inject_runtime_script: Optional[Path] = Field(
        None,
        description="Path to script injected at runtime",
        metadata={"group": "Dev"},
    )

    amrex_debug_level: Optional[int] = Field(
        None, ge=0, le=5, metadata={"group": "AMReX", "reloadable": True}
    )

    # ------------------------------------------------------------------
    # Metadata + hashing
    advanced_scope: Literal["global", "diagnostics_only", "io", "debug"] = "debug"
    developer_notes: Optional[str] = Field(
        None, description="Optional explanation for dev settings"
    )
    reloadable_flags: Optional[List[str]] = Field(default_factory=list)
    advanced_config_hash: Optional[str] = None

    # ------------------------------------------------------------------
    @classmethod
    def with_defaults(cls) -> "AdvancedOptions":
        return cls()

    def resolve_defaults(self) -> "AdvancedOptions":
        data = self.model_dump()
        return self.model_validate(data)

    @classmethod
    def required_fields(cls) -> List[str]:
        return [n for n, f in cls.model_fields.items() if f.is_required()]

    @classmethod
    def get_field_metadata(cls) -> Dict[str, Dict[str, Any]]:
        return {
            name: (field.json_schema_extra or field.metadata or {})
            for name, field in cls.model_fields.items()
        }

    def summarize(self) -> str:
        tile = (
            f"{list(self.force_amrex_tile_size)}"
            if self.force_amrex_tile_size is not None
            else "None"
        )
        script = (
            self.inject_runtime_script.name if self.inject_runtime_script else "None"
        )
        hash_short = (
            self.advanced_config_hash[:6] if self.advanced_config_hash else "none"
        )
        return (
            f"Advanced Mode: diagnostics mocked = {self.enable_diagnostics_mock_mode}, "
            f"validators disabled = {self.disable_all_validators}\n"
            f"Tile size override: {tile}, Runtime script: {script}\n"
            f"Debug level: {self.amrex_debug_level}, Scope: {self.advanced_scope}, "
            f"Hash: {hash_short}"
        )

    # Helper ---------------------------------------------------------------
    def hash_advanced_options(self) -> str:
        data = self.model_dump(exclude={"advanced_config_hash"}, by_alias=True)
        path_key = data.get("injectRuntimeScript")
        if path_key is not None:
            p = Path(path_key)
            try:
                if p.exists():
                    data["injectRuntimeScript"] = p.read_bytes().decode(errors="ignore")
            except Exception as e:
                warnings.warn(f"Failed to read {p}: {e}", RuntimeWarning)
                data["injectRuntimeScript"] = str(path_key)
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def export_diagnostics_stub_files(self, output_dir: Path) -> List[Path]:
        """Write deterministic stub diagnostics for tooling or CI.

        Parameters
        ----------
        output_dir:
            Directory where stub files will be written. Created if necessary.

        Returns
        -------
        List[Path]
            Paths to the generated files, or an empty list if stubs are disabled.
        """
        if not self.export_diagnostics_stub:
            return []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_path = output_dir / "summary.json"
        waveform_path = output_dir / "current.csv"

        summary_data = {
            "neutron_yield": 0.0,
            "max_b_field_T": 0.0,
            "pinch_time_ns": 0.0,
        }
        summary_path.write_text(json.dumps(summary_data, indent=2, sort_keys=True))
        waveform_path.write_text("time_ns,current_kA\n0,0\n")
        return [summary_path, waveform_path]

    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def check_rules(cls, values: "AdvancedOptions") -> "AdvancedOptions":
        if values.disable_all_validators and not values.disable_reason:
            raise ValueError("disable_reason must be set when validators are disabled")
        if values.force_amrex_tile_size is not None:
            if len(values.force_amrex_tile_size) != 3 or any(
                d <= 0 for d in values.force_amrex_tile_size
            ):
                raise ValueError(
                    "force_amrex_tile_size must be three positive integers"
                )
        if values.inject_runtime_script is not None:
            p = Path(values.inject_runtime_script)
            if not p.exists():
                raise ValueError("inject_runtime_script path does not exist")
        if values.advanced_scope == "global" and (
            values.disable_all_validators or values.override_initial_conditions_check
        ):
            warnings.warn("Advanced options may alter global campaign behavior")
        values = values.model_copy(
            update={"advanced_config_hash": values.hash_advanced_options()}
        )
        return values


__all__ = ["AdvancedOptions"]
