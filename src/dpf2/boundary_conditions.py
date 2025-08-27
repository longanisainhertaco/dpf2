from __future__ import annotations

from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional, Literal
import math

from pydantic import BaseModel, ConfigDict, Field, root_validator

# ---------------------------------------------------------------------------
# Compatibility helpers mirroring pydantic v2 model_validator

def model_validator(*, mode: str = "after"):
    def decorator(func):
        if mode == "after":
            def wrapper(cls, values):
                inst = cls.construct(**values)
                result = func(cls, inst)
                return result.__dict__ if isinstance(result, cls) else values
            return root_validator(pre=False, skip_on_failure=True, allow_reuse=True)(wrapper)
        else:
            def wrapper(cls, values):
                out = func(values)
                return out if out is not None else values
            return root_validator(pre=True, skip_on_failure=True, allow_reuse=True)(wrapper)
    return decorator

if not hasattr(BaseModel, "model_validate"):
    BaseModel.model_validate = classmethod(lambda cls, d, **_: cls(**d))
if not hasattr(BaseModel, "model_dump"):
    BaseModel.model_dump = lambda self, **_: getattr(self, "__dict__", {})
if not hasattr(BaseModel, "model_dump_json"):
    BaseModel.model_dump_json = lambda self, **_: str(getattr(self, "__dict__", {}))
if not hasattr(BaseModel, "model_copy"):
    BaseModel.model_copy = lambda self, **_: self

# Local imports ---------------------------------------------------------------
try:  # pragma: no cover - during early import core_schema may not be ready
    from .core_schema import ConfigSectionBase, to_camel_case
except Exception:  # pragma: no cover - fall back to minimal placeholders
    ConfigSectionBase = object  # type: ignore
    def to_camel_case(name: str) -> str:  # type: ignore
        return name
from dpf2.diagnostics import OutputField

# ---------------------------------------------------------------------------
# Lightweight kinetic sheath model used by Hall-MHD unit tests

class KineticSheath:
    """Evolve ion/electron fluxes at a boundary.

    The model is intentionally minimal.  It computes Bohm-like ion and
    thermal electron fluxes from supplied densities and temperatures and
    returns the sheath thickness together with any impurity flux emitted
    from the surface.  All quantities are in arbitrary units which keeps
    the implementation dependency free yet sufficient for regression
    tests.
    """

    def __init__(
        self,
        *,
        ion_mass: float = 1.0,
        impurity_fraction: float = 0.0,
        max_thickness: float = 1.0,
    ) -> None:
        self.ion_mass = ion_mass
        self.impurity_fraction = impurity_fraction
        self.max_thickness = max_thickness
        self.last_ion_flux = 0.0
        self.last_electron_flux = 0.0
        self.last_impurity_flux = 0.0
        self.last_thickness = 0.0

    def evolve(self, ni: float, ne: float, Te: float, dt: float) -> tuple[float, float, float]:
        """Return ``(thickness, impurity_flux, ion_flux)``.

        ``ni``/``ne`` are ion and electron number densities, ``Te`` is the
        electron temperature and ``dt`` the time step.  The expressions are
        simple analytic estimates and are not meant to be physically
        comprehensive; they merely provide a consistent source for the unit
        tests.
        """

        # Bohm sound speed ~ sqrt(Te/mi)
        v_bohm = math.sqrt(max(Te, 0.0) / max(self.ion_mass, 1e-30))
        ion_flux = ni * v_bohm
        elec_flux = 0.25 * ne * math.sqrt(max(Te, 0.0))
        thickness = min(math.sqrt(max(Te, 0.0) / max(ne, 1e-30)), self.max_thickness)
        impurity_flux = self.impurity_fraction * ion_flux

        self.last_ion_flux = ion_flux
        self.last_electron_flux = elec_flux
        self.last_impurity_flux = impurity_flux
        self.last_thickness = thickness

        return thickness, impurity_flux, ion_flux


class BoundaryTypeEnum(str, Enum):
    """Defines boundary condition types per face and field."""

    REFLECTING = "reflecting"
    CONDUCTING = "conducting"
    ABSORBING = "absorbing"
    PERIODIC = "periodic"
    USER_FUNC = "user_func"


class BoundaryConditions(ConfigSectionBase):
    """Validated schema for domain boundary conditions."""

    config_section_id: ClassVar[Literal["boundary"]] = "boundary"

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        validate_default=True,
    )

    # -- Face types -------------------------------------------------------
    x_low: BoundaryTypeEnum = Field(
        ..., alias="xLow", metadata={"group": "X", "category": "BoundaryConditions"}
    )
    x_high: BoundaryTypeEnum = Field(
        ..., alias="xHigh", metadata={"group": "X"}
    )
    y_low: BoundaryTypeEnum = Field(
        ..., alias="yLow", metadata={"group": "Y"}
    )
    y_high: BoundaryTypeEnum = Field(
        ..., alias="yHigh", metadata={"group": "Y"}
    )
    z_low: BoundaryTypeEnum = Field(
        ..., alias="zLow", metadata={"group": "Z"}
    )
    z_high: BoundaryTypeEnum = Field(
        ..., alias="zHigh", metadata={"group": "Z"}
    )

    _alias_map: ClassVar[Dict[str, str]] = {
        "xLow": "x_low",
        "xHigh": "x_high",
        "yLow": "y_low",
        "yHigh": "y_high",
        "zLow": "z_low",
        "zHigh": "z_high",
    }

    # -- Optional extensions --------------------------------------------
    excluded_faces: Optional[List[Literal[
        "xLow",
        "xHigh",
        "yLow",
        "yHigh",
        "zLow",
        "zHigh",
    ]]] = None

    absorbing_layer_thickness_cells: Optional[int] = Field(
        4,
        ge=0,
        alias="absorbingLayerThicknessCells",
        metadata={"units": "cells"},
    )

    # Perfectly matched layer (PML) configuration
    pml_thickness: Optional[int] = Field(
        2,
        ge=0,
        alias="pmlThickness",
        metadata={"units": "cells"},
    )
    pml_sigma: Optional[float] = Field(
        2.0,
        ge=0.0,
        alias="pmlSigma",
    )
    pml_profile: Optional[Literal["exponential", "linear"]] = Field(
        "exponential",
        alias="pmlProfile",
    )

    ghost_zone_extrapolation: Optional[Literal[
        "constant",
        "linear",
        "parabolic",
    ]] = Field("constant", alias="ghostZoneExtrapolation")

    field_extrapolation_overrides: Optional[
        Dict[str, Literal["constant", "linear", "parabolic"]]
    ] = Field(None, alias="fieldExtrapolationOverrides")

    boundary_field_overrides: Optional[
        Dict[str, Dict[str, BoundaryTypeEnum]]
    ] = Field(None, alias="boundaryFieldOverrides")

    conflict_resolution_policy: Literal[
        "prefer_geometry",
        "prefer_override",
        "error",
    ] = Field("prefer_override", alias="conflictResolutionPolicy")

    # ------------------------------------------------------------------
    @classmethod
    def with_defaults(cls) -> "BoundaryConditions":
        return cls(
            x_low=BoundaryTypeEnum.REFLECTING,
            x_high=BoundaryTypeEnum.REFLECTING,
            y_low=BoundaryTypeEnum.REFLECTING,
            y_high=BoundaryTypeEnum.REFLECTING,
            z_low=BoundaryTypeEnum.REFLECTING,
            z_high=BoundaryTypeEnum.REFLECTING,
            absorbing_layer_thickness_cells=4,
            ghost_zone_extrapolation="constant",
            pml_thickness=2,
            pml_sigma=2.0,
            pml_profile="exponential",
        )

    def resolve_defaults(self) -> "BoundaryConditions":
        data = self.model_dump()
        for f in [
            "x_low",
            "x_high",
            "y_low",
            "y_high",
            "z_low",
            "z_high",
        ]:
            data.setdefault(f, BoundaryTypeEnum.REFLECTING)
        data.setdefault("absorbing_layer_thickness_cells", 4)
        data.setdefault("pml_thickness", 2)
        data.setdefault("pml_sigma", 2.0)
        data.setdefault("pml_profile", "exponential")
        return self.model_validate(data, context=getattr(self, "_context", {}))

    def required_fields(self) -> List[str]:
        return [n for n, f in self.model_fields.items() if f.is_required()]

    def get_field_metadata(self) -> Dict[str, Dict[str, Any]]:
        return {n: field.json_schema_extra or {} for n, field in self.model_fields.items()}

    def normalize_units(self, spatial_units: Literal["cm", "m"]) -> "BoundaryConditions":
        # this model has no dimensional fields to scale
        return self

    def summarize(self) -> str:
        parts = [
            f"X boundaries: {self.x_low.value} | {self.x_high.value}",
            f"Y boundaries: {self.y_low.value} | {self.y_high.value}",
            f"Z boundaries: {self.z_low.value} | {self.z_high.value}",
            f"Absorbing layer: {self.absorbing_layer_thickness_cells} cells, Extrapolation: {self.ghost_zone_extrapolation}",
            f"PML: thickness={self.pml_thickness}, sigma={self.pml_sigma}, profile={self.pml_profile}",
        ]
        if self.boundary_field_overrides:
            lines = []
            for field, mapping in self.boundary_field_overrides.items():
                pairs = [f"{face} = {btype.value}" for face, btype in mapping.items()]
                lines.append(f"  {field}: {', '.join(pairs)}")
            parts.append("Overrides:\n" + "\n".join(lines))
        return "\n".join(parts)

    # ------------------------------------------------------------------
    @classmethod
    def check_rules(cls, values: "BoundaryConditions") -> "BoundaryConditions":
        geometry = getattr(values, "_context", {}).get("geometry")

        valid_faces = {"xLow", "xHigh", "yLow", "yHigh", "zLow", "zHigh"}
        if values.excluded_faces:
            for face in values.excluded_faces:
                field_name = values._alias_map.get(face)
                if field_name and getattr(values, field_name) is not None:
                    raise ValueError(f"{face} must not be set when excluded")

        if geometry == "2D_RZ":
            if values.y_low != BoundaryTypeEnum.REFLECTING or values.y_high != BoundaryTypeEnum.REFLECTING:
                if values.conflict_resolution_policy == "prefer_geometry":
                    values = values.model_copy(update={
                        "y_low": BoundaryTypeEnum.REFLECTING,
                        "y_high": BoundaryTypeEnum.REFLECTING,
                    })
                elif values.conflict_resolution_policy == "error":
                    raise ValueError("2D_RZ geometry requires reflecting Y boundaries")
                # prefer_override leaves as is
            if values.boundary_field_overrides:
                overrides = dict(values.boundary_field_overrides)
                changed = False
                for field, mapping in overrides.items():
                    for face, btype in list(mapping.items()):
                        if face in {"yLow", "yHigh"} and btype != BoundaryTypeEnum.REFLECTING:
                            if values.conflict_resolution_policy == "prefer_geometry":
                                mapping[face] = BoundaryTypeEnum.REFLECTING
                                changed = True
                            elif values.conflict_resolution_policy == "error":
                                raise ValueError("2D_RZ geometry requires reflecting Y overrides")
                if changed:
                    values = values.model_copy(update={"boundary_field_overrides": overrides})

        if values.boundary_field_overrides:
            allowed_fields = {f.value for f in OutputField}
            for field, mapping in values.boundary_field_overrides.items():
                if field not in allowed_fields:
                    raise ValueError(f"unrecognized field name {field}")
                for face in mapping.keys():
                    if face not in valid_faces:
                        raise ValueError(f"invalid face name {face}")

        for low_name, high_name in [("x_low", "x_high"), ("y_low", "y_high"), ("z_low", "z_high")]:
            low = getattr(values, low_name)
            high = getattr(values, high_name)
            if BoundaryTypeEnum.PERIODIC in (low, high) and low != high:
                raise ValueError("periodic faces must be paired")
            if {low, high} == {BoundaryTypeEnum.REFLECTING, BoundaryTypeEnum.ABSORBING}:
                raise ValueError("opposite faces cannot mix reflecting and absorbing")
        return values

    # Custom model_validate to capture context ---------------------------
    @classmethod
    def model_validate(cls, data: Any, **kwargs) -> "BoundaryConditions":
        context = kwargs.get("context") or {}
        obj = super().model_validate(data)
        object.__setattr__(obj, "_context", context)
        obj = cls.check_rules(obj)
        return obj


__all__ = ["BoundaryConditions", "BoundaryTypeEnum", "KineticSheath"]
