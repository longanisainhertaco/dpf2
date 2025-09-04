from __future__ import annotations

import hashlib
import json
from pathlib import Path
from bisect import bisect_right
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Literal, Callable, Sequence, Protocol

from pydantic import BaseModel, ConfigDict, Field
from .utils.pydantic_compat import model_validator as _model_validator


# ---------------------------------------------------------------------------


from .core_schema import ConfigSectionBase, to_camel_case


def from_camel_case(string: str) -> str:
    out = []
    for ch in string:
        if ch.isupper():
            out.append("_")
            out.append(ch.lower())
        else:
            out.append(ch)
    return "".join(out)
from .units_settings import UnitsSettings


class IonBeamEDF(Protocol):
    """Protocol providing ion energy distributions by angle."""

    def energy_distribution(self, angle_deg: float) -> Tuple[Sequence[float], Sequence[float]]:
        ...


class NeutronYieldModel(ConfigSectionBase):
    """Configuration for neutron yield modeling in DPF simulations."""

    config_section_id: ClassVar[Literal["neutron_yield"]] = "neutron_yield"

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        validate_default=True,
    )

    # ------------------------------------------------------------------
    # Core fusion toggles
    fusion_fuel_type: Literal["DD", "DT"] = "DD"
    beam_target_model_enabled: bool = True
    thermonuclear_model_enabled: bool = True
    separate_yield_components: bool = True
    yield_integration_window_us: Optional[Tuple[float, float]] = None

    # ------------------------------------------------------------------
    # Beam-target fusion configuration
    beam_ion_species: str
    target_density_source: Literal["diagnostics", "constant", "user_file"] = "diagnostics"
    target_density_constant: Optional[float] = Field(None, metadata={"units": "cm^-3"})
    target_density_diagnostic_path: Optional[Path] = None
    iedf_source: Literal["diagnostics", "user_file", "synthetic_gaussian"] = "diagnostics"
    iedf_user_path: Optional[Path] = None
    iedf_diagnostic_path: Optional[Path] = None
    iedf_format: Optional[Literal["csv", "OpenPMD", "json"]] = "csv"
    fusion_cross_section_model: Literal["Bosch-Hale", "EXFOR", "tabulated"] = "Bosch-Hale"
    cross_section_table_path: Optional[Path] = None
    cross_section_table_units: Optional[Dict[str, str]] = {"energy": "MeV", "sigma": "barn"}

    # ------------------------------------------------------------------
    # Thermonuclear fusion configuration
    reactivity_source: Literal["look-up", "analytic", "FLYCHK"] = "look-up"
    maxwellian_assumed: bool = True
    average_ion_temperature_keV: Optional[float] = None
    average_ion_density_cm3: Optional[float] = None
    dd_branching_ratio: Optional[float] = Field(0.5, ge=0.0, le=1.0)
    reactivity_table_path: Optional[Path] = None
    reactivity_table_units: Optional[Dict[str, str]] = {"Ti": "keV", "reactivity": "cm^3/s"}

    # ------------------------------------------------------------------
    # Spectrum output and detector modeling
    neutron_spectrum_output_enabled: bool = True
    spectrum_energy_bins_MeV: Optional[List[float]] = None
    anisotropic_spectrum: bool = False
    spectrum_output_format: Optional[Literal["csv", "OpenPMD", "plot", "hdf5"]] = "csv"
    apply_detector_response_function: bool = False
    detector_response_file: Optional[Path] = None
    detector_response_normalization: Optional[Literal["none", "area", "peak", "custom"]] = "none"

    # ------------------------------------------------------------------
    @classmethod
    def with_defaults(cls) -> "NeutronYieldModel":
        return cls(
            beam_ion_species="D+",
            reactivity_table_path=Path("reactivity.dat"),
        )

    def resolve_defaults(self) -> "NeutronYieldModel":
        data = self.model_dump()
        return self.model_validate(data)

    def required_fields(self) -> List[str]:
        return [n for n, f in self.model_fields.items() if f.is_required()]

    def get_field_metadata(self) -> Dict[str, Dict[str, Any]]:
        return {n: (f.json_schema_extra or f.metadata or {}) for n, f in self.model_fields.items()}

    def normalize_units(self, units: UnitsSettings) -> "NeutronYieldModel":
        unit_map = units.normalize_units()
        scale_t = unit_map.get("temporal", 1.0)
        win = None
        if self.yield_integration_window_us is not None:
            win = (
                self.yield_integration_window_us[0] * scale_t,
                self.yield_integration_window_us[1] * scale_t,
            )
        return self.model_copy(update={"yield_integration_window_us": win})

    def summarize(self) -> str:
        fuel = self.fusion_fuel_type
        beam = "ON" if self.beam_target_model_enabled else "OFF"
        th = "ON" if self.thermonuclear_model_enabled else "OFF"
        ion = self.beam_ion_species
        sigma = self.fusion_cross_section_model
        ti = (
            str(self.average_ion_temperature_keV)
            if self.average_ion_temperature_keV is not None
            else "n/a"
        )
        spec_bins = (
            f"[{', '.join(str(b).rstrip('0').rstrip('.') for b in self.spectrum_energy_bins_MeV)}]"
            if self.spectrum_energy_bins_MeV
            else "None"
        )
        fmt = self.spectrum_output_format or "n/a"
        anis = "anisotropic" if self.anisotropic_spectrum else "isotropic"
        parts = [
            f"Fusion: {fuel} | Beam-target: {beam}, Ion: {ion}, σ(E): {sigma}",
            f"Thermonuclear: {th}, Maxwellian = {self.maxwellian_assumed}, Ti = {ti} keV",
            f"Branching: DDn = {self.dd_branching_ratio} | Spectrum ({anis}): {spec_bins} MeV → {fmt}",
        ]
        if self.apply_detector_response_function:
            resp = self.detector_response_file.name if self.detector_response_file else "none"
            parts.append(
                f"Detector: applied, Norm = {self.detector_response_normalization}, Response = {resp}"
            )
        else:
            parts.append("Detector: none")
        return "\n".join(parts)

    def hash_neutron_yield_config(self) -> str:
        data = self.model_dump(by_alias=True)
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    # ------------------------------------------------------------------
    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> "NeutronYieldModel":
        alias_map = {to_camel_case(n): n for n in cls.__annotations__}
        for n in list(alias_map):
            if n.endswith("Mev"):
                alias_map[n[:-3] + "MeV"] = alias_map[n]
        cleaned = {
            alias_map.get(k, from_camel_case(k)): v for k, v in data.items()
        }
        inst = cls(**cleaned)
        return cls.check_rules(inst)

    @classmethod
    def check_rules(cls, values: "NeutronYieldModel") -> "NeutronYieldModel":
        if (
            values.thermonuclear_model_enabled
            and values.reactivity_source == "analytic"
        ):
            if (
                values.average_ion_temperature_keV is None
                or values.average_ion_density_cm3 is None
            ):
                raise ValueError(
                    "average_ion_temperature_keV and average_ion_density_cm3 required for analytic reactivity"
                )

        if (
            values.reactivity_source in {"look-up", "FLYCHK"}
            and values.thermonuclear_model_enabled
            and values.reactivity_table_path is None
        ):
            raise ValueError("reactivity_table_path required for table-based reactivity")

        if (
            values.fusion_cross_section_model == "tabulated"
            and values.cross_section_table_path is None
        ):
            raise ValueError("cross_section_table_path required for tabulated cross sections")

        if values.apply_detector_response_function and values.detector_response_file is None:
            raise ValueError("detector_response_file required when apply_detector_response_function is True")

        if values.dd_branching_ratio is not None and not (
            0.0 <= values.dd_branching_ratio <= 1.0
        ):
            raise ValueError("dd_branching_ratio must be between 0 and 1")

        if values.spectrum_energy_bins_MeV is not None:
            if values.spectrum_energy_bins_MeV != sorted(values.spectrum_energy_bins_MeV):
                raise ValueError("spectrum_energy_bins_MeV must be monotonically increasing")

        if values.anisotropic_spectrum and values.spectrum_output_format != "hdf5":
            raise ValueError("anisotropic_spectrum requires spectrum_output_format='hdf5'")

        if values.yield_integration_window_us is not None:
            s, e = values.yield_integration_window_us
            if s >= e:
                raise ValueError("yield_integration_window_us must have start < end")

        return values


class TabulatedIonEDF(IonBeamEDF):
    """Simple in-memory implementation of :class:`IonBeamEDF`.

    The distribution is stored as a mapping from detector angle in degrees to a
    tuple ``(energies, flux)`` where both entries are sequences of equal length
    containing the ion energy grid and the corresponding differential flux
    values.
    """

    def __init__(self, data: Dict[float, Tuple[Sequence[float], Sequence[float]]]):
        self._data: Dict[float, Tuple[List[float], List[float]]] = {
            float(a): ([float(e) for e in en], [float(f) for f in fl])
            for a, (en, fl) in data.items()
        }

    def energy_distribution(self, angle_deg: float) -> Tuple[Sequence[float], Sequence[float]]:
        return self._data.get(float(angle_deg), ([], []))

    @classmethod
    def from_json(cls, path: str | Path) -> "TabulatedIonEDF":
        """Create an instance from a JSON file.

        The JSON structure is expected to have ``angles``, ``energies`` and
        ``distributions`` fields where ``distributions[i]`` corresponds to the
        differential flux at ``angles[i]`` over the shared ``energies`` grid.
        """

        obj = json.loads(Path(path).read_text())
        angles = obj.get("angles", [])
        energies = obj.get("energies", [])
        dists = obj.get("distributions", [])
        if len(angles) != len(dists):
            raise ValueError("angles and distributions length mismatch")
        data = {
            float(ang): (energies, dists[i])
            for i, ang in enumerate(angles)
        }
        return cls(data)


def compute_directional_spectrum(
    ion_edf: IonBeamEDF,
    cross_section: Callable[[float], float],
    angles: Sequence[float],
    energy_bins: Sequence[float],
) -> List[List[float]]:
    """Compute energy spectra ``dN/dE`` for multiple detector angles.

    Parameters
    ----------
    ion_edf:
        Provider of ion energy distributions.
    cross_section:
        Callable returning the reaction cross section for a given energy.
    angles:
        Sequence of detector angles in degrees.
    energy_bins:
        Monotonic sequence of energy bin edges in joules.

    Returns
    -------
    list of list of float
        Spectral yield for each angle and energy bin.
    """

    if any(energy_bins[i] >= energy_bins[i + 1] for i in range(len(energy_bins) - 1)):
        raise ValueError("energy_bins must be monotonically increasing")

    spectra: List[List[float]] = []
    for ang in angles:
        energies, dist = ion_edf.energy_distribution(float(ang))
        e = [float(v) for v in energies]
        f = [float(v) for v in dist]
        if len(e) != len(f):
            raise ValueError("energies and distribution must have the same length")
        hist = [0.0 for _ in range(len(energy_bins) - 1)]
        for i in range(len(e) - 1):
            e1, e2 = e[i], e[i + 1]
            f1, f2 = f[i], f[i + 1]
            s1, s2 = cross_section(e1), cross_section(e2)
            dE = e2 - e1
            contrib = 0.5 * (f1 * s1 + f2 * s2) * dE
            e_mid = (e1 + e2) / 2.0
            idx = bisect_right(energy_bins, e_mid) - 1
            if 0 <= idx < len(hist):
                hist[idx] += contrib
        spectra.append(hist)
    return spectra


__all__ = [
    "NeutronYieldModel",
    "TabulatedIonEDF",
    "compute_directional_spectrum",
]
