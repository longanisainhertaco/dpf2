from __future__ import annotations

import hashlib
import json
from pathlib import Path

from typing import Any, ClassVar, Dict, List, Optional, Literal, Iterable, Sequence

import csv
import math
try:  # pragma: no cover - h5py may be optional
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    import h5py_stub as h5py  # type: ignore


import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from ..utils.pydantic_compat import model_validator



from ..core_schema import ConfigSectionBase, to_camel_case
from ..units_settings import UnitsSettings
from ..core.bases import CouplingState
from ..diagnostics.synthetic_signals import (
    current_waveform,
    voltage_waveform,
    coupled_current_waveform,
    coupled_voltage_waveform,
    rogowski_signal,
    bdot_signal,
)


class AngularDistribution:
    """Simple histogram of particle counts versus angle."""

    def __init__(self, bins: int = 36) -> None:
        self.bins = bins
        self.edges = np.linspace(-180.0, 180.0, bins + 1)
        self.counts = np.zeros(bins)

    def add(self, angle_deg: float) -> None:
        """Accumulate a count for ``angle_deg``."""

        idx = np.searchsorted(self.edges, angle_deg, side="right") - 1
        if 0 <= idx < self.bins:
            self.counts[idx] += 1.0

    def distribution(self) -> np.ndarray:
        """Return the normalized angular distribution."""

        total = self.counts.sum()
        if total > 0.0:
            return self.counts / total
        return self.counts


def generate_tof_spectrum(
    energies_mev: Sequence[float],
    distance_m: float,
    bins: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic time-of-flight spectrum from neutron energies."""

    m_n = 1.67492749804e-27  # neutron mass (kg)
    tof_vals: List[float] = []
    for e in energies_mev:
        e_j = float(e) * 1.602176634e-13
        v = (2.0 * e_j / m_n) ** 0.5
        tof_vals.append(distance_m / v)
    try:  # pragma: no cover - prefer numpy implementation
        counts, edges = np.histogram(tof_vals, bins=bins)
    except Exception:  # pragma: no cover - fallback for stub numpy
        if not tof_vals:
            edges = np.linspace(0.0, 1.0, bins + 1)
            counts = np.zeros(bins)
        else:
            t_min = min(tof_vals)
            t_max = max(tof_vals)
            if t_max == t_min:
                t_max = t_min + 1e-12
            edges = np.linspace(t_min, t_max, bins + 1)
            counts = np.zeros(bins)
            span = t_max - t_min
            for t in tof_vals:
                idx = int((t - t_min) / span * bins)
                if idx >= bins:
                    idx = bins - 1
                counts[idx] += 1.0
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts


def beam_target_angular_spectrum(
    beam_energy_keV: float,
    n_beam: float,
    n_target: float,
    angles_deg: Sequence[float],
) -> np.ndarray:
    """Convenience wrapper exposing :func:`dd_beam_target_angular_spectrum`.

    Parameters
    ----------
    beam_energy_keV:
        Incident deuteron energy.
    n_beam, n_target:
        Beam and target densities in m^-3.
    angles_deg:
        Sequence of angles spanning ``-180`` to ``180`` degrees.
    """

    return dd_beam_target_angular_spectrum(beam_energy_keV, n_beam, n_target, angles_deg)


def directional_yields(
    beam_energy_keV: float,
    n_beam: float,
    n_target: float,
    bins: int = 360,
) -> Dict[str, float]:
    """Return forward, radial and backward yield components.

    This simply forwards to :func:`fusion.dd_directional_yields` to expose the
    calculation through the :mod:`synthetic_diagnostics` namespace.
    """

    return dd_directional_yields(beam_energy_keV, n_beam, n_target, bins=bins)


def flashover_delay_stats(delays: Sequence[float]) -> Dict[str, float]:
    """Return simple statistics for flashover delays.

    A pure-Python implementation is used to remain compatible with the
    lightweight numerical stubs bundled with the test suite.
    """

    vals = [float(d) for d in delays]
    n = len(vals)
    if n == 0:
        return {"count": 0, "mean": 0.0, "stddev": 0.0}
    mean = sum(vals) / n
    var = sum((d - mean) ** 2 for d in vals) / n
    return {"count": n, "mean": mean, "stddev": var ** 0.5}


def flashover_jitter_stats(holdoffs: Sequence[float]) -> Dict[str, float]:
    """Return jitter statistics for flashover hold-off voltages.

    This simply forwards to :func:`flashover_delay_stats` for clarity.
    """

    return flashover_delay_stats(holdoffs)


def export_directional_yields(path: Path | str, totals: Dict[str, float]) -> Path:
    """Write directional yield ``totals`` to ``path`` in JSON format."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(totals))
    return out_path


def synthetic_tof_trace(
    history: Sequence[CouplingState],
    dt: float,
    distance_m: float,
    energies_mev: Sequence[float],
) -> tuple[List[float], List[float]]:
    """Generate a synthetic time-of-flight trace from a history of states."""

    hist = list(history)
    if not hist or not energies_mev:
        return [], []

    m_n = 1.67492749804e-27  # neutron mass (kg)
    # Maximum time-of-flight determines padding length
    tofs = []
    for e in energies_mev:
        e_j = float(e) * 1.602176634e-13
        v = (2.0 * e_j / m_n) ** 0.5
        tofs.append(distance_m / v)
    max_tof = max(tofs)
    extra = int(math.ceil(max_tof / dt)) + 1
    total = len(hist) + extra
    counts = [0.0] * total
    for i, state in enumerate(hist):
        amp = abs(state.current)
        for tof in tofs:
            idx = i + int(round(tof / dt))
            if idx < total:
                counts[idx] += amp
    times = [i * dt for i in range(total)]
    return times, counts


def autocorrelated_tof_iv_report(
    history: Sequence[CouplingState],
    dt: float,
    distance_m: float,
    energies_mev: Sequence[float],
    output_dir: Path | str = Path("diagnostics/anisotropy"),
) -> Path:
    """Export an auto-correlated ToF versus I–V spike report.

    The correlation is performed between the synthetic ToF trace and the
    instantaneous ``I*V`` product representing power spikes.
    """

    times, counts = synthetic_tof_trace(history, dt, distance_m, energies_mev)
    power = [abs(s.current * s.voltage) for s in history]
    padded_power = power + [0.0] * (len(counts) - len(power))
    mean_counts = sum(counts) / len(counts) if counts else 0.0
    mean_power = sum(padded_power) / len(padded_power) if padded_power else 0.0
    n = len(counts)
    corr = []
    lags = []
    for lag in range(-n + 1, n):
        val = 0.0
        for i in range(n):
            j = i - lag
            if 0 <= j < n:
                val += (counts[i] - mean_counts) * (padded_power[j] - mean_power)
        corr.append(val)
        lags.append(lag * dt)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "tof_iv_autocorrelation.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["lag_s", "correlation"])
        for lag, val in zip(lags, corr):
            writer.writerow([lag, val])
    return path


def anisotropy_report(
    history: Sequence[CouplingState],
    dt: float,
    distance_m: float,
    energies_mev: Sequence[float],
    beam_energy_keV: float = 100.0,
    n_beam: float = 1e18,
    n_target: float = 1e20,
    output_dir: Path | str = Path("diagnostics/anisotropy"),
) -> Dict[str, Path]:
    """Compute yield ratios and correlate ToF counts with ``I*V`` spikes.

    The function aggregates forward, radial and backward yield components
    using :func:`directional_yields`, normalizes them to ratios and performs
    a cross-correlation between the synthetic time-of-flight trace and the
    instantaneous electrical power ``I*V``.  Summary CSV files and (where
    possible) plots are written to ``output_dir``.
    """

    totals = directional_yields(beam_energy_keV, n_beam, n_target)
    total_sum = sum(totals.values()) or 1.0
    ratios = {k: v / total_sum for k, v in totals.items()}

    times, counts = synthetic_tof_trace(history, dt, distance_m, energies_mev)
    power = [abs(s.current * s.voltage) for s in history]
    padded_power = power + [0.0] * (len(counts) - len(power))
    mean_counts = sum(counts) / len(counts) if counts else 0.0
    mean_power = sum(padded_power) / len(padded_power) if padded_power else 0.0
    n = len(counts)
    corr: List[float] = []
    lags: List[float] = []
    for lag in range(-n + 1, n):
        val = 0.0
        for i in range(n):
            j = i - lag
            if 0 <= j < n:
                val += (counts[i] - mean_counts) * (padded_power[j] - mean_power)
        corr.append(val)
        lags.append(lag * dt)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ratio_csv = out_dir / "yield_ratios.csv"
    with ratio_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["component", "ratio"])
        for comp, val in ratios.items():
            writer.writerow([comp, val])

    corr_csv = out_dir / "tof_iv_correlation.csv"
    with corr_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["lag_s", "correlation"])
        for lag, val in zip(lags, corr):
            writer.writerow([lag, val])

    ratio_plot: Optional[Path] = None
    corr_plot: Optional[Path] = None
    try:  # pragma: no cover - matplotlib optional
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.bar(list(ratios.keys()), list(ratios.values()))
        ax.set_ylabel("Yield ratio")
        fig.savefig(out_dir / "yield_ratios.png")
        plt.close(fig)
        ratio_plot = out_dir / "yield_ratios.png"

        fig, ax = plt.subplots()
        ax.plot(lags, corr)
        ax.set_xlabel("Lag (s)")
        ax.set_ylabel("Correlation")
        fig.savefig(out_dir / "tof_iv_correlation.png")
        plt.close(fig)
        corr_plot = out_dir / "tof_iv_correlation.png"
    except Exception:
        pass

    result: Dict[str, Path] = {
        "ratio_csv": ratio_csv,
        "correlation_csv": corr_csv,
    }
    if ratio_plot is not None:
        result["ratio_plot"] = ratio_plot
    if corr_plot is not None:
        result["correlation_plot"] = corr_plot
    return result



def _cr39_image(history: Sequence[CouplingState], size: int = 64) -> List[List[float]]:
    """Return a simple Gaussian spot image scaled by peak current."""

    amp = max((abs(s.current) for s in history), default=0.0)
    x = np.linspace(-1.0, 1.0, size)
    y = np.linspace(-1.0, 1.0, size)
    xv, yv = np.meshgrid(x, y)
    sigma = 0.3
    img = amp * np.exp(-(xv ** 2 + yv ** 2) / (2.0 * sigma ** 2))
    return img.tolist()


def _rcf_image(history: Sequence[CouplingState], size: int = 64) -> List[List[float]]:
    """Return a ring-shaped image scaled by peak voltage."""

    amp = max((abs(s.voltage) for s in history), default=0.0)
    x = np.linspace(-1.0, 1.0, size)
    y = np.linspace(-1.0, 1.0, size)
    xv, yv = np.meshgrid(x, y)
    r = np.sqrt(xv ** 2 + yv ** 2)
    sigma = 0.1
    img = amp * np.exp(-((r - 0.5) ** 2) / (2.0 * sigma ** 2))
    return img.tolist()


def _faraday_iedf(history: Sequence[CouplingState], bins: int = 50) -> List[float]:
    """Generate a synthetic ion energy distribution function."""

    max_energy = max((abs(s.voltage) for s in history), default=1.0)
    energies = np.linspace(0.0, max_energy, bins)
    temp = max_energy / 5.0 if max_energy > 0 else 1.0
    dist = energies * np.exp(-energies / temp)
    return dist.tolist()


def _faraday_eedf(history: Sequence[CouplingState], bins: int = 50) -> List[float]:
    """Generate a synthetic electron energy distribution function."""

    max_energy = max((abs(s.voltage) for s in history), default=1.0)
    energies = np.linspace(0.0, max_energy, bins)
    temp = max_energy / 7.0 if max_energy > 0 else 1.0
    dist = np.sqrt(energies) * np.exp(-energies / temp)
    return dist.tolist()



class SyntheticInstrument(BaseModel):
    """Per-instrument overrides for synthetic diagnostics."""

    response_file: Optional[Path] = None
    noise_model: Optional[str] = None
    geometry: Optional[str] = None
    sampling_override_ns: Optional[float] = Field(
        None, alias="samplingOverrideNs", metadata={"units": "ns"}
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        validate_default=True,
    )


class SyntheticDiagnostics(ConfigSectionBase):
    """Synthetic diagnostic modeling configuration."""

    config_section_id: ClassVar[Literal["synthetic_diagnostics"]] = "synthetic_diagnostics"

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        validate_default=True,
    )

    # ------------------------------------------------------------------
    # Global output configuration
    output_dir: str = "synthetic_diagnostics/"
    output_format: Literal["csv", "hdf5", "ascii"] = "csv"
    sampling_interval_ns: float = Field(1.0, alias="samplingIntervalNs", metadata={"units": "ns"})
    runtime_synthetic_enabled: bool = Field(True, alias="runtimeSyntheticEnabled")
    postprocessing_only: bool = Field(False, alias="postprocessingOnly")
    synthetic_diagnostics_config_hash: Optional[str] = Field(
        None, alias="syntheticDiagnosticsConfigHash"
    )

    # Detector control flags
    apply_time_response: bool = Field(True, alias="applyTimeResponse")
    apply_energy_filter: bool = Field(True, alias="applyEnergyFilter")
    apply_spatial_psf: bool = Field(False, alias="applySpatialPsf")

    # Enabled diagnostics
    synthetic_current_waveform_enabled: bool = Field(True, alias="syntheticCurrentWaveformEnabled")
    synthetic_voltage_waveform_enabled: bool = Field(True, alias="syntheticVoltageWaveformEnabled")
    synthetic_coupled_current_waveform_enabled: bool = Field(
        False, alias="syntheticCoupledCurrentWaveformEnabled"
    )
    synthetic_coupled_voltage_waveform_enabled: bool = Field(
        False, alias="syntheticCoupledVoltageWaveformEnabled"
    )
    synthetic_rogowski_signal_enabled: bool = Field(True, alias="syntheticRogowskiSignalEnabled")
    synthetic_bdot_signal_enabled: bool = Field(True, alias="syntheticBdotSignalEnabled")
    synthetic_neutron_tof_enabled: bool = Field(True, alias="syntheticNeutronTofEnabled")
    synthetic_xray_pinhole_enabled: bool = Field(True, alias="syntheticXrayPinholeEnabled")
    synthetic_thomson_parabola_enabled: bool = Field(False, alias="syntheticThomsonParabolaEnabled")
    synthetic_optical_interferogram_enabled: bool = Field(False, alias="syntheticOpticalInterferogramEnabled")
    synthetic_cr39_image_enabled: bool = Field(False, alias="syntheticCr39ImageEnabled")
    synthetic_rcf_image_enabled: bool = Field(False, alias="syntheticRcfImageEnabled")
    synthetic_faraday_iedf_enabled: bool = Field(False, alias="syntheticFaradayIedfEnabled")
    synthetic_faraday_eedf_enabled: bool = Field(False, alias="syntheticFaradayEedfEnabled")

    # Diagnostic classification and labeling
    detector_ids: Optional[List[str]] = Field(None, alias="detectorIds")
    diagnostic_output_type: Dict[str, Literal["time_series", "spatial_map", "image"]] = Field(
        default_factory=dict, alias="diagnosticOutputType"
    )
    detector_positions_path: Optional[Path] = Field(None, alias="detectorPositionsPath")
    diagnostic_geometry_model: Optional[Literal["1D", "2D", "3D", "raycast"]] = Field(
        None, alias="diagnosticGeometryModel"
    )

    # Global paths
    detector_definitions_path: Optional[Path] = Field(None, alias="detectorDefinitionsPath")
    instrument_response_directory: Optional[Path] = Field(None, alias="instrumentResponseDirectory")

    # Noise and filter modeling
    apply_electrical_filter: bool = Field(False, alias="applyElectricalFilter")
    filter_type: Optional[Literal["RC", "bandpass", "gaussian"]] = Field(None, alias="filterType")
    filter_parameters: Optional[Dict[str, float]] = Field(None, alias="filterParameters")
    include_detector_noise: bool = Field(False, alias="includeDetectorNoise")
    noise_model: Optional[Literal["gaussian", "poisson", "custom"]] = Field(None, alias="noiseModel")
    noise_parameters: Optional[Dict[str, float]] = Field(None, alias="noiseParameters")

    # Per-instrument overrides
    instrument_overrides: Optional[Dict[str, SyntheticInstrument]] = Field(
        None, alias="instrumentOverrides"
    )

    # ------------------------------------------------------------------
    @classmethod
    def with_defaults(cls) -> "SyntheticDiagnostics":
        return cls(apply_time_response=False, apply_energy_filter=False)

    def model_copy(self, update: Optional[Dict[str, Any]] = None, **kwargs: Any) -> "SyntheticDiagnostics":  # type: ignore[override]
        data = self.model_dump()
        if update:
            data.update(update)
        return SyntheticDiagnostics(**data)

    def resolve_defaults(self) -> "SyntheticDiagnostics":
        data = self.model_dump()
        return self.model_validate(data)

    @classmethod
    def required_fields(cls) -> List[str]:
        return [n for n, f in cls.model_fields.items() if f.is_required()]

    @classmethod
    def get_field_metadata(cls) -> Dict[str, Dict[str, Any]]:
        return {n: (f.json_schema_extra or f.metadata or {}) for n, f in cls.model_fields.items()}

    def normalize_units(self, units: UnitsSettings) -> "SyntheticDiagnostics":
        unit_map = units.normalize_units()
        scale = unit_map.get("temporal", 1.0)
        interval = self.sampling_interval_ns * scale
        overrides = None
        if self.instrument_overrides:
            overrides = {}
            for name, inst in self.instrument_overrides.items():
                val = inst.sampling_override_ns
                if val is not None:
                    val = val * scale
                overrides[name] = inst.model_copy(update={"sampling_override_ns": val})
        return self.model_copy(update={"sampling_interval_ns": interval, "instrument_overrides": overrides})

    def summarize(self) -> str:
        diag_flags = [
            (self.synthetic_current_waveform_enabled, "Current"),
            (self.synthetic_voltage_waveform_enabled, "Voltage"),
            (self.synthetic_coupled_current_waveform_enabled, "CoupledCurrent"),
            (self.synthetic_coupled_voltage_waveform_enabled, "CoupledVoltage"),
            (self.synthetic_rogowski_signal_enabled, "Rogowski"),
            (self.synthetic_bdot_signal_enabled, "B-dot"),
            (self.synthetic_neutron_tof_enabled, "TOF"),
            (self.synthetic_xray_pinhole_enabled, "X-ray"),
            (self.synthetic_cr39_image_enabled, "CR39"),
            (self.synthetic_rcf_image_enabled, "RCF"),
            (self.synthetic_faraday_iedf_enabled, "IEDF"),
            (self.synthetic_faraday_eedf_enabled, "EEDF"),
        ]
        active = [name for flag, name in diag_flags if flag]
        filt = "None"
        if self.apply_electrical_filter and self.filter_type and self.filter_parameters:
            cutoff = self.filter_parameters.get("cutoff")
            unit = " Hz" if cutoff is not None else ""
            val = f"{cutoff}{unit}" if cutoff is not None else ""
            filt = f"{self.filter_type}({val})"
        noise = self.noise_model.capitalize() if self.include_detector_noise and self.noise_model else "None"
        num_det = len(self.detector_ids) if self.detector_ids else 0
        ids = ", ".join(self.detector_ids[:2]) if self.detector_ids else ""
        geom = self.diagnostic_geometry_model or "n/a"
        return (
            f"Synthetic Diagnostics: [{', '.join(active)}]\n"
            f"Output: {self.output_format.upper()} @ {self.sampling_interval_ns} ns, "
            f"TimeResponse: {'ON' if self.apply_time_response else 'OFF'}, "
            f"Filter: {filt}, Noise: {noise}\n"
            f"Detectors: {num_det}, Geometry: {geom}, IDs: {ids}"
        )

    def enabled_modules(self) -> List[str]:
        """Return names of diagnostic modules enabled via configuration."""

        mapping = {
            "current_waveform": self.synthetic_current_waveform_enabled,
            "voltage_waveform": self.synthetic_voltage_waveform_enabled,
            "coupled_current_waveform": self.synthetic_coupled_current_waveform_enabled,
            "coupled_voltage_waveform": self.synthetic_coupled_voltage_waveform_enabled,
            "rogowski_signal": self.synthetic_rogowski_signal_enabled,
            "bdot_signal": self.synthetic_bdot_signal_enabled,
            "neutron_tof": self.synthetic_neutron_tof_enabled,
            "xray_pinhole": self.synthetic_xray_pinhole_enabled,
            "cr39_image": self.synthetic_cr39_image_enabled,
            "rcf_image": self.synthetic_rcf_image_enabled,
            "faraday_iedf": self.synthetic_faraday_iedf_enabled,
            "faraday_eedf": self.synthetic_faraday_eedf_enabled,
        }
        return [name for name, flag in mapping.items() if flag]

    def hash_synthetic_diagnostics_config(self) -> str:
        data = self.model_dump(by_alias=True, exclude={"synthetic_diagnostics_config_hash"})
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def check_rules(cls, values: "SyntheticDiagnostics") -> "SyntheticDiagnostics":
        if values.apply_electrical_filter:
            if values.filter_type is None or values.filter_parameters is None:
                raise ValueError("filter_parameters required when apply_electrical_filter is True")
        if values.include_detector_noise:
            if values.noise_model is None or values.noise_parameters is None:
                raise ValueError("noise_parameters required when include_detector_noise is True")
        if (
            values.apply_time_response or values.apply_energy_filter or values.apply_spatial_psf
        ) and values.instrument_response_directory is None:
            raise ValueError("instrument_response_directory required when response modeling enabled")
        if values.instrument_overrides:
            if values.detector_ids:
                for key in values.instrument_overrides.keys():
                    if key not in values.detector_ids:
                        raise ValueError("instrument_override key not listed in detector_ids")
            new_overrides = {}
            for name, inst in values.instrument_overrides.items():
                if values.apply_time_response and inst.response_file is None and values.instrument_response_directory is None:
                    raise ValueError("response_file required for instrument when time response applied")
                if inst.sampling_override_ns is not None and inst.sampling_override_ns <= 0:
                    raise ValueError("sampling_override_ns must be positive")
                new_overrides[name] = inst
            values = values.model_copy(update={"instrument_overrides": new_overrides})
        values = values.model_copy(update={"synthetic_diagnostics_config_hash": values.hash_synthetic_diagnostics_config()})
        return values



def run_diagnostic_calculations(
    history: Iterable[CouplingState],
    cfg: "SyntheticDiagnostics",
    dt: float,
    bdot_radius: float = 0.01,
) -> Dict[str, List[float]]:
    """Compute enabled synthetic diagnostic signals.

    Parameters
    ----------
    history:
        Iterable of :class:`~dpf2.core.bases.CouplingState` objects.
    cfg:
        Diagnostic configuration controlling which calculators run.
    dt:
        Time step between successive states in seconds.
    bdot_radius:
        Probe radius used for ``bdot`` calculations.

    Returns
    -------
    dict
        Mapping of diagnostic name to generated data sequence.
    """

    hist = list(history)
    outputs: Dict[str, List[float]] = {}
    if cfg.synthetic_current_waveform_enabled:
        outputs["current"] = current_waveform(hist)
    if cfg.synthetic_voltage_waveform_enabled:
        outputs["voltage"] = voltage_waveform(hist)
    if cfg.synthetic_coupled_current_waveform_enabled:
        outputs["coupled_current"] = coupled_current_waveform(hist)
    if cfg.synthetic_coupled_voltage_waveform_enabled:
        outputs["coupled_voltage"] = coupled_voltage_waveform(hist)
    if cfg.synthetic_rogowski_signal_enabled:
        outputs["rogowski"] = rogowski_signal(hist, dt)
    if cfg.synthetic_bdot_signal_enabled:
        outputs["bdot"] = bdot_signal(hist, bdot_radius, dt)

    if cfg.synthetic_cr39_image_enabled:
        outputs["cr39_image"] = _cr39_image(hist)
    if cfg.synthetic_rcf_image_enabled:
        outputs["rcf_image"] = _rcf_image(hist)
    if cfg.synthetic_faraday_iedf_enabled:
        outputs["faraday_iedf"] = _faraday_iedf(hist)
    if cfg.synthetic_faraday_eedf_enabled:
        outputs["faraday_eedf"] = _faraday_eedf(hist)

    return outputs


def _export_csv(path: Path, data: Sequence[Any], kind: str) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        if kind == "time_series":
            writer.writerow(["index", "value"])
            for i, val in enumerate(data):
                writer.writerow([i, val])
        else:
            for row in data:
                writer.writerow(list(row))


def _export_hdf5(path: Path, name: str, data: Sequence[Any]) -> None:
    with h5py.File(path, "w") as fh:
        fh.create_dataset(name, data=data)


def export_diagnostic_data(
    data: Dict[str, Sequence[Any]],
    cfg: "SyntheticDiagnostics",
    output_dir: Path | str,
) -> List[Path]:
    """Write diagnostic ``data`` to ``output_dir`` according to ``cfg``."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    dtype = cfg.diagnostic_output_type or {}
    for name, values in data.items():
        kind = dtype.get(name, "time_series")
        if cfg.output_format == "csv":
            file_path = out_dir / f"{name}.csv"
            _export_csv(file_path, values, kind)
        elif cfg.output_format == "hdf5":
            file_path = out_dir / f"{name}.h5"
            _export_hdf5(file_path, name, values)
        else:
            file_path = out_dir / f"{name}.txt"
            if kind == "time_series":
                file_path.write_text("\n".join(str(v) for v in values))
            else:
                file_path.write_text("\n".join(",".join(str(v) for v in row) for row in values))
        written.append(file_path)
    return written


def _sd_model_validate(cls, data: Any, *args: Any, **kwargs: Any) -> "SyntheticDiagnostics":
    if isinstance(data, dict):
        annotations = getattr(cls, "__annotations__", {})
        mapping = {to_camel_case(name): name for name in annotations}
        data = {mapping.get(k, k): v for k, v in data.items()}
    obj = BaseModel.model_validate.__func__(cls, data, *args, **kwargs)
    # Manually invoke validation hook when running with the lightweight stub
    if hasattr(cls, "check_rules"):
        try:
            obj = cls.check_rules(cls, obj)  # type: ignore[arg-type]
        except Exception as exc:
            raise exc
    return obj


SyntheticDiagnostics.model_validate = classmethod(_sd_model_validate)  # type: ignore[assignment]


__all__ = [
    "SyntheticDiagnostics",
    "SyntheticInstrument",
    "run_diagnostic_calculations",
    "export_diagnostic_data",
    "generate_tof_spectrum",
    "beam_target_angular_spectrum",
    "directional_yields",
    "synthetic_tof_trace",
    "export_directional_yields",
    "autocorrelated_tof_iv_report",
    "anisotropy_report",
    "flashover_delay_stats",
    "flashover_jitter_stats",
]
