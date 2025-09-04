from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Literal

import numpy as np

from pydantic import BaseModel, ConfigDict, Field



from .core_schema import ConfigSectionBase, UnitsSystem, UNIT_SCALE_MAP, to_camel_case


class ValidationSuite(ConfigSectionBase):
    """Validated configuration for benchmarking simulation outputs."""

    config_section_id: ClassVar[Literal["validation_suite"]] = "validation_suite"

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        alias_generator=to_camel_case,
        populate_by_name=True,
        allow_population_by_field_name=True,
        validate_default=True,
    )

    # ------------------------------------------------------------------
    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> "ValidationSuite":
        norm: Dict[str, Any] = {}
        for name in cls.__annotations__:
            alias = to_camel_case(name)
            if alias in data:
                norm[name] = data[alias]
            elif name in data:
                norm[name] = data[name]
        inst = cls(**norm)
        inst.check_rules()
        return inst

    # ------------------------------------------------------------------
    # Experimental metadata
    experiment_device_id: Literal["PF1000", "NX2", "UNU", "LLNL-DPF", "custom"] = Field(
        ..., alias="experimentDeviceId"
    )
    experiment_campaign_id: str = Field(..., alias="experimentCampaignId")
    dataset_directory: Path = Field(..., alias="datasetDirectory")
    dataset_format: Literal["csv", "json", "hdf5"] = Field(..., alias="datasetFormat")
    observable_file_map: Dict[str, Path] = Field(..., alias="observableFileMap")
    observable_format_spec: Optional[Dict[str, Dict[str, str]]] = Field(
        None, alias="observableFormatSpec"
    )
    observable_uncertainties: Optional[Dict[str, float]] = Field(
        None, alias="observableUncertainties"
    )
    observable_uncertainty_ranges: Optional[Dict[str, Tuple[float, float]]] = Field(
        None, alias="observableUncertaintyRanges"
    )

    # Validation target configuration
    validation_targets: List[str] = Field(
        default_factory=lambda: ["I(t)", "Yn"], alias="validationTargets"
    )
    observable_tolerances: Dict[str, float] = Field(..., alias="observableTolerances")
    observable_weighting: Optional[Dict[str, float]] = Field(
        None, alias="observableWeighting"
    )
    validation_score_model: Literal["L2", "RMSE", "MAE", "weighted"] = Field(
        "RMSE", alias="validationScoreModel"
    )
    require_all_targets: bool = Field(True, alias="requireAllTargets")
    score_pass_threshold: float = Field(0.85, alias="scorePassThreshold")
    computed_validation_score: Optional[float] = Field(
        None, alias="computedValidationScore"
    )
    validation_passed: Optional[bool] = Field(None, alias="validationPassed")

    # Timing / matching
    validation_time_window_us: Optional[Tuple[float, float]] = Field(
        None, alias="validationTimeWindowUs"
    )
    resample_method: Optional[Literal["interpolate", "zero_order", "downsample"]] = Field(
        "interpolate", alias="resampleMethod"
    )
    interpolation_mode: Optional[Literal["linear", "cubic", "spline"]] = Field(
        "linear", alias="interpolationMode"
    )
    match_on_t_peak: bool = Field(False, alias="matchOnTPeak")

    # ------------------------------------------------------------------
    @classmethod
    def with_defaults(cls) -> "ValidationSuite":
        return cls(
            experiment_device_id="PF1000",
            experiment_campaign_id="shot0001",
            dataset_directory=Path("data/experiments"),
            dataset_format="csv",
            observable_file_map={"I(t)": Path("current.csv"), "Yn": Path("yield.csv")},
            observable_tolerances={"I(t)": 0.1, "Yn": 0.3},
        )

    def resolve_defaults(self) -> "ValidationSuite":
        data = self.model_dump()
        return self.model_validate(data)

    @classmethod
    def required_fields(cls) -> List[str]:
        return [n for n, f in cls.model_fields.items() if f.is_required()]

    @classmethod
    def get_field_metadata(cls) -> Dict[str, Dict[str, Any]]:
        return {n: (f.json_schema_extra or f.metadata or {}) for n, f in cls.model_fields.items()}

    def normalize_units(self, base_units: UnitsSystem) -> "ValidationSuite":
        scale = UNIT_SCALE_MAP.get(base_units, 1.0)
        win = None
        if self.validation_time_window_us is not None:
            win = (
                self.validation_time_window_us[0] * scale,
                self.validation_time_window_us[1] * scale,
            )
        return self.model_copy(update={"validation_time_window_us": win})

    def summarize(self) -> str:
        targets = ", ".join(self.validation_targets)
        tol_vals = [self.observable_tolerances.get(t, 0.0) for t in self.validation_targets]
        tstr = ", ".join(f"{v*100:.0f}%" for v in tol_vals)
        resample = self.resample_method or "none"
        interp = self.interpolation_mode or "linear"
        match = "ON" if self.match_on_t_peak else "OFF"
        return (
            "Validation Suite:\n"
            f"  Device = {self.experiment_device_id}, Campaign = {self.experiment_campaign_id}\n"
            f"  Targets: {targets} | Tolerance: {tstr}\n"
            f"  Score Model: {self.validation_score_model} → Pass ≥ {self.score_pass_threshold}\n"
            f"  Resample = {resample}({interp}), Match on T-peak: {match}\n"
            f"  Format: {self.dataset_format}, Files in {self.dataset_directory}"
        )

    def hash_validation_suite_config(self) -> str:
        data = self.model_dump(by_alias=True, exclude={"computed_validation_score", "validation_passed"})
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    # ------------------------------------------------------------------
    def check_rules(self) -> None:
        if not Path(self.dataset_directory).exists():
            raise ValueError("dataset_directory must exist")
        for obs, path in self.observable_file_map.items():
            p = Path(path)
            if not p.exists():
                raise ValueError(f"observable file {p} must exist")
        if self.observable_format_spec:
            if set(self.observable_format_spec.keys()) != set(self.observable_file_map.keys()):
                raise ValueError("observable_format_spec keys must match observable_file_map")
            for spec in self.observable_format_spec.values():
                if "time" not in spec or "value" not in spec:
                    raise ValueError("observable_format_spec entries must contain time and value")
        if self.validation_time_window_us is not None:
            start, end = self.validation_time_window_us
            if start >= end:
                raise ValueError("validation_time_window_us start must be < end")
        if self.require_all_targets:
            missing = [t for t in self.validation_targets if t not in self.observable_file_map]
            if missing:
                raise ValueError(f"missing targets in observable_file_map: {missing}")
        weights = self.observable_weighting or {t: 1.0 for t in self.validation_targets}
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("observable_weighting must sum to > 0")
        norm_weights = {k: v / total for k, v in weights.items()}
        if self.observable_weighting:
            self.observable_weighting = norm_weights

        # Combine uncertainties from ranges and scalars ----------------
        uncertainties = dict(self.observable_uncertainties or {})
        if self.observable_uncertainty_ranges:
            for obs, rng in self.observable_uncertainty_ranges.items():
                if len(rng) != 2:
                    raise ValueError("uncertainty ranges must have two entries")
                lo, hi = rng
                if lo > hi:
                    raise ValueError("uncertainty range lower bound must be <= upper bound")
                uncertainties[obs] = (hi - lo) / 2.0

        if uncertainties and self.validation_score_model == "weighted":
            score = 1.0
            for t in self.validation_targets:
                u = uncertainties.get(t, 0.0)
                w = norm_weights.get(t, 0.0)
                score -= u * w
            self.computed_validation_score = score
            self.observable_uncertainties = uncertainties
        score = getattr(self, "computed_validation_score", None)
        if score is not None:
            self.validation_passed = score >= self.score_pass_threshold


def _load_profile_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a time-series profile from a two-column CSV file."""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, 0], data[:, 1]


def load_benchmark_dataset(dataset_dir: Path) -> Dict[str, Any]:
    """Return benchmark GV timing and L(t)/I(t) profiles from ``dataset_dir``."""
    gv_path = dataset_dir / "gv_timing.json"
    current_path = dataset_dir / "current_profile.csv"
    inductance_path = dataset_dir / "inductance_profile.csv"
    benchmark = {
        "gv_time_us": json.loads(gv_path.read_text())["gv_time_us"],
        "I": _load_profile_csv(current_path),
        "L": _load_profile_csv(inductance_path),
    }
    return benchmark


def compare_gv_timing(sim_gv_us: float, ref_gv_us: float) -> float:
    """Absolute difference between simulated and reference GV timing."""
    return abs(sim_gv_us - ref_gv_us)


def compare_profiles(
    sim_profile: Tuple[np.ndarray, np.ndarray],
    ref_profile: Tuple[np.ndarray, np.ndarray],
) -> float:
    """RMSE between a simulated profile and reference profile."""
    sim_t, sim_v = sim_profile
    ref_t, ref_v = ref_profile
    interp_v = np.interp(ref_t, sim_t, sim_v)
    diff = interp_v - ref_v
    return float(np.sqrt(np.mean(diff * diff)))


def compute_error_metrics(
    sim_outputs: Dict[str, Any], dataset_dir: Path, tolerances: Dict[str, float]
) -> Dict[str, Any]:
    """Compute error metrics for GV timing and L(t)/I(t) profiles.

    Parameters
    ----------
    sim_outputs:
        Mapping containing ``gv_time_us`` and profile tuples ``I`` and ``L``.
    dataset_dir:
        Directory with benchmark CSV/JSON files.
    tolerances:
        Acceptable error bounds for ``gv_timing_us``, ``I(t)`` and ``L(t)``.
    """

    bench = load_benchmark_dataset(dataset_dir)
    errors = {
        "gv_timing_us": compare_gv_timing(
            sim_outputs["gv_time_us"], bench["gv_time_us"]
        ),
        "I_rmse": compare_profiles(sim_outputs["I"], bench["I"]),
        "L_rmse": compare_profiles(sim_outputs["L"], bench["L"]),
    }
    errors["passed"] = (
        errors["gv_timing_us"] <= tolerances.get("gv_timing_us", float("inf"))
        and errors["I_rmse"] <= tolerances.get("I(t)", float("inf"))
        and errors["L_rmse"] <= tolerances.get("L(t)", float("inf"))
    )
    return errors


def load_pinch_dataset(dataset_dir: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load benchmark pinch traces from ``dataset_dir``.

    The directory must contain CSV files named ``current.csv``, ``voltage.csv``,
    ``neutron_yield.csv`` and ``radius.csv`` each with two columns:
    ``time`` and ``value``.
    """

    traces = {}
    for name in ["current", "voltage", "neutron_yield", "radius"]:
        traces[name] = _load_profile_csv(dataset_dir / f"{name}.csv")
    return traces


def _rmse(sim: Tuple[np.ndarray, np.ndarray], ref: Tuple[np.ndarray, np.ndarray]) -> float:
    """Compute RMSE between two time series."""

    sim_t, sim_v = sim
    ref_t, ref_v = ref
    interp_v = np.interp(ref_t, sim_t, sim_v)
    diff = interp_v - ref_v
    return float(np.sqrt(np.mean(diff * diff)))


def _peak_time(profile: Tuple[np.ndarray, np.ndarray]) -> float:
    t, v = profile
    return float(t[int(np.argmax(v))])


def _peak_timing_error(
    sim: Tuple[np.ndarray, np.ndarray], ref: Tuple[np.ndarray, np.ndarray]
) -> float:
    """Absolute difference in peak times between two profiles."""

    return abs(_peak_time(sim) - _peak_time(ref))


def _integrated_energy(
    current: Tuple[np.ndarray, np.ndarray], voltage: Tuple[np.ndarray, np.ndarray]
) -> float:
    t, i = current
    v = np.interp(t, voltage[0], voltage[1])
    return float(np.trapz(i * v, t))


def _energy_balance_error(
    sim_I: Tuple[np.ndarray, np.ndarray],
    sim_V: Tuple[np.ndarray, np.ndarray],
    ref_I: Tuple[np.ndarray, np.ndarray],
    ref_V: Tuple[np.ndarray, np.ndarray],
) -> float:
    """Difference in discharge energy between simulation and reference."""

    return abs(
        _integrated_energy(sim_I, sim_V) - _integrated_energy(ref_I, ref_V)
    )


def compute_pinch_error_metrics(
    sim_outputs: Dict[str, Tuple[np.ndarray, np.ndarray]],
    dataset_dir: Path,
    tolerances: Dict[str, float],
) -> Dict[str, Any]:
    """Compute RMSE, peak timing and energy balance errors for pinch traces."""

    bench = load_pinch_dataset(dataset_dir)
    errors = {
        f"{name}_rmse": _rmse(sim_outputs[name], bench[name])
        for name in ["current", "voltage", "neutron_yield", "radius"]
    }
    errors.update(
        {
            f"{name}_t_peak": _peak_timing_error(
                sim_outputs[name], bench[name]
            )
            for name in ["current", "voltage", "radius"]
        }
    )
    errors["energy_diff"] = _energy_balance_error(
        sim_outputs["current"],
        sim_outputs["voltage"],
        bench["current"],
        bench["voltage"],
    )
    errors["passed"] = all(
        errors.get(k, 0.0) <= tolerances.get(k, float("inf")) for k in errors
    )
    return errors


# ---------------------------------------------------------------------------
# Experimental validation helpers

# Root directory for packaged validation datasets ---------------------------
VALIDATION_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "validation"


def load_validation_dataset(device: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load waveform and yield data for ``device`` from packaged CSV files."""

    dataset_dir = VALIDATION_DATA_DIR / device.upper()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"unknown device '{device}'")
    traces: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name in ["current", "voltage", "neutron_yield"]:
        traces[name] = _load_profile_csv(dataset_dir / f"{name}.csv")
    return traces


def resample_profile(
    profile: Tuple[np.ndarray, np.ndarray],
    new_t: np.ndarray,
    *,
    method: str = "interpolate",
) -> np.ndarray:
    """Resample ``profile`` onto ``new_t`` using ``method``."""

    t, v = profile
    if method == "interpolate":
        return np.interp(new_t, t, v)
    if method == "zero_order":
        idx = np.searchsorted(t, new_t, side="right") - 1
        idx = np.clip(idx, 0, len(v) - 1)
        return v[idx]
    if method == "downsample":
        factor = max(1, len(t) // len(new_t))
        return v[::factor][: len(new_t)]
    raise ValueError(f"unknown resample method '{method}'")


def score_simulation(
    sim_outputs: Dict[str, Tuple[np.ndarray, np.ndarray]],
    device: str,
    tolerances: Dict[str, float],
    *,
    resample_method: str = "interpolate",
    weights: Optional[Dict[str, float]] = None,
    pass_threshold: float = 0.85,
) -> Dict[str, Any]:
    """Compute per-observable and aggregate validation scores.

    Parameters
    ----------
    sim_outputs:
        Mapping of observable name to ``(time, value)`` tuples representing the
        simulation result.
    device:
        Identifier of the experimental dataset to compare against.  The dataset
        must be present under :mod:`data/validation`.
    tolerances:
        Relative error tolerance for each observable.  Values are interpreted as
        fractions of the peak reference magnitude.
    resample_method:
        Method passed to :func:`resample_profile` for aligning simulation time
        grids with the reference data.
    weights:
        Optional weighting for each observable when computing the aggregate
        score.  If omitted, all observables contribute equally.
    pass_threshold:
        Minimum aggregate score required for a "passed" result.
    """

    ref = load_validation_dataset(device)
    scores: Dict[str, float] = {}
    rmse_metrics: Dict[str, float] = {}
    l2_metrics: Dict[str, float] = {}
    for name, ref_profile in ref.items():
        if name not in sim_outputs:
            continue
        ref_t, ref_v = ref_profile
        sim_t, sim_v = sim_outputs[name]
        sim_rs = resample_profile((sim_t, sim_v), ref_t, method=resample_method)
        diff = sim_rs - ref_v
        rmse = float(np.sqrt(np.mean(diff**2)))
        l2 = float(np.sqrt(np.sum(diff**2)))
        rmse_metrics[name] = rmse
        l2_metrics[name] = l2
        norm = np.max(np.abs(ref_v)) or 1.0
        tol = tolerances.get(name, 1.0)
        scores[name] = max(0.0, 1.0 - rmse / (norm * tol))

    if not scores:
        overall = 0.0
    else:
        if weights:
            total_w = sum(weights.get(k, 0.0) for k in scores)
            total_w = total_w or 1.0
            overall = (
                sum(scores[k] * weights.get(k, 0.0) for k in scores) / total_w
            )
        else:
            overall = sum(scores.values()) / len(scores)
    return {
        "scores": scores,
        "rmse": rmse_metrics,
        "l2": l2_metrics,
        "overall": overall,
        "passed": overall >= pass_threshold,
    }


__all__ = [
    "ValidationSuite",
    "load_benchmark_dataset",
    "compare_gv_timing",
    "compare_profiles",
    "compute_error_metrics",
    "load_pinch_dataset",
    "compute_pinch_error_metrics",
    "load_validation_dataset",
    "resample_profile",
    "score_simulation",
]


