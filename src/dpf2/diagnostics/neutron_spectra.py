from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_right
from pathlib import Path
from typing import Dict, Sequence, List, Tuple
import json
import math
import numpy as np

# Neutron mass used for simple time-of-flight calculations (kg)
M_N = 1.674e-27


@dataclass
class Detector:
    """Simple representation of a neutron detector."""

    angle_deg: float
    distance_m: float
    name: str


@dataclass
class DetectorLayout:
    """Container describing a collection of detectors on a ring."""

    angles: Sequence[float]
    distance_m: float | Sequence[float]
    names: Sequence[str] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.distance_m, Sequence) and not isinstance(
            self.distance_m, str
        ):
            if len(self.distance_m) != len(self.angles):
                raise ValueError("distance_m sequence must match number of angles")
            distances = [float(d) for d in self.distance_m]
        else:
            if float(self.distance_m) <= 0:
                raise ValueError("distance_m must be positive")
            distances = [float(self.distance_m) for _ in self.angles]
        if self.names and len(self.names) != len(self.angles):
            raise ValueError("names must match number of angles")
        self.detectors: List[Detector] = []
        for i, ang in enumerate(self.angles):
            name = self.names[i] if self.names else f"detector_{i}"
            self.detectors.append(Detector(float(ang), distances[i], name))

    def angles_deg(self) -> List[float]:
        """Return detector viewing angles in degrees."""

        return [d.angle_deg for d in self.detectors]

    def names_list(self) -> List[str]:
        """Return detector names in layout order."""

        return [d.name for d in self.detectors]


def synthetic_tof_spectrum(
    energies: Sequence[float],
    flux: Sequence[float],
    distance: float,
    time_bins: Sequence[float],
    m_n: float = M_N,
) -> List[float]:
    """Generate a simple neutron time-of-flight histogram.

    Parameters
    ----------
    energies:
        Monotonically increasing energy grid in joules.
    flux:
        Differential flux corresponding to ``energies``.
    distance:
        Source-to-detector distance in meters.
    time_bins:
        Edges of time-of-flight histogram bins in seconds.
    m_n:
        Neutron mass in kg.  Defaults to :data:`M_N`.

    Returns
    -------
    list of float
        Bin-integrated counts for the requested histogram.
    """

    if len(energies) != len(flux):
        raise ValueError("energies and flux must be same length")
    if any(time_bins[i] >= time_bins[i + 1] for i in range(len(time_bins) - 1)):
        raise ValueError("time_bins must be monotonically increasing")
    hist = [0.0 for _ in range(len(time_bins) - 1)]
    for e1, e2, f1, f2 in zip(energies[:-1], energies[1:], flux[:-1], flux[1:]):
        dE = e2 - e1
        contrib = 0.5 * (f1 + f2) * dE
        e_mid = 0.5 * (e1 + e2)
        t = distance / math.sqrt(2.0 * e_mid / m_n)
        idx = bisect_right(time_bins, t) - 1
        if 0 <= idx < len(hist):
            hist[idx] += contrib
    return hist


def angular_spectrum(
    angles: Sequence[float],
    base_yield: float,
    anisotropy: float = 0.0,
) -> List[float]:
    """Create a simple angular yield spectrum using a cosine model."""

    spectrum: List[float] = []
    for ang in angles:
        val = base_yield * (1.0 + anisotropy * math.cos(math.radians(ang)))
        spectrum.append(float(val))
    return spectrum


def anisotropy_metric(values: Sequence[float]) -> float:
    """Return a rudimentary anisotropy metric ``(max - min) / mean``."""

    if not values:
        return 0.0
    mean = sum(float(v) for v in values) / len(values)
    if mean == 0.0:
        return 0.0
    return (max(values) - min(values)) / mean


def load_detector_layout(path: str | Path) -> DetectorLayout:
    """Load a :class:`DetectorLayout` from a JSON or STL file.

    Parameters
    ----------
    path:
        Path to the geometry description.  JSON expects ``{"distance_m":
        float, "detectors": [{"angle_deg": float, "name": str}]}``.
        STL parsing requires the :mod:`trimesh` package and uses the centroid
        of each geometry component to determine detector angles and distances.
    """

    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".json":
        with p.open("r", encoding="utf8") as f:
            data = json.load(f)
        dets = data.get("detectors", [])
        angles = [float(d["angle_deg"]) for d in dets]
        distances = [
            float(d.get("distance_m", data.get("distance_m", 0.0))) for d in dets
        ]
        names = [d.get("name") for d in dets]
        return DetectorLayout(angles=angles, distance_m=distances, names=names)
    if suffix == ".stl":
        try:
            import trimesh  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("trimesh is required for STL layouts") from exc
        mesh = trimesh.load_mesh(str(p))
        if isinstance(mesh, trimesh.Scene):  # pragma: no cover - unlikely
            geoms = list(mesh.geometry.values())
        else:
            geoms = mesh.split(only_watertight=False)
        angles: List[float] = []
        distances: List[float] = []
        names: List[str] = []
        for i, g in enumerate(geoms):
            centroid = g.centroid
            ang = math.degrees(math.atan2(float(centroid[1]), float(centroid[0])))
            dist = float(math.hypot(float(centroid[0]), float(centroid[1])))
            angles.append(ang)
            distances.append(dist)
            names.append(f"detector_{i}")
        return DetectorLayout(angles=angles, distance_m=distances, names=names)
    raise ValueError("Unsupported detector layout format")


def time_resolved_spectra(
    layout: DetectorLayout,
    energies: Sequence[float],
    flux: Sequence[float],
    time_bins: Sequence[float],
    base_yield: float = 1.0,
    anisotropy: float = 0.0,
) -> Dict[str, List[float]]:
    """Compute time-resolved spectra for each detector in ``layout``.

    ``base_yield`` and ``anisotropy`` control a simple cosine angular yield
    model applied to each detector before the time-of-flight histogram is
    generated.
    """

    angle_weights = angular_spectrum(layout.angles_deg(), base_yield, anisotropy)
    spectra: Dict[str, List[float]] = {}
    for det, weight in zip(layout.detectors, angle_weights):
        hist = synthetic_tof_spectrum(energies, flux, det.distance_m, time_bins)
        hist = [h * weight for h in hist]
        spectra[det.name] = hist
    return spectra


def directional_time_resolved_spectra(
    layout: DetectorLayout,
    energies: Sequence[float],
    flux: Sequence[float],
    time_bins: Sequence[float],
    base_yield: float = 1.0,
    anisotropy: float = 0.0,
) -> Dict[str, List[float]]:
    """Return forward, radial and backward time-resolved spectra.

    This convenience wrapper combines :func:`time_resolved_spectra` with
    directional aggregation so that the resulting histograms are grouped by
    detector orientation.  The returned dictionary contains keys
    ``"forward"``, ``"radial"`` and ``"backward"`` with values describing the
    counts in each time bin for detectors in that group.
    """

    per_det = time_resolved_spectra(
        layout, energies, flux, time_bins, base_yield=base_yield, anisotropy=anisotropy
    )
    grouped: Dict[str, List[float]] = {
        "forward": [0.0] * (len(time_bins) - 1),
        "radial": [0.0] * (len(time_bins) - 1),
        "backward": [0.0] * (len(time_bins) - 1),
    }
    for det in layout.detectors:
        ang = det.angle_deg % 360.0
        if ang <= 45.0 or ang >= 315.0:
            group = "forward"
        elif 135.0 <= ang <= 225.0:
            group = "backward"
        else:
            group = "radial"
        hist = per_det.get(det.name, [0.0] * (len(time_bins) - 1))
        grouped[group] = [g + h for g, h in zip(grouped[group], hist)]
    return grouped


def forward_radial_backward_counts(
    layout: DetectorLayout, spectra: Dict[str, Sequence[float]]
) -> Dict[str, float]:
    """Aggregate counts into forward, radial and backward groups."""

    totals: Dict[str, float] = {"forward": 0.0, "radial": 0.0, "backward": 0.0}
    for det in layout.detectors:
        ang = det.angle_deg % 360.0
        count = float(sum(spectra.get(det.name, [])))
        if ang <= 45.0 or ang >= 315.0:
            totals["forward"] += count
        elif 135.0 <= ang <= 225.0:
            totals["backward"] += count
        else:
            totals["radial"] += count
    return totals


def directional_counts_from_geometry(
    geometry: str | Path,
    energies: Sequence[float],
    flux: Sequence[float],
    time_bins: Sequence[float],
    base_yield: float = 1.0,
    anisotropy: float = 0.0,
) -> Dict[str, float]:
    """Load a geometry file and return forward/radial/backward totals."""

    layout = load_detector_layout(geometry)
    spectra = time_resolved_spectra(
        layout, energies, flux, time_bins, base_yield=base_yield, anisotropy=anisotropy
    )
    return forward_radial_backward_counts(layout, spectra)


def anisotropy_ratios(counts: Dict[str, float]) -> Dict[str, float]:
    """Return simple forward/backward and radial/backward ratios."""

    backward = counts.get("backward", 0.0)
    ratios: Dict[str, float] = {}
    if backward > 0.0:
        ratios["forward_backward"] = counts.get("forward", 0.0) / backward
        ratios["radial_backward"] = counts.get("radial", 0.0) / backward
    return ratios


def cross_correlate_tof_with_circuit(
    time_bins: Sequence[float],
    counts: Sequence[float],
    circuit_time: Sequence[float],
    circuit_signal: Sequence[float],
) -> Tuple[List[float], List[float], float]:
    """Cross-correlate a ToF spectrum with a circuit waveform.

    Returns the correlation array, corresponding lags in seconds and the lag at
    maximum correlation.  The circuit waveform is interpolated onto the midpoints
    of the ToF bins before correlation.
    """

    if len(time_bins) < 2:
        raise ValueError("time_bins must have at least two entries")
    mid = 0.5 * (np.asarray(time_bins[:-1]) + np.asarray(time_bins[1:]))
    interp = np.interp(mid, np.asarray(circuit_time), np.asarray(circuit_signal))
    a = np.asarray(counts) - np.mean(counts)
    b = interp - np.mean(interp)
    if hasattr(np, "correlate"):
        corr = np.correlate(a, b, mode="full")
    else:  # pragma: no cover - exercised only with numpy stub lacking correlate
        # simple Python correlation implementation
        a_list = list(a)
        b_list = list(b)
        corr = [0.0] * (len(a_list) + len(b_list) - 1)
        for i, av in enumerate(a_list):
            for j, bv in enumerate(b_list):
                corr[i + j] += av * bv
        corr = np.asarray(corr)
    dt = mid[1] - mid[0] if len(mid) > 1 else 0.0
    lags = [(i - len(mid) + 1) * dt for i in range(len(corr))]
    max_lag = float(lags[int(np.argmax(corr))])
    return lags, list(corr), max_lag


def correlate_tof_peaks_with_circuit_iv(
    time_bins: Sequence[float],
    counts: Sequence[float],
    circuit_time: Sequence[float],
    current: Sequence[float],
    voltage: Sequence[float],
) -> Tuple[List[Tuple[float, float]], List[float], List[float], float]:
    """Correlate ToF peaks with circuit power derived from ``I`` and ``V``.

    The instantaneous power trace (``I * V``) is interpolated onto the
    midpoints of the ToF histogram and cross-correlated with the counts.  In
    addition, local maxima in the ToF spectrum are paired with the
    corresponding circuit power to provide a direct comparison of peak
    features.
    """

    power = np.asarray(current) * np.asarray(voltage)
    lags, corr, max_lag = cross_correlate_tof_with_circuit(
        time_bins, counts, circuit_time, power
    )
    if len(time_bins) < 2:
        raise ValueError("time_bins must have at least two entries")
    mid = 0.5 * (np.asarray(time_bins[:-1]) + np.asarray(time_bins[1:]))
    counts_arr = np.asarray(counts)
    peak_idx = [
        i
        for i in range(1, len(counts_arr) - 1)
        if counts_arr[i] > counts_arr[i - 1] and counts_arr[i] > counts_arr[i + 1]
    ]
    peak_times = [float(mid[i]) for i in peak_idx]
    peak_power = np.interp(peak_times, np.asarray(circuit_time), power)
    peaks = [(t, float(p)) for t, p in zip(peak_times, peak_power)]
    return peaks, lags, corr, max_lag


__all__ = [
    "Detector",
    "DetectorLayout",
    "synthetic_tof_spectrum",
    "angular_spectrum",
    "anisotropy_metric",
    "load_detector_layout",
    "time_resolved_spectra",
    "directional_time_resolved_spectra",
    "forward_radial_backward_counts",
    "directional_counts_from_geometry",
    "anisotropy_ratios",
    "cross_correlate_tof_with_circuit",
    "correlate_tof_peaks_with_circuit_iv",
]
