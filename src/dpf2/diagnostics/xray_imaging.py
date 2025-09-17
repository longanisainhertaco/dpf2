from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Filter pack definitions
# ---------------------------------------------------------------------------


def _be_filter(E: float) -> float:
    """Crude beryllium filter transmission."""

    return 0.5 if E <= 1.0 else 0.2


def _al_filter(E: float) -> float:
    """Crude aluminium filter transmission."""

    return 0.4 if E <= 2.0 else 0.1


def _ti_filter(E: float) -> float:
    """Crude titanium filter transmission."""

    return 0.3 if E <= 4.0 else 0.05


FILTER_PACKS: Dict[str, Callable[[float], float]] = {
    "open": lambda E: 1.0,
    "Be": _be_filter,
    "Al": _al_filter,
    "Ti": _ti_filter,
    "BeAl": lambda E: _be_filter(E) * _al_filter(E),
    "AlTi": lambda E: _al_filter(E) * _ti_filter(E),
}


def apply_filter_pack(energies: Sequence[float], pack: str) -> List[float]:
    """Apply transmission of a filter pack to photon energies."""

    if pack not in FILTER_PACKS:
        raise ValueError(f"Unknown filter pack '{pack}'")
    fn = FILTER_PACKS[pack]
    return [float(E) * fn(float(E)) for E in energies]


def xray_image(
    photon_positions: Sequence[Tuple[float, float, float]],
    photon_energies: Sequence[float],
    detector_distance: float,
    detector_pixels: Tuple[int, int],
    pixel_size: float,
    response_fn: Callable[[float], float] | None = None,
    noise_fn: Callable[[float], float] | None = None,
) -> List[List[float]]:
    """Form a simple pinhole X-ray image from photon positions and energies.

    Parameters
    ----------
    photon_positions:
        Sequence of ``(x, y, z)`` photon emission points in meters.
    photon_energies:
        Corresponding photon energies (arbitrary units) used as weights.
    detector_distance:
        Distance from pinhole to detector plane in meters.
    detector_pixels:
        Number of pixels ``(nx, ny)`` on the detector plane.
    pixel_size:
        Physical size of each pixel in meters.
    response_fn, noise_fn:
        Optional callables applied to each pixel after accumulation. ``response_fn``
        is evaluated first and ``noise_fn`` should return a noise contribution to
        be added to the response-corrected value.

    Returns
    -------
    List[List[float]]
        2D image array with accumulated energy per pixel.
    """
    if len(photon_positions) != len(photon_energies):
        raise ValueError("photon_positions and photon_energies must be the same length")
    if detector_distance <= 0 or pixel_size <= 0:
        raise ValueError("detector_distance and pixel_size must be positive")
    nx, ny = detector_pixels
    image: List[List[float]] = [[0.0 for _ in range(nx)] for _ in range(ny)]
    half_x = nx * pixel_size / 2.0
    half_y = ny * pixel_size / 2.0
    for (x, y, z), E in zip(photon_positions, photon_energies):
        if z == 0.0:
            continue
        scale = detector_distance / z
        x_det = x * scale
        y_det = y * scale
        i = int((x_det + half_x) / pixel_size)
        j = int((y_det + half_y) / pixel_size)
        if 0 <= i < nx and 0 <= j < ny:
            image[j][i] += float(E)
    if response_fn or noise_fn:
        for j in range(ny):
            for i in range(nx):
                val = image[j][i]
                if response_fn:
                    val = response_fn(val)
                if noise_fn:
                    val += noise_fn(val)
                image[j][i] = val
    return image


def pinhole_camera(
    photon_positions: Sequence[Tuple[float, float, float]],
    photon_energies: Sequence[float],
    detector_distance: float,
    detector_pixels: Tuple[int, int],
    pixel_size: float,
    filter_pack: str = "open",
    noise_fn: Callable[[float], float] | None = None,
) -> List[List[float]]:
    """Synthesize a pinhole camera image including filter pack effects."""

    energies = apply_filter_pack(photon_energies, filter_pack)
    return xray_image(
        photon_positions,
        energies,
        detector_distance,
        detector_pixels,
        pixel_size,
        response_fn=None,
        noise_fn=noise_fn,
    )


__all__ = [
    "xray_image",
    "pinhole_camera",
    "apply_filter_pack",
    "FILTER_PACKS",
]
