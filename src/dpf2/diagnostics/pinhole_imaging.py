from __future__ import annotations

import math
from typing import Callable, List, Sequence, Tuple


def pinhole_image(
    source_positions: Sequence[Tuple[float, float, float]],
    source_intensities: Sequence[float],
    detector_distance: float,
    detector_pixels: Tuple[int, int],
    pixel_size: float,
    response_fn: Callable[[float], float] | None = None,
    noise_fn: Callable[[float], float] | None = None,
) -> List[List[float]]:
    """Generate a simple pinhole camera image from point sources.

    Parameters
    ----------
    source_positions:
        Sequence of ``(x, y, z)`` source positions in meters.
    source_intensities:
        Corresponding source strengths (arbitrary units).
    detector_distance:
        Distance from pinhole to detector plane in meters.
    detector_pixels:
        Number of pixels ``(nx, ny)`` on the detector plane.
    pixel_size:
        Physical size of each pixel in meters.
    response_fn, noise_fn:
        Optional callables applied to each pixel value after accumulation.
        ``response_fn`` is evaluated first and ``noise_fn`` should return a
        noise contribution which is then added to the response-corrected
        pixel value.

    Returns
    -------
    List[List[float]]
        2D image array with intensity accumulated per pixel.
    """
    if len(source_positions) != len(source_intensities):
        raise ValueError(
            "source_positions and source_intensities must be the same length"
        )
    if detector_distance <= 0 or pixel_size <= 0:
        raise ValueError("detector_distance and pixel_size must be positive")
    nx, ny = detector_pixels
    image: List[List[float]] = [[0.0 for _ in range(nx)] for _ in range(ny)]
    half_x = nx * pixel_size / 2.0
    half_y = ny * pixel_size / 2.0
    for (x, y, z), I in zip(source_positions, source_intensities):
        if z == 0.0:
            continue  # avoid divide-by-zero
        scale = detector_distance / z
        x_det = x * scale
        y_det = y * scale
        i = int((x_det + half_x) / pixel_size)
        j = int((y_det + half_y) / pixel_size)
        if 0 <= i < nx and 0 <= j < ny:
            r2 = x_det * x_det + y_det * y_det + detector_distance * detector_distance
            image[j][i] += I / (4.0 * math.pi * r2)
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


__all__ = ["pinhole_image"]
