from __future__ import annotations


def paschen_breakdown_time(gap: float, pressure: float, voltage: float) -> float:
    """Estimate the breakdown delay using a simple Paschen-like scaling.

    The model assumes the delay scales with the reduced field ``(p * d) / V``.
    The caller is responsible for using consistent units for all arguments.

    Parameters
    ----------
    gap:
        Electrode separation distance.
    pressure:
        Gas pressure.
    voltage:
        Applied voltage.

    Returns
    -------
    float
        Estimated time to breakdown.
    """
    if voltage <= 0:
        raise ValueError("voltage must be positive")
    return gap * pressure / voltage
