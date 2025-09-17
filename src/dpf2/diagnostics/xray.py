from __future__ import annotations

from pathlib import Path
from typing import Sequence, List, Dict, Any, Mapping
import json

from .detector_models import apply_irf


def load_response(
    path: str | Path, overrides: Mapping[str, Any] | None = None
) -> Dict[str, Any]:
    """Load an X-ray detector response description from *path*.

    Parameters
    ----------
    path:
        Location of the JSON configuration file.
    overrides:
        Optional mapping of values that override those read from *path*.
    """
    with open(Path(path), "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if overrides:
        data.update(overrides)
    return data


def apply_response(
    times: Sequence[float],
    signal: Sequence[float],
    response: Dict[str, Any],
) -> List[float]:
    """Apply detector response effects to ``signal`` sampled at ``times``."""

    return apply_irf(times, signal, response)


__all__ = ["load_response", "apply_response"]
