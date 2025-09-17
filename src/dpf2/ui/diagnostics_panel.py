"""Simple helpers for applying instrument response functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

from ..diagnostics.detector_models import apply_irf


@dataclass
class DiagnosticsPanelUI:
    """UI helper exposing IRF application for diagnostics."""

    instrument_response: Dict[str, Any] | None = None

    def apply_irf(self, times: Sequence[float], signal: Sequence[float]) -> list[float]:
        """Return ``signal`` after applying the configured IRF."""
        if self.instrument_response:
            return apply_irf(times, signal, self.instrument_response)
        return list(signal)


__all__ = ["DiagnosticsPanelUI"]
