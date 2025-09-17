from __future__ import annotations

"""Lightweight lab-mode UI helpers."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class LabModeUI:
    """Simple interface for enabling jitter and exporting results."""

    jitter_enabled: bool = False

    def toggle_jitter(self, enable: bool) -> None:
        """Enable or disable stochastic jitter."""
        self.jitter_enabled = bool(enable)

    def export_results(self, results: Dict[str, Any], path: str | Path) -> Path:
        """Export ``results`` to ``path`` as JSON."""
        p = Path(path)
        p.write_text(json.dumps(results, indent=2))
        return p


__all__ = ["LabModeUI"]
