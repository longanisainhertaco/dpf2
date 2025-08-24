from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class RestartManager:
    """Handle writing and reading simple restart files."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def save(self, state: Dict[str, Any]) -> None:
        """Persist simulation state to a restart file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(state, f, sort_keys=True)

    def load(self) -> Dict[str, Any]:
        """Load simulation state from the restart file."""
        with self.path.open("r", encoding="utf-8") as f:
            return json.load(f)
