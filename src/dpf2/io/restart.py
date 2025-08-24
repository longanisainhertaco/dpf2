from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple


class RestartManager:
    """Handle writing and reading restart files with provenance metadata."""

    def __init__(self, path: Path | str, config: Dict[str, Any] | None = None) -> None:
        self.path = Path(path)
        self.metadata = {
            "config_hash": self._hash_config(config),
            "git_commit": self._git_commit(),
        }

    @staticmethod
    def _hash_config(config: Dict[str, Any] | None) -> str:
        if config is None:
            return "unknown"
        data = json.dumps(config, sort_keys=True).encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def _git_commit() -> str:
        try:
            repo = Path(__file__).resolve().parents[2]
            return (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo)
                .decode()
                .strip()
            )
        except Exception:  # pragma: no cover - git not available
            return "unknown"

    def save(self, state: Dict[str, Any]) -> None:
        """Persist simulation state and metadata to a restart file."""
        payload = {"state": state, "metadata": self.metadata}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True)

    def load(self) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Load simulation state and metadata from the restart file."""
        with self.path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if "state" in payload:
            return payload["state"], payload.get("metadata", {})
        # backwards compatibility: file contained only state
        return payload, {}
