from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict

try:  # optional yaml support
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


class StructuredOutputWriter:
    """Write structured diagnostic output such as JSON or YAML."""

    def __init__(self, output_dir: str, config: Dict[str, Any] | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
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

    def write_json(self, data: Dict[str, Any], name: str) -> Path:
        path = self.output_dir / (name if name.endswith(".json") else f"{name}.json")
        payload = {"data": data, "metadata": self.metadata}
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        return path

    def write_yaml(self, data: Dict[str, Any], name: str) -> Path:
        if yaml is None:
            raise RuntimeError("yaml package is required for YAML output")
        path = self.output_dir / (name if name.endswith(".yaml") else f"{name}.yaml")
        payload = {"data": data, "metadata": self.metadata}
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f)
        return path
