from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

try:  # optional yaml support
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


class StructuredOutputWriter:
    """Write structured diagnostic output such as JSON or YAML."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, data: Dict[str, Any], name: str) -> Path:
        path = self.output_dir / (name if name.endswith(".json") else f"{name}.json")
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        return path

    def write_yaml(self, data: Dict[str, Any], name: str) -> Path:
        if yaml is None:
            raise RuntimeError("yaml package is required for YAML output")
        path = self.output_dir / (name if name.endswith(".yaml") else f"{name}.yaml")
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)
        return path
