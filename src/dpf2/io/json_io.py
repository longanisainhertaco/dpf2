from __future__ import annotations

from pathlib import Path
from typing import Any

from dpf2.dpf_config import DPFConfig


def export_config(cfg: DPFConfig, path: Path) -> None:
    """Write a :class:`~dpf2.dpf_config.DPFConfig` to ``path`` in JSON format."""
    path = Path(path)
    path.write_text(cfg.model_dump_json(indent=2))


def import_config(path: Path) -> DPFConfig:
    """Read ``path`` and return a :class:`~dpf2.dpf_config.DPFConfig`."""
    path = Path(path)
    return DPFConfig.model_validate_json(path.read_text())


__all__ = ["export_config", "import_config"]
