"""Synthetic diagnostics package with compatibility shim.

This package exposes the legacy :mod:`dpf2.synthetic_diagnostics` module
API while also providing access to submodules such as
``dpf2.synthetic_diagnostics.modes``.
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

# Load the legacy module residing alongside this package and re-export its
# public API so that existing imports remain valid.
_module_path = Path(__file__).resolve().parent.parent / "synthetic_diagnostics.py"
_spec = spec_from_file_location("dpf2.synthetic_diagnostics_legacy", _module_path)
_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)  # type: ignore[attr-defined]

__all__ = getattr(_module, "__all__", [])
for name in __all__:
    globals()[name] = getattr(_module, name)
