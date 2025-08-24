"""Legacy simulation components for compatibility.

This package re-exposes the historical ``Simulation`` modules under the new
``dpf2.simulation`` namespace.  To maintain backwards compatibility with the
original flat module structure, lightweight proxies are inserted into
``sys.modules`` so that statements like ``from utils import FieldManager``
continue to function when the simulation package is imported.
"""

from importlib import import_module
import pkgutil
import sys
import types


class _ModuleProxy(types.ModuleType):
    """Lazily import simulation submodules on first attribute access."""

    def __init__(self, name: str) -> None:  # pragma: no cover - trivial
        super().__init__(name)
        self._name = name

    def _load(self) -> types.ModuleType:
        module = import_module(f"{__name__}.{self._name}")
        sys.modules[self._name] = module
        return module

    def __getattr__(self, attr: str):  # pragma: no cover - simple delegation
        return getattr(self._load(), attr)


for info in pkgutil.iter_modules(__path__):
    sys.modules.setdefault(info.name, _ModuleProxy(info.name))

__all__ = [m.name for m in pkgutil.iter_modules(__path__)]
