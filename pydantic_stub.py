"""Lightweight stub of :mod:`pydantic` used for documentation builds and tests."""

from __future__ import annotations

import dataclasses as _dc
import types as _types


class BaseModel:
    """Minimal stand-in for :class:`pydantic.BaseModel`.

    The implementation only stores provided keyword arguments as attributes and
    performs no validation.  It is sufficient for tests that need to import
    modules with an optional ``pydantic`` dependency without installing the real
    package.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def validator(*args, **kwargs):  # pragma: no cover - trivial passthrough
    def decorator(func):
        return func

    return decorator


def root_validator(*args, **kwargs):  # pragma: no cover - trivial passthrough
    def decorator(func):
        return func

    return decorator


def Field(default=None, **kwargs):  # pragma: no cover - trivial passthrough
    return default


ConfigDict = dict

# ---------------------------------------------------------------------------
# ``pydantic.dataclasses`` compatibility
# ---------------------------------------------------------------------------
dataclasses = _types.ModuleType("pydantic.dataclasses")
dataclasses.dataclass = _dc.dataclass

