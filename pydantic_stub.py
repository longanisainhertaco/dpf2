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


    # Mimic minimal pydantic API used in the code base
    def dict(self, **kwargs):  # pragma: no cover - trivial
        return self.__dict__.copy()

    def json(self, **kwargs):  # pragma: no cover - simple serialization
        return str(self.dict())

    def copy(self, **kwargs):  # pragma: no cover - shallow copy
        return type(self)(**self.dict())

    @classmethod
    def parse_obj(cls, data):  # pragma: no cover - simple constructor
        return cls(**data)



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

import sys as _sys

_sys.modules.setdefault("pydantic", _sys.modules[__name__])

