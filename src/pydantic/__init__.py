from __future__ import annotations

import dataclasses as _dc
import types as _types
import sys


class BaseModel:
    """Very small subset of :class:`pydantic.BaseModel`.

    This stub stores provided keyword arguments as attributes and implements a
    handful of convenience methods used within the tests.  It is **not** a full
    validation library.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def dict(self, *_, **__):  # pragma: no cover - minimal helper
        return self.__dict__

    def json(self, *_, **__):  # pragma: no cover - minimal helper
        import json

        return json.dumps(self.__dict__)

    def copy(self, *_, **__):  # pragma: no cover - minimal helper
        return self.__class__(**self.__dict__)


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

# ``pydantic.dataclasses`` compatibility
_dataclasses = _types.ModuleType("dataclasses")
_dataclasses.dataclass = _dc.dataclass

dataclasses = _types.ModuleType("pydantic.dataclasses")
dataclasses.dataclass = _dc.dataclass
sys.modules[__name__ + ".dataclasses"] = dataclasses


class ValidationError(Exception):  # pragma: no cover - simple stub
    """Minimal stand-in for :class:`pydantic.ValidationError`."""

    pass
