"""Compatibility helpers bridging Pydantic v1 and v2 APIs."""

from __future__ import annotations

try:  # pragma: no cover - use real pydantic when available
    from pydantic import BaseModel, ConfigDict, Field, root_validator  # type: ignore
except Exception:  # pragma: no cover - fallback to lightweight stubs
    from pydantic_stub import BaseModel, ConfigDict, Field, root_validator  # type: ignore


def model_validator(*, mode: str = "after"):
    """Polyfill for the :func:`pydantic.v2` ``model_validator`` decorator."""

    def decorator(func):
        if mode == "after":
            def wrapper(cls, values):
                inst = values if isinstance(values, cls) else cls.construct(**values)
                result = func(cls, inst)
                return result.__dict__ if isinstance(result, cls) else values

            return root_validator(pre=False, skip_on_failure=True, allow_reuse=True)(wrapper)
        else:
            def wrapper(cls, values):
                out = func(values)
                return out if out is not None else values

            return root_validator(pre=True, skip_on_failure=True, allow_reuse=True)(wrapper)

    return decorator


# ---------------------------------------------------------------------------
# ``BaseModel`` polyfills mirroring pydantic v2 behavior

if not hasattr(BaseModel, "parse_obj"):
    BaseModel.parse_obj = classmethod(lambda cls, data, **_: cls(**data))  # type: ignore[attr-defined]

if not hasattr(BaseModel, "model_validate"):
    BaseModel.model_validate = classmethod(
        lambda cls, data, **_: cls.parse_obj(data)  # type: ignore[attr-defined]
    )

if not hasattr(BaseModel, "model_dump"):
    if hasattr(BaseModel, "dict"):
        BaseModel.model_dump = BaseModel.dict  # type: ignore[attr-defined]
    else:  # pragma: no cover - minimal stub
        BaseModel.model_dump = (
            lambda self, *_, **__: getattr(self, "__dict__", {})
        )  # type: ignore[attr-defined]

if not hasattr(BaseModel, "model_dump_json"):
    if hasattr(BaseModel, "json"):
        BaseModel.model_dump_json = BaseModel.json  # type: ignore[attr-defined]
    else:  # pragma: no cover - minimal stub
        import json as _json

        BaseModel.model_dump_json = (
            lambda self, *_, **__: _json.dumps(getattr(self, "__dict__", {}))
        )  # type: ignore[attr-defined]

if not hasattr(BaseModel, "model_copy"):
    import inspect as _inspect

    def _copy(self, update=None, **__):  # type: ignore
        if hasattr(self.__class__, "copy"):
            sig = _inspect.signature(self.__class__.copy)
            if "update" in sig.parameters:
                new = self.__class__.copy(self, update=update)  # type: ignore[misc]
            else:
                new = self.__class__.copy(self)  # type: ignore[misc]
        else:  # pragma: no cover - minimal stub
            new = self.__class__()  # type: ignore[attr-defined]
            for k, v in getattr(self, "__dict__", {}).items():
                setattr(new, k, v)
        if update:
            for k, v in update.items():
                setattr(new, k, v)
        return new

    BaseModel.model_copy = _copy  # type: ignore[attr-defined]


if not hasattr(BaseModel, "model_rebuild"):
    if hasattr(BaseModel, "update_forward_refs"):
        BaseModel.model_rebuild = classmethod(
            lambda cls, *_, **__: cls.update_forward_refs()  # type: ignore[attr-defined]
        )
    else:  # pragma: no cover - minimal stub
        BaseModel.model_rebuild = classmethod(lambda cls, *_, **__: None)

__all__ = ["model_validator"]
