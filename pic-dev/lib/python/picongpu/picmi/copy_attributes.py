"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import inspect
from typing import Callable


def has_attribute(instance, name):
    # It should be this:
    #
    #     return hasattr(instance, name)
    #
    # But this seems to interact weirdly with our util.build_typesafe_property.
    #
    # This version works fine but throws a warning for pydantic models.
    # We could get rid of this by using instance.model_dump() instead.
    return name in dir(instance)


def _sanitize_conversions(conversions, from_instance, to_instance):
    conversions = conversions or {}

    for new_key, old in conversions.items():
        if not has_attribute(to_instance, new_key):
            failed_conversion = {new_key: old}
            message = (
                f"Conversion failed. '{new_key}' is not a member of the receiver {to_instance}. "
                f"You gave {failed_conversion}."
            )
            raise ValueError(message)
        if isinstance(old, str) and not has_attribute(from_instance, old):
            failed_conversion = {new_key: old}
            message = (
                f"Conversion failed. '{old}' is not a member of the provider {from_instance}. "
                f"You gave {failed_conversion}."
            )
            raise ValueError(message)

    return {name: _value_generator(old) if isinstance(old, str) else old for name, old in conversions.items()}


def _value_generator(name):
    return lambda self: getattr(self, name)


def copy_attributes(
    from_instance,
    to,
    conversions: None | dict[str, str | Callable] = None,
    remove_prefix: str = "",
    ignore=tuple(),
    default_converter=lambda self: self,
):
    """
    Copy attributes from one object to another.

    This function copies attributes from the `from_instance` to `to` if `to` is an class instance.
    If `to` is a class, an instance will be created via `to()`, filled and returned.

    This function only copies attributes under the following circumstances:
        - The attribute exists in `to`.
        - The attribute is not private in `from_instance`.

    Optionally, `conversions` allows to define custom mappings between attributes.
    Keys are strings and interpreted as the name of the attribute to be set on `to`.
    For the values, two semantics are supported:
        - If a `value` is a string, the `(key, value)` pair is interpreted as renaming, i.e.
          `to.key = from_instance.value`.
        - Otherwise, `value` must be a Callable that takes `from_instance` as first and only argument
          and returns the value that `key` is supposed to have in `to`, i.e.
          `to.key = value(from_instance)`.
    """
    if isinstance(to, type):
        try:
            to_instance = to()
        except TypeError as e:
            message = (
                "Instantiation failed. The receiving class must be default constructible. "
                f"You gave {to} which expects {len(inspect.signature(to.__init__).parameters) - 1} argument in its constructor. "
                "You can work with an instance instead of a class in this case."
            )
            raise ValueError(message) from e
        return copy_attributes(
            from_instance,
            to_instance,
            conversions=conversions,
            remove_prefix=remove_prefix,
            ignore=ignore,
            default_converter=default_converter,
        )

    assignments = {
        to_name: _value_generator(from_name)
        for from_name, _ in inspect.getmembers(from_instance)
        if from_name not in ignore
        and not from_name.startswith("_")
        and has_attribute(to, to_name := from_name.removeprefix(remove_prefix))
    } | _sanitize_conversions(conversions, from_instance, to)

    for key, value_generator in assignments.items():
        setattr(to, key, default_converter(value_generator(from_instance)))
    return to


def converts_to(
    to_class,
    conversions=None,
    preamble=None,
    remove_prefix="",
    ignore=tuple(),
    default_converter=lambda self, *args, **kwargs: self,
):
    """
    Add a get_as_pypicongpu method that uses copy_attributes.

    This class decorator adds a method `get_as_pypicongpu`
    that copies the content of `self` to a new instance of `to_class`
    by virtue of the `copy_attributes` function.
    See the `copy_attributes` docstring for details.

    If given, it runs `preamble(self)` before copying.
    """
    if has_attribute(to_class, "get_as_pypicongpu"):
        message = f"Adding 'get_as_pypicongpu' failed because it already existed on receiving class {to_class}."
        raise TypeError(message)

    def decorator(cls):
        def get_as_pypicongpu(self, *args, **kwargs):
            if preamble:
                preamble(self, *args, **kwargs)
            local_conversions = {
                key: value
                if isinstance(value, str)
                # Python binds lambdas by reference, so if we were to use the seemingly obvious implementation,
                # all lambdas in this dictionary would use the last value of `value`.
                # That is why we need to force immediate evaluation:
                else (lambda local_value: lambda self: local_value(self, *args, **kwargs))(value)
                for key, value in (conversions or {}).items()
            }
            return copy_attributes(
                self,
                to_class,
                conversions=local_conversions,
                remove_prefix=remove_prefix,
                ignore=ignore,
                default_converter=lambda self: default_converter(self, *args, **kwargs),
            )

        cls.get_as_pypicongpu = get_as_pypicongpu
        return cls

    return decorator


def default_converts_to(to_class, conversions=None, preamble=None, remove_prefix="", ignore=tuple()):
    return converts_to(
        to_class,
        conversions=conversions,
        preamble=preamble
        or (lambda self, *args, **kwargs: self.check(*args, **kwargs) if has_attribute(self, "check") else None),
        remove_prefix=remove_prefix or "picongpu_",
        ignore=ignore or ("check",),
        default_converter=lambda self, *args, **kwargs: self.get_as_pypicongpu(*args, **kwargs)
        if has_attribute(self, "get_as_pypicongpu")
        else self,
    )
