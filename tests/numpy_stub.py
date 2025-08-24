"""Lightweight ``numpy`` substitute used for tests without the real dependency.

The test-suite for this kata exercises only a tiny subset of ``numpy``.  A
small, pure Python stand-in is provided so that the physics modules can be
executed on systems where the compiled ``numpy`` package is unavailable.  The
implementation is intentionally compact and supports just the operations needed
by the tests:

* creation of one and two dimensional arrays via :func:`array`,
  :func:`zeros` and :func:`vstack`;
* element-wise arithmetic with scalars or other arrays;
* basic transcendental functions such as :func:`sin` and :func:`exp`;
* simple reductions like :func:`dot`, :func:`max` and :func:`abs`;
* utilities :func:`linspace`, :func:`arange`, :func:`gradient`,
  :func:`isclose` and :func:`allclose`.

The goal of the stub is not to be fast or feature complete – it merely mimics
enough of the ``numpy`` interface for the pedagogical Hall‑MHD examples.
"""

from __future__ import annotations

import math
import sys
from copy import deepcopy
from typing import Iterable, Sequence


class Array:
    """Very small array type implementing a subset of ``numpy.ndarray``."""

    def __init__(self, data):
        if isinstance(data, Array):
            data = data.data
        self.data = data

    # ------------------------------------------------------------------
    # container protocol
    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            # handle simple 2-D column access ``arr[:, j]``
            if len(idx) == 2 and isinstance(idx[0], slice) and idx[0] == slice(None) and isinstance(idx[1], int):
                return Array([row[idx[1]] for row in self.data])
            arr = self.data
            for i in idx:
                arr = arr[i]
            return Array(arr) if isinstance(arr, list) else arr
        item = self.data[idx]
        return Array(item) if isinstance(item, list) else item

    def __setitem__(self, idx, value):
        if isinstance(value, Array):
            value = value.data
        if isinstance(idx, tuple):
            if len(idx) == 2 and isinstance(idx[0], slice) and idx[0] == slice(None) and isinstance(idx[1], int):
                for row, val in zip(self.data, value):
                    row[idx[1]] = val
                return
            arr = self.data
            for i in idx[:-1]:
                arr = arr[i]
            arr[idx[-1]] = value
        else:
            self.data[idx] = value

    def copy(self):
        return Array(deepcopy(self.data))

    @property
    def shape(self):
        if isinstance(self.data, list):
            if len(self.data) == 0:
                return (0,)
            if isinstance(self.data[0], list):
                return (len(self.data),) + Array(self.data[0]).shape
            return (len(self.data),)
        return ()

    # ------------------------------------------------------------------
    # arithmetic operations
    def _binary(self, other, op):
        if isinstance(other, Array):
            other = other.data
        if isinstance(self.data, list):
            if isinstance(other, list):
                return Array([Array(a)._binary(b, op).data for a, b in zip(self.data, other)])
            return Array([Array(a)._binary(other, op).data for a in self.data])
        return Array(op(self.data, other))

    def __add__(self, other):
        return self._binary(other, lambda a, b: a + b)

    def __sub__(self, other):
        return self._binary(other, lambda a, b: a - b)

    def __mul__(self, other):
        return self._binary(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._binary(other, lambda a, b: a / b)

    def __neg__(self):
        if isinstance(self.data, list):
            return Array([(-Array(x)).data if isinstance(x, list) else -x for x in self.data])
        return Array(-self.data)

    # helper used by ``__neg__``
    def __repr__(self):  # pragma: no cover - debug helper
        return f"Array({self.data!r})"


# ----------------------------------------------------------------------
# Constructors and elementary functions
def array(data):
    if isinstance(data, Array):
        return data
    if isinstance(data, (list, tuple)):
        return Array([array(x).data if isinstance(x, (list, tuple)) else x for x in data])
    return Array(data)


def zeros(shape):
    if isinstance(shape, tuple):
        if len(shape) == 1:
            return Array([0.0] * shape[0])
        return Array([zeros(shape[1:]).data for _ in range(shape[0])])
    return Array([0.0] * shape)


def zeros_like(arr):
    return zeros(array(arr).shape)


def vstack(arrs: Sequence[Array]) -> Array:
    return Array([array(a).data for a in arrs])


def linspace(a: float, b: float, n: int) -> Array:
    if n == 1:
        return Array([a])
    step = (b - a) / (n - 1)
    return Array([a + i * step for i in range(n)])


def arange(n: int) -> Array:
    return Array(list(range(n)))


def sin(vals):
    arr = array(vals)
    if isinstance(arr.data, list):
        return Array([sin(v).data if isinstance(v, list) else math.sin(v) for v in arr.data])
    return Array(math.sin(arr.data))


def exp(vals):
    arr = array(vals)
    if isinstance(arr.data, list):
        return Array([exp(v).data if isinstance(v, list) else math.exp(v) for v in arr.data])
    return Array(math.exp(arr.data))


def abs_(vals):
    arr = array(vals)
    if isinstance(arr.data, list):
        return Array([abs_(v).data if isinstance(v, list) else abs(v) for v in arr.data])
    return Array(abs(arr.data))


def max_(vals):
    arr = array(vals)
    if isinstance(arr.data, list):
        return max(arr.data)
    return arr.data


def dot(a: Array, b: Array) -> float:
    return sum(x * y for x, y in zip(array(a), array(b)))


def cross(a: Array, b: Array) -> Array:
    ax, ay, az = array(a)
    bx, by, bz = array(b)
    return Array([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx])


def clip(vals, lo, hi):
    arr = array(vals)
    if isinstance(arr.data, list):
        return Array([clip(v, lo, hi).data if isinstance(v, list) else min(max(v, lo), hi) for v in arr.data])
    return Array(min(max(arr.data, lo), hi))


def gradient(vals, dx, edge_order=2):
    arr = array(vals).data
    n = len(arr)
    grad = []
    for i in range(n):
        if i == 0:
            grad.append((arr[1] - arr[0]) / dx)
        elif i == n - 1:
            grad.append((arr[-1] - arr[-2]) / dx)
        else:
            grad.append((arr[i + 1] - arr[i - 1]) / (2 * dx))
    return Array(grad)


def isclose(a, b, rtol=1.0e-8, atol=1.0e-8):
    a = array(a)
    b = array(b)
    if isinstance(a.data, list):
        return [isclose(x, y, rtol, atol) for x, y in zip(a.data, b.data)]
    return abs(a.data - b.data) <= atol + rtol * abs(b.data)


def allclose(a, b, rtol=1.0e-8, atol=1.0e-8):
    comp = isclose(a, b, rtol, atol)
    if isinstance(comp, list):
        return all(comp)
    return bool(comp)


sqrt = math.sqrt


# Register the stub as ``numpy`` so ``import numpy as np`` works.
import types

np = types.SimpleNamespace(
    array=array,
    zeros=zeros,
    zeros_like=zeros_like,
    vstack=vstack,
    linspace=linspace,
    arange=arange,
    sin=sin,
    exp=exp,
    abs=abs_,
    max=max_,
    dot=dot,
    cross=cross,
    clip=clip,
    sqrt=sqrt,
    gradient=gradient,
    isclose=isclose,
    allclose=allclose,
    Array=Array,
    inf=float("inf"),
    pi=math.pi,
)

sys.modules.setdefault("numpy", np)

__all__ = ["Array", "array", "zeros", "zeros_like", "vstack", "linspace", "arange", "sin", "exp", "abs_", "max_", "dot", "clip", "sqrt", "gradient", "isclose", "allclose", "np"]

