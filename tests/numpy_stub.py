import sys
import types
import math


def _array(data):
    return list(data) if isinstance(data, (list, tuple)) else [data]


def _zeros(shape):
    if isinstance(shape, tuple):
        if len(shape) == 1:
            return [0.0 for _ in range(shape[0])]
        return [_zeros(shape[1:]) for _ in range(shape[0])]
    return [0.0 for _ in range(shape)]


def _linspace(a, b, n):
    if n == 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


def _isclose(a, b, atol=1.0e-8):
    return abs(a - b) <= atol


def _zeros_like(arr):
    return [0.0 for _ in arr]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _exp(vals):
    if isinstance(vals, list):
        return [math.exp(v) for v in vals]
    return math.exp(vals)


def _clip(vals, lo, hi):
    if isinstance(vals, list):
        return [min(max(v, lo), hi) for v in vals]
    return min(max(vals, lo), hi)


np = types.SimpleNamespace(
    array=_array,
    zeros=_zeros,
    linspace=_linspace,
    isclose=_isclose,
    zeros_like=_zeros_like,
    dot=_dot,
    exp=_exp,
    clip=_clip,
    sqrt=math.sqrt,
    arange=lambda n: list(range(n)),
    inf=float("inf"),
)

sys.modules.setdefault("numpy", np)

