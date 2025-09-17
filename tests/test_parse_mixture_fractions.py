import importlib.util
import pathlib
import sys
import types

# Provide minimal stubs for heavy dependencies so the module can be imported
import h5py_stub as h5py

sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("scipy", types.ModuleType("scipy"))
sys.modules.setdefault("scipy.interpolate", types.ModuleType("scipy.interpolate"))
sys.modules["scipy.interpolate"].RegularGridInterpolator = object

spec = importlib.util.spec_from_file_location(
    "eos",
    pathlib.Path(__file__).resolve().parent.parent
    / "src"
    / "dpf2"
    / "simulation"
    / "eos.py",
)
eos = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eos)
parse_mixture_fractions = eos.parse_mixture_fractions


import pytest


def test_parse_mixture_fractions_string():
    fractions = parse_mixture_fractions("A:0.5,B:0.5")
    assert fractions == {"A": 0.5, "B": 0.5}


def test_parse_mixture_fractions_invalid_sum():
    with pytest.raises(ValueError):
        parse_mixture_fractions("A:0.4,B:0.4")


def test_parse_mixture_fractions_negative():
    with pytest.raises(ValueError):
        parse_mixture_fractions("A:-0.1,B:1.1")
