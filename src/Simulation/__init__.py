"""Compatibility package exposing the legacy Simulation namespace."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("simulation")
