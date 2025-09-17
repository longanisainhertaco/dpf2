"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Pawel Ordyna
License: GPLv3+
"""

import typeguard

from ...pypicongpu.output.auto import Auto as PyPIConGPUAuto
from ..copy_attributes import default_converts_to
from .timestepspec import TimeStepSpec


@default_converts_to(PyPIConGPUAuto)
@typeguard.typechecked
class Auto:
    """
    Specifies the parameters for the Auto output.

    Parameters
    ----------
    period: int
        Number of simulation steps between consecutive outputs.
        Unit: steps (simulation time steps).
    """

    period: TimeStepSpec
    """Number of simulation steps between consecutive outputs. Unit: steps (simulation time steps)."""

    def __init__(self, period: TimeStepSpec) -> None:
        self.period = period
