"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari, Julian Lenz
License: GPLv3+
"""

from .. import util
from ..species import Species

from .plugin import Plugin
from .timestepspec import TimeStepSpec


import typeguard
import typing
from enum import Enum


class EMFieldScaleEnum(Enum):
    AUTO = -1
    PLASMA_WAVE = 3
    CUSTOM = 6
    INCIDENT = 7

    @classmethod
    def _missing_(cls, value):
        """Ensure strings map correctly to Enum values."""
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"{value} is not a valid {cls.__name__}")


class ColorScaleEnum(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    GRAY = "gray"
    GRAY_INV = "grayInv"
    NONE = "none"

    @classmethod
    def _missing_(cls, value):
        """Ensure strings map correctly to Enum values."""
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"{value} is not a valid {cls.__name__}")


@typeguard.typechecked
class Png(Plugin):
    species = util.build_typesafe_property(Species)
    period = util.build_typesafe_property(TimeStepSpec)
    axis = util.build_typesafe_property(str)
    slice_point = util.build_typesafe_property(float)
    folder_name = util.build_typesafe_property(str)
    scale_image = util.build_typesafe_property(float)
    scale_to_cellsize = util.build_typesafe_property(bool)
    white_box_per_gpu = util.build_typesafe_property(bool)
    em_field_scale_channel1 = util.build_typesafe_property(EMFieldScaleEnum)
    em_field_scale_channel2 = util.build_typesafe_property(EMFieldScaleEnum)
    em_field_scale_channel3 = util.build_typesafe_property(EMFieldScaleEnum)
    pre_particle_density_color_scales = util.build_typesafe_property(ColorScaleEnum)
    pre_channel1_color_scales = util.build_typesafe_property(ColorScaleEnum)
    pre_channel2_color_scales = util.build_typesafe_property(ColorScaleEnum)
    pre_channel3_color_scales = util.build_typesafe_property(ColorScaleEnum)
    custom_normalization_si = util.build_typesafe_property(typing.List[float])
    pre_particle_density_opacity = util.build_typesafe_property(float)
    pre_channel1_opacity = util.build_typesafe_property(float)
    pre_channel2_opacity = util.build_typesafe_property(float)
    pre_channel3_opacity = util.build_typesafe_property(float)
    pre_channel1 = util.build_typesafe_property(str)
    pre_channel2 = util.build_typesafe_property(str)
    pre_channel3 = util.build_typesafe_property(str)

    _name = "png"

    def __init__(self):
        "do nothing"

    def _get_serialized(self) -> typing.Dict:
        """Return the serialized representation of the object."""

        # Transform customNormalizationSI into a list of dictionaries
        custom_normalization_si_serialized = [{"value": val} for val in self.custom_normalization_si]

        return {
            "species": self.species.get_rendering_context(),
            "period": self.period.get_rendering_context(),
            "axis": self.axis,
            "slicePoint": self.slice_point,
            "folder": self.folder_name,
            "scale_image": self.scale_image,
            "scale_to_cellsize": self.scale_to_cellsize,
            "white_box_per_GPU": self.white_box_per_gpu,
            "EM_FIELD_SCALE_CHANNEL1": self.em_field_scale_channel1.value,
            "EM_FIELD_SCALE_CHANNEL2": self.em_field_scale_channel2.value,
            "EM_FIELD_SCALE_CHANNEL3": self.em_field_scale_channel3.value,
            "preParticleDensCol": self.pre_particle_density_color_scales.value,
            "preChannel1Col": self.pre_channel1_color_scales.value,
            "preChannel2Col": self.pre_channel2_color_scales.value,
            "preChannel3Col": self.pre_channel3_color_scales.value,
            "customNormalizationSI": custom_normalization_si_serialized,
            "preParticleDens_opacity": self.pre_particle_density_opacity,
            "preChannel1_opacity": self.pre_channel1_opacity,
            "preChannel2_opacity": self.pre_channel2_opacity,
            "preChannel3_opacity": self.pre_channel3_opacity,
            "preChannel1": self.pre_channel1,
            "preChannel2": self.pre_channel2,
            "preChannel3": self.pre_channel3,
        }
