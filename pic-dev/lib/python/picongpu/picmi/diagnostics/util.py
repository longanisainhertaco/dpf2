"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from picongpu.picmi.copy_attributes import default_converts_to


def diagnostic_converts_to(*args, **kwargs):
    kwargs["conversions"] = {
        "species": lambda self, *args, **kwargs: kwargs["dict_species_picmi_to_pypicongpu"].get(self.species)
    } | kwargs.get("conversions", {})
    return default_converts_to(*args, **kwargs)
