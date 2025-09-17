"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import math
import typing

import typeguard

from ...pypicongpu import laser
from ..copy_attributes import default_converts_to
from .base_laser import BaseLaser
from .polarization_type import PolarizationType


@default_converts_to(laser.DispersivePulseLaser)
@typeguard.typechecked
class DispersivePulseLaser(BaseLaser):
    """
    Specifies a dispersive Gaussian pulse with dispersion parameters

    Parameters
    ----------
    wavelength: float
        Laser wavelength [m], defined as :math:`\\lambda_0` in the above formula

    duration: float
        Duration of the Gaussian pulse [s], defined as :math:`\\tau` in the above formula

    propagation_direction: unit vector of length 3 of floats
        Direction of propagation [1]

    polarization_direction: unit vector of length 3 of floats
        Direction of polarization [1]

    focal_position: vector of length 3 of floats
        Position of the laser focus [m]

    centroid_position: vector of length 3 of floats
        Position of the laser centroid at time 0 [m]

    a0: float
        Normalized vector potential at focus
        Specify either a0 or E0 (E0 takes precedence).

    E0: float
        Maximum amplitude of the laser field [V/m]
        Specify either a0 or E0 (E0 takes precedence).

    phi0: float
        Carrier envelope phase (CEP) [rad]

    spectral_support: float
        Width of the spectral support for the discrete Fourier transform [none]

    sd_si: float
        Spatial dispersion in focus [m*s]

    ad_si: float
        Angular dispersion in focus [rad*s]

    gdd_si: float
        Group velocity dispersion in focus [s^2]

    tod_si: float
        Third order dispersion in focus [s^3]
    """

    def __init__(
        self,
        waist,
        wavelength,
        duration,
        propagation_direction,
        polarization_direction,
        focal_position,
        centroid_position,
        a0=None,
        E0=None,
        phi0: float = 0.0,
        picongpu_polarization_type=(PolarizationType.LINEAR),
        picongpu_spectral_support: float = 6.0,
        picongpu_sd_si: float = 0.0,
        picongpu_ad_si: float = 0.0,
        picongpu_gdd_si: float = 0.0,
        picongpu_tod_si: float = 0.0,
        # make sure to always place Huygens-surface inside PML-boundaries,
        # default is valid for standard PMLs
        # @todo create check for insufficient dimension
        # @todo create check in simulation for conflict between PMLs and
        # Huygens-surfaces
        picongpu_huygens_surface_positions: typing.List[typing.List[int]] = [
            [16, -16],
            [16, -16],
            [16, -16],
        ],
    ):
        if waist <= 0:
            raise ValueError(f"waist must be > 0. You gave {waist=}.")
        if wavelength <= 0:
            raise ValueError(f"wavelength must be > 0. You gave {wavelength=}.")
        if duration <= 0:
            raise ValueError(f"laser pulse duration must be > 0. You gave {duration=}.")

        self.waist = waist
        self.wavelength = wavelength
        self.k0 = 2.0 * math.pi / wavelength
        self.duration = duration
        self.focal_position = focal_position
        self.centroid_position = centroid_position
        self.propagation_direction = propagation_direction
        self.polarization_direction = polarization_direction
        self.a0, self.E0 = self._compute_E0_and_a0(self.k0, E0, a0)
        self.phi0 = phi0

        self.picongpu_polarization_type = picongpu_polarization_type
        self.picongpu_spectral_support = picongpu_spectral_support
        self.picongpu_sd_si = picongpu_sd_si
        self.picongpu_ad_si = picongpu_ad_si
        self.picongpu_gdd_si = picongpu_gdd_si
        self.picongpu_tod_si = picongpu_tod_si
        self.picongpu_huygens_surface_positions = picongpu_huygens_surface_positions
        self._validate_common_properties()
        self.pulse_init = self._compute_pulse_init()

    def check(self):
        self._validate_common_properties()
        assert self._propagation_connects_centroid_and_focus(), (
            "propagation_direction must connect centroid_position and focus_position"
        )
