from __future__ import annotations

"""Light‑weight representations of distributed circuit components.

The real project contains a sophisticated distributed circuit solver.  For the
purposes of the exercises in this kata we implement only the minimal features
required by the tests.  The goal is to provide simple data classes describing
transmission line segments and switches together with a helper to assemble
system matrices used by the time integrator in :mod:`dpf2.rlc_solver`.

Each :class:`TransmissionLineSegment` stores per–unit length parameters and
optional parasitic components that apply to the whole segment.  Time dependant
profiles can be supplied for the R, L and C values in the form of ``[(t, val)]``
pairs.  The :class:`TriggeredSwitch` models an ideal resistive switch which can
change state at user supplied trigger times and may also have fixed parasitic
components attached.

The :func:`assemble_matrices` function converts a list of segments and switches
into diagonal R, L and C matrices.  The matrices are intentionally extremely
simple – topology is ignored and all elements are assumed to be in series –
which is perfectly adequate for the unit tests that exercise this module.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence, List, Any

import numpy as np
import math
import cmath
import warnings

from dpf2.core.bases import PlasmaSolverBase
from .switches import TriggeredSwitch, CrowbarStage

__all__ = [
    "TransmissionLineSegment",
    "TriggeredSwitch",
    "CrowbarStage",
    "BlumleinSection",
    "MultiSectionLine",
    "PlasmaInductance",
    "assemble_matrices",
]


# ---------------------------------------------------------------------------
# Utility helpers


def _interp_profile(profile: Sequence[tuple[float, float]] | None, t: float) -> float:
    """Return interpolated profile value at time ``t``.

    Profiles are specified as ``[(time, value), ...]`` and are linearly
    interpolated.  If no profile is supplied the return value is ``0.0``.  The
    profile is considered to contain absolute contributions to the quantity in
    question (i.e. they are *added* to the base value).
    """

    if not profile:
        return 0.0
    arr = np.asarray(profile, dtype=float)
    times = arr[:, 0]
    values = arr[:, 1]
    return float(np.interp(t, times, values, left=values[0], right=values[-1]))


# ---------------------------------------------------------------------------
# Component definitions


@dataclass
class TransmissionLineSegment:
    """Simple RLC transmission line segment with optional parasitics.

    Parameters
    ----------
    from_node, to_node:
        Identifiers of the nodes connected by this segment.  The topology is
        not used directly in the tests but is parsed to mirror the behaviour of
        the real application.
    length:
        Physical length of the segment in metres.
    L_per_m, R_per_m, C_per_m:
        Inductance, resistance and capacitance per metre.
    L_parasitic, R_parasitic, C_parasitic:
        Fixed parasitic components attached to the whole segment.
    L_profile, R_profile, C_profile:
        Optional time dependant adjustments (absolute values) for the
        respective quantities.  Each profile is a list of ``(time, value)``
        pairs in SI units.
    """

    from_node: int
    to_node: int
    length: float
    L_per_m: float
    R_per_m: float
    C_per_m: float
    L_parasitic: float = 0.0
    R_parasitic: float = 0.0
    C_parasitic: float = 0.0
    L_profile: Sequence[tuple[float, float]] | None = None
    R_profile: Sequence[tuple[float, float]] | None = None
    C_profile: Sequence[tuple[float, float]] | None = None
    propagation_velocity: float | None = None
    skin_effect_coeff: float = 0.0
    dielectric_loss_coeff: float = 0.0
    material: str | None = None

    def __post_init__(self) -> None:
        """Populate electrical properties from material tables when available."""

        if self.material:
            try:
                from dpf2.materials import get_resistivity, get_skin_effect_coeff

                if self.R_per_m == 0.0:
                    try:
                        self.R_per_m = get_resistivity(self.material)
                    except KeyError:
                        warnings.warn(
                            f"No resistivity data for material '{self.material}', using default {self.R_per_m}",
                            stacklevel=2,
                        )
                if self.skin_effect_coeff == 0.0:
                    try:
                        self.skin_effect_coeff = get_skin_effect_coeff(self.material)
                    except KeyError:
                        warnings.warn(
                            f"No skin-effect data for material '{self.material}', using default {self.skin_effect_coeff}",
                            stacklevel=2,
                        )
            except Exception:
                if self.R_per_m == 0.0:
                    warnings.warn(
                        f"Material tables unavailable; using default resistivity {self.R_per_m} for '{self.material}'",
                        stacklevel=2,
                    )
                if self.skin_effect_coeff == 0.0:
                    warnings.warn(
                        f"Material tables unavailable; using default skin-effect coefficient {self.skin_effect_coeff} for '{self.material}'",
                        stacklevel=2,
                    )

    def delay(self) -> float:
        """Return propagation delay for this segment in seconds."""

        if self.propagation_velocity:
            return self.length / self.propagation_velocity
        return 0.0

    def totals(
        self, t: float = 0.0, frequency: float | None = None
    ) -> tuple[float, float, float | complex]:
        """Return the total ``(L, R, C)`` for this segment.

        ``frequency`` can be provided to account for frequency dependant skin
        effect resistance and dielectric losses.  When ``frequency`` is given the
        returned capacitance may be complex to model dielectric loss via a loss
        tangent.  The simplistic models here are sufficient for the unit tests
        and mirror the behaviour of the real application only qualitatively.
        """

        L = (
            self.L_per_m * self.length
            + self.L_parasitic
            + _interp_profile(self.L_profile, t)
        )
        R = (
            self.R_per_m * self.length
            + self.R_parasitic
            + _interp_profile(self.R_profile, t)
        )
        if frequency is not None and self.skin_effect_coeff:
            R += self.skin_effect_coeff * self.length * float(math.sqrt(frequency))
        C = (
            self.C_per_m * self.length
            + self.C_parasitic
            + _interp_profile(self.C_profile, t)
        )
        if frequency is not None and self.dielectric_loss_coeff:
            loss_tan = self.dielectric_loss_coeff * float(math.sqrt(frequency))
            C = complex(C) * (1.0 - 1j * loss_tan)
        return L, R, C

    # ------------------------------------------------------------------
    # Frequency domain helpers

    def _params_at_freq(self, frequency: float) -> tuple[float, float, float, float]:
        """Return per-unit ``(R, L, C, G)`` accounting for dispersion models."""

        w = 2.0 * np.pi * frequency
        R = self.R_per_m
        if self.skin_effect_coeff:
            R += self.skin_effect_coeff * float(math.sqrt(frequency))
        L = self.L_per_m
        C = self.C_per_m
        G = 0.0
        if self.dielectric_loss_coeff:
            loss_tan = self.dielectric_loss_coeff * float(math.sqrt(frequency))
            G = w * C * loss_tan
        return R, L, C, G

    def propagation_constant(self, frequency: float) -> complex:
        """Return the complex propagation constant ``gamma`` at ``frequency``."""

        w = 2.0 * np.pi * frequency
        R, L, C, G = self._params_at_freq(frequency)
        return cmath.sqrt((R + 1j * w * L) * (G + 1j * w * C))

    def characteristic_impedance(self, frequency: float) -> complex:
        """Return the characteristic impedance at ``frequency``."""

        w = 2.0 * np.pi * frequency
        R, L, C, G = self._params_at_freq(frequency)
        return cmath.sqrt((R + 1j * w * L) / (G + 1j * w * C))

    def reflection_coefficient(
        self, frequency: float, Z_load: float | complex | None
    ) -> complex:
        """Return the reflection coefficient for a load ``Z_load``.

        ``Z_load`` may be ``None`` or ``np.inf`` to model an open circuit.
        """

        Z0 = self.characteristic_impedance(frequency)
        if Z_load is None or Z_load == np.inf:
            return 1.0 + 0.0j
        ZL = complex(Z_load)
        return (ZL - Z0) / (ZL + Z0)


@dataclass
class BlumleinSection:
    """Transmission line segment with a triggerable switch representing a Blumlein block."""

    segment: TransmissionLineSegment
    trigger: TriggeredSwitch

    def __init__(
        self,
        segment: TransmissionLineSegment,
        trigger_time: float,
        jitter_std: float = 0.0,
        R_on: float = 1e-3,
        R_off: float = 1e6,
    ) -> None:
        self.segment = segment
        self.trigger = TriggeredSwitch(
            from_node=segment.to_node,
            to_node=segment.from_node,
            closed=False,
            R_on=R_on,
            R_off=R_off,
            trigger_times=[trigger_time],
            jitter_std=jitter_std,
        )

    def components(self) -> tuple[TransmissionLineSegment, TriggeredSwitch]:
        """Return the underlying segment and trigger switch."""

        return self.segment, self.trigger


@dataclass
class MultiSectionLine:
    """Container representing a chain of transmission line segments.

    Realistic drivers are often approximated by several lumped sections.  The
    solver itself operates purely on individual ``TransmissionLineSegment``
    objects; this helper merely groups a number of segments so they can be
    passed around as a single object before being expanded by
    :func:`assemble_matrices`.
    """

    sections: Sequence[TransmissionLineSegment]

    def components(self) -> list[TransmissionLineSegment]:
        """Return the contained segments as a list."""

        return list(self.sections)


@dataclass
class PlasmaInductance:
    """Dynamic inductive branch sourced from an external plasma solver."""

    from_node: int
    to_node: int
    solver: PlasmaSolverBase

    def delay(self) -> float:
        return 0.0

    def totals(
        self, t: float = 0.0, frequency: float | None = None
    ) -> tuple[float, float, float]:
        """Return instantaneous plasma inductance ``Lp``."""

        try:
            fb = self.solver.coupling_interface()
            Lp = float(getattr(fb, "Lp", 0.0))
        except Exception:
            Lp = 0.0
        return Lp, 0.0, 0.0


# ---------------------------------------------------------------------------
# Matrix assembly


def assemble_matrices(
    segments: Sequence[Any],
    switches: Sequence[TriggeredSwitch] | None = None,
    t: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble node based ``R``, ``L`` and ``C`` matrices for a network.

    The original implementation used simple diagonal matrices assuming all
    elements were connected in series.  For the extended exercises we need to be
    able to handle arbitrary connectivity, branched networks and multiple
    switches.  The matrices returned by this function therefore follow a very
    small nodal analysis scheme: each two–terminal component stamps its value
    into the matrices for the nodes it connects.  Kirchhoff's current and
    voltage laws are implicitly satisfied by adding the value to both node
    diagonals and subtracting it from the off–diagonal entries.

    The row/column order of the matrices corresponds to the sorted list of all
    node identifiers appearing in ``segments`` and ``switches``.  Components with
    zero values simply have no effect on the matrices which keeps the
    implementation concise while remaining perfectly adequate for the unit
    tests.
    """

    # Expand composite helpers into raw segments and switches
    seg_list: list[TransmissionLineSegment] = []
    sw_list: list[TriggeredSwitch] = list(switches or [])
    for item in segments:
        if isinstance(item, BlumleinSection):
            seg, trig = item.components()
            seg_list.append(seg)
            sw_list.append(trig)
        elif isinstance(item, MultiSectionLine):
            seg_list.extend(item.components())
        else:
            seg_list.append(item)  # type: ignore[arg-type]

    segments = seg_list
    switches = sw_list

    # Determine mapping from node identifiers to matrix indices
    nodes = set()
    for seg in segments:
        nodes.add(seg.from_node)
        nodes.add(seg.to_node)
    for sw in switches:
        nodes.add(sw.from_node)
        nodes.add(sw.to_node)

    if not nodes:
        return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0))

    node_list = sorted(nodes)
    idx = {node: i for i, node in enumerate(node_list)}
    n = len(node_list)

    R = np.zeros((n, n))
    L = np.zeros((n, n))
    C = np.zeros((n, n))

    def _stamp(i: int, j: int, value: float, mat: np.ndarray) -> None:
        """Stamp a two‑terminal component value into ``mat``."""

        if value == 0.0:
            return
        mat[i, i] += value
        mat[j, j] += value
        mat[i, j] -= value
        mat[j, i] -= value

    # Stamp transmission line segments
    for seg in segments:
        i, j = idx[seg.from_node], idx[seg.to_node]
        L_val, R_val, C_val = seg.totals(t)
        _stamp(i, j, R_val, R)
        _stamp(i, j, L_val, L)
        _stamp(i, j, C_val, C)

    # Stamp switches (including parasitics)
    for sw in switches:
        i, j = idx[sw.from_node], idx[sw.to_node]
        _stamp(i, j, sw.resistance(t), R)
        _stamp(i, j, sw.L_parasitic, L)
        _stamp(i, j, sw.C_parasitic, C)

    return R, L, C
