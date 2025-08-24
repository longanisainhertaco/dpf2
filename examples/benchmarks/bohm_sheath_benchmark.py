"""Benchmark for Bohm sheath formation.

This script compares the electric field and velocity imposed by the
:class:`dpf2.simulation.sheath_model.BohmSheath` class against the analytic
Bohm sheath relations.  The expected sheath potential drop is

.. math:: \phi_s = T_e \ln \sqrt{m_i / (2\pi m_e)}.

The ion flow into the sheath is expected to reach the Bohm velocity

.. math:: v_B = \sqrt{e T_e / m_i}.

References
----------
* D. Bohm, ``The Characteristics of Electrical Discharges in Magnetic
  Fields``, McGraw-Hill (1949).
"""

from __future__ import annotations

import numpy as np

from dpf2.simulation.sheath_model import BohmSheath, e_charge, m_e
from dpf2.simulation.utils import FieldManager, SimulationState


def _make_state() -> tuple[SimulationState, FieldManager]:
    """Create a minimal simulation state for the benchmark."""
    fm = FieldManager(
        grid_shape=(4, 4, 4),
        dx=1.0,
        dy=1.0,
        dz=1.0,
        domain_lo=(0.0, 0.0, 0.0),
        boundary_conditions={
            "x_lo": "periodic",
            "x_hi": "periodic",
            "y_lo": "periodic",
            "y_hi": "periodic",
            "z_lo": "periodic",
            "z_hi": "periodic",
        },
    )
    state = SimulationState(
        grid_shape=(4, 4, 4),
        dx=1.0,
        dy=1.0,
        dz=1.0,
        domain_lo=(0.0, 0.0, 0.0),
        boundary_conditions={},
        field_manager=fm,
    )
    return state, fm


def run_benchmark() -> dict[str, float]:
    """Run the Bohm sheath benchmark and return diagnostic errors."""
    state, fm = _make_state()
    sheath = BohmSheath(electron_temperature=5.0, ion_mass=1.67e-27)
    sheath.apply(state)

    mass_ratio = 1.67e-27 / (2 * np.pi * m_e)
    phi_s = 5.0 * np.log(np.sqrt(mass_ratio))
    v_bohm = np.sqrt(e_charge * 5.0 / 1.67e-27)
    expected_field = phi_s / fm.dz

    field_error = float(np.max(np.abs(fm.get_E()[2, :, :, -1] - expected_field)))
    velocity_error = float(np.max(np.abs(state.velocity[2, :, :, -1] - v_bohm)))

    return {"field_error": field_error, "velocity_error": velocity_error}


if __name__ == "__main__":
    errors = run_benchmark()
    print("Bohm sheath benchmark errors:")
    for key, val in errors.items():
        print(f"  {key}: {val:.3e}")
