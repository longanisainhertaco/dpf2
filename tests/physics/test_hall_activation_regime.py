import numpy as np
import pytest

from dpf2.physics.hall_mhd import HallMHD, TwoFluidHallMHD, hall_parameters
from dpf2.physics.lower_hybrid_drift import LowerHybridDrift
from dpf2.physics.anomalous_resistivity import SpectralResistivity


def test_hall_activation_requires_both_regime_metrics():
    model = HallMHD(hall_coeff=0.5, omega_ce_tau_e_min=1.0, di_over_L_min=0.05)
    ne, Te, B = 1.0e18, 200.0, 1.0
    L_active = 0.05

    omega_tau, di_over_L = hall_parameters(ne, Te, B, L_active)
    assert omega_tau > model.omega_ce_tau_e_min
    assert di_over_L > model.di_over_L_min

    model.update_transport(ne, Te, B, L_active)
    metrics = model.regime_metrics(ne, Te, B, L_active)

    assert model.hall_active
    assert metrics["omega_ce_tau_e"] == pytest.approx(omega_tau)
    assert metrics["di_over_L"] == pytest.approx(di_over_L)

    # Stretch the scale length to suppress the Hall gate while keeping magnetisation high
    model.update_transport(ne, Te, B, 5.0)
    assert not model.hall_active


def test_anomalous_resistivity_impedance_exposed():
    lhd = LowerHybridDrift(B=1.0, n_i=1e18, amplitude=0.3)
    spectral = SpectralResistivity(lhd, scale=0.2, floor=0.05)
    model = TwoFluidHallMHD(
        hall_coeff=0.1,
        lhdi_resistivity=spectral,
        omega_ce_tau_e_min=0.5,
    )

    rho = 1.0e-6
    n = rho / model.ion_mass
    Te = 200.0
    p = n * 1.380649e-23 * Te
    prim = np.array([rho, 0.0, 0.0, 0.0, p, 20.0, 0.0, 0.0])
    U = model.conservative_variables(prim)

    model.update_transport(n, Te, 20.0, 0.05)
    model.step(U, dt=1.0e-9, current=1.0, instability_amp=np.array([1.0, 0.0, 0.0]))
    metrics = model.step_twofluid(U, dt=1.0e-9, L=0.05)

    assert model.plasma_impedance > 0.0
    assert metrics["effective_impedance"] >= model.eta
    assert metrics["hall_active"]
    assert metrics["omega_ce_tau_e"] >= model.omega_ce_tau_e_min
