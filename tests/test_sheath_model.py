import numpy as np
from dpf2.simulation.sheath_model import PlasmaSheathFormation, e_charge, m_e
from dpf2.simulation.config_schema import SheathConfig


def maxwellian_distribution(v, Te):
    return np.exp(-m_e * v**2 / (2 * e_charge * Te))


def make_config(**kwargs):
    base = dict(
        ion_density=1e18,
        electron_density=1e18,
        sheath_voltage=0.0,
        ion_temperature=0.0,
        electron_temperature=10.0,
        ion_mass=1.67e-27,
        dx=1e-5,
        max_sheath_thickness=1e-3,
        num_grid_points=50,
        plasma_edge_potential=0.0,
    )
    base.update(kwargs)
    return SheathConfig(**base)


def test_non_maxwellian_flux_matches_maxwellian():
    cfg = make_config(
        electron_distribution="analytic",
        electron_distribution_params={
            "distribution_fn": lambda v: maxwellian_distribution(v, 10.0),
            "v_max": 5e7,
            "num_points": 5000,
        },
    )
    sheath = PlasmaSheathFormation(cfg)
    flux = sheath.compute_electron_flux()
    v_th = np.sqrt(8 * e_charge * cfg.electron_temperature / (np.pi * m_e))
    expected = 0.25 * cfg.electron_density * v_th
    assert np.isclose(flux, expected, rtol=1e-2)


def test_analytic_sheath_drop_reduces_to_standard():
    cfg = make_config(secondary_emission_coefficient=0.0)
    sheath = PlasmaSheathFormation(cfg)
    drop = sheath.analytic_sheath_drop()
    c_s = np.sqrt(e_charge * (cfg.electron_temperature + cfg.ion_temperature) / cfg.ion_mass)
    v_th = np.sqrt(8 * e_charge * cfg.electron_temperature / (np.pi * m_e))
    expected = cfg.electron_temperature * np.log(4 * c_s / v_th)
    assert np.isclose(drop, expected, rtol=1e-12)
