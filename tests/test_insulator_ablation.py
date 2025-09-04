import pytest
from dpf2.device_profiles import DeviceProfiles
from dpf2.ablation import insulator_sleeve_area, ablation_mass_energy_source


def test_device_profile_has_insulator_sleeve():
    cfg = DeviceProfiles.with_defaults()
    sleeve = cfg.devices["PF1000"].insulator_sleeve
    assert sleeve is not None
    assert sleeve.inner_radius_cm == pytest.approx(
        cfg.devices["PF1000"].anode_radius_cm
    )
    assert sleeve.length_cm == pytest.approx(
        cfg.devices["PF1000"].insulator_length_cm
    )
    assert sleeve.material is not None
    assert (
        sleeve.material.material_id
        == cfg.devices["PF1000"].insulator_material.material_id
    )


def test_ablation_mass_energy_source():
    cfg = DeviceProfiles.with_defaults()
    sleeve = cfg.devices["PF1000"].insulator_sleeve
    area = insulator_sleeve_area(sleeve.inner_radius_cm * 1e-2, sleeve.length_cm * 1e-2)
    m_dot, e_dot = ablation_mass_energy_source(1e-5, area, 2e6)
    assert m_dot == pytest.approx(1e-5 * area)
    assert e_dot == pytest.approx(m_dot * 2e6)
