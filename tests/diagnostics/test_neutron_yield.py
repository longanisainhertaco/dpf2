import math

from dpf2.neutron_yield_model import partition_yield
from dpf2.fusion import dd_yield_components


def test_partition_yield_components():
    thermo = [2.0, 2.0, 2.0]
    beam = [1.0, 1.0, 1.0]
    res = partition_yield(thermo, beam, dt=0.5)
    th, th_u = res["thermonuclear"]
    bt, bt_u = res["beam_target"]
    tot, tot_u = res["total"]
    assert math.isclose(th + bt, tot)
    assert math.isclose(th_u, math.sqrt(th))
    assert math.isclose(bt_u, math.sqrt(bt))
    assert math.isclose(tot_u, math.sqrt(tot))


def test_partition_yield_with_uncertainties():
    thermo = [2.0, 2.0, 2.0]
    beam = [1.0, 1.0, 1.0]
    thermo_sigma = [0.5, 0.5, 0.5]
    beam_sigma = [0.2, 0.2, 0.2]
    res = partition_yield(thermo, beam, dt=0.5, thermonuclear_sigma=thermo_sigma, beam_target_sigma=beam_sigma)
    th, th_u = res["thermonuclear"]
    bt, bt_u = res["beam_target"]
    tot, tot_u = res["total"]
    expected_th_var = th + sum((s * 0.5) ** 2 for s in thermo_sigma)
    expected_bt_var = bt + sum((s * 0.5) ** 2 for s in beam_sigma)
    assert math.isclose(th_u, math.sqrt(expected_th_var))
    assert math.isclose(bt_u, math.sqrt(expected_bt_var))
    assert math.isclose(tot_u, math.sqrt(expected_th_var + expected_bt_var))


def test_dd_yield_components_zero_beam():
    res = dd_yield_components(
        T_keV=10.0,
        n_thermal=1e20,
        n_beam=0.0,
        beam_energy_keV=50.0,
        volume=1.0,
        duration=1.0,
    )
    assert res["beam_target"][0] == 0.0
    assert math.isclose(res["total"][0], res["thermonuclear"][0])
