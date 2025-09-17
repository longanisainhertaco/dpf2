import math

import math

from dpf2.neutron_yield_model import partition_yield, directional_counts
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
    res = partition_yield(
        thermo,
        beam,
        dt=0.5,
        thermonuclear_sigma=thermo_sigma,
        beam_target_sigma=beam_sigma,
    )
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


def test_directional_counts_components():
    fwd = [1.0, 1.0]
    rad = [2.0, 2.0]
    back = [3.0, 3.0]
    res = directional_counts(fwd, rad, back, dt=1.0)
    assert math.isclose(res["forward"][0], 2.0)
    assert math.isclose(res["radial"][0], 4.0)
    assert math.isclose(res["backward"][0], 6.0)
    assert math.isclose(res["total"][0], 12.0)
    assert math.isclose(res["forward"][1], math.sqrt(2.0))


def test_directional_counts_uncertainty():
    fwd = [1.0, 1.0]
    rad = [2.0, 2.0]
    back = [3.0, 3.0]
    fwd_s = [0.1, 0.1]
    rad_s = [0.2, 0.2]
    back_s = [0.3, 0.3]
    res = directional_counts(
        fwd,
        rad,
        back,
        dt=1.0,
        forward_sigma=fwd_s,
        radial_sigma=rad_s,
        backward_sigma=back_s,
    )

    def _expected(count, sigs):
        return math.sqrt(count + sum(s**2 for s in sigs))

    assert math.isclose(res["forward"][1], _expected(2.0, fwd_s))
    assert math.isclose(res["radial"][1], _expected(4.0, rad_s))
    assert math.isclose(res["backward"][1], _expected(6.0, back_s))
