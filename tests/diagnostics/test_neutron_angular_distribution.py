from dpf2.diagnostics.neutron.angular_distribution import (
    per_angle_yield,
    forward_radial_backward_totals,
    directional_yield,
)


def test_per_angle_yield_and_totals():
    spectra = {0.0: [1.0, 2.0], 90.0: [3.0, 4.0], 180.0: [5.0, 6.0]}
    yields = per_angle_yield(spectra)
    assert yields[0.0] == 3.0
    assert yields[90.0] == 7.0
    assert yields[180.0] == 11.0
    totals = forward_radial_backward_totals(yields)
    assert totals["forward"] == 3.0
    assert totals["radial"] == 7.0
    assert totals["backward"] == 11.0


def test_instrument_response_weighting():
    spectra = {0.0: [1.0, 2.0]}
    responses = {0.0: [0.5, 1.5]}
    yields = per_angle_yield(spectra, responses)
    assert yields[0.0] == 1.0 * 0.5 + 2.0 * 1.5
    totals = directional_yield(spectra, responses)
    assert totals["forward"] == yields[0.0]
    assert totals["radial"] == 0.0
    assert totals["backward"] == 0.0
