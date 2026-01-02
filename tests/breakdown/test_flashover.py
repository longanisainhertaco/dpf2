from dpf2.breakdown.flashover import (
    FlashoverParameters,
    conditioning_curve,
    seea_stochastic_delay,
    holdoff_voltage,
    holdoff_series,
    vacuum_surface_flashover,
    FlashoverSwitchCoupler,
)
from dpf2.geometry import (
    triple_junction_field,
    set_triple_junction_field_map,
    triple_junction_enhancement,
)
from dpf2.dpf_config import BreakdownModel
from dpf2.synthetic_diagnostics import flashover_delay_stats, flashover_jitter_stats
from dpf2.circuit.switches import TriggeredSwitch
import pytest

def test_conditioning_curve_monotonic():
    vals = [conditioning_curve(i, 0.1) for i in range(5)]
    assert vals[0] == 1.0
    assert all(v2 <= v1 for v1, v2 in zip(vals, vals[1:]))

def test_stochastic_delay_reproducible_and_conditioned():
    params = FlashoverParameters(field_threshold=10.0, sigma=0.1, conditioning=0.2, seed=123)
    d1 = seea_stochastic_delay(5.0, params, shot=0)
    d2 = seea_stochastic_delay(5.0, params, shot=0)
    assert d1 == d2
    d_conditioned = seea_stochastic_delay(5.0, params, shot=5)
    assert d_conditioned < d1

def test_triple_junction_field_map():
    base = triple_junction_field("unknown")
    set_triple_junction_field_map("custom", 42.0)
    assert triple_junction_field("custom") == 42.0
    assert base == 1.0


def test_triple_junction_enhancement_geometry_ratio():
    base = triple_junction_field("mather")
    enhanced = triple_junction_enhancement("mather", anode_radius=1.0, cathode_radius=2.0)
    assert enhanced > base

def test_breakdown_model_exposes_parameters():
    bm = BreakdownModel(type="flashover", seea_sigma=0.2, conditioning_alpha=0.1)
    assert bm.seea_sigma == 0.2
    assert bm.conditioning_alpha == 0.1

def test_flashover_delay_stats():
    stats = flashover_delay_stats([1.0, 3.0, 5.0])
    assert stats["count"] == 3
    assert abs(stats["mean"] - 3.0) < 1e-12
    assert "stddev" in stats and stats["stddev"] > 0


def test_holdoff_voltage_evolves_and_geometry_factor():
    params = FlashoverParameters(field_threshold=10.0, sigma=0.0, conditioning=0.1, seed=123)
    h0 = holdoff_voltage("mather", params, shot=0)
    h5 = holdoff_voltage("mather", params, shot=5)
    h_geom = holdoff_voltage("tapered", params, shot=0)
    assert h5 > h0
    assert h_geom > h0


def test_flashover_jitter_stats():
    params = FlashoverParameters(field_threshold=10.0, sigma=0.2, conditioning=0.0, seed=1)
    series = holdoff_series("mather", params, shots=5)
    stats = flashover_jitter_stats(series)
    assert stats["count"] == 5
    assert stats["stddev"] > 0


def test_flashover_switch_coupling_with_triple_point_field():
    params = FlashoverParameters(field_threshold=5.0, sigma=0.0, conditioning=0.0, seed=42)
    switch = TriggeredSwitch(from_node=0, to_node=1, closed=False, trigger_times=[])
    coupler = FlashoverSwitchCoupler(
        geometry="tapered",
        params=params,
        switch=switch,
        anode_radius_cm=1.0,
        cathode_radius_cm=3.0,
    )
    result = coupler.schedule(field=10.0, t0=1e-6)
    assert pytest.approx(switch.trigger_times[0]) == result.switch_trigger_time
    # ensure triple junction scaling is applied
    assert result.triple_junction_factor > 1.0
    switch.update(result.switch_trigger_time or 0.0)
    assert switch.closed is True
