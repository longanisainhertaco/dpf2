from dpf2.breakdown.flashover import (
    FlashoverParameters,
    conditioning_curve,
    seea_stochastic_delay,
)
from dpf2.geometry import (
    triple_junction_field,
    set_triple_junction_field_map,
)
from dpf2.dpf_config import BreakdownModel
from dpf2.synthetic_diagnostics import flashover_delay_stats

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

def test_breakdown_model_exposes_parameters():
    bm = BreakdownModel(type="flashover", seea_sigma=0.2, conditioning_alpha=0.1)
    assert bm.seea_sigma == 0.2
    assert bm.conditioning_alpha == 0.1

def test_flashover_delay_stats():
    stats = flashover_delay_stats([1.0, 3.0, 5.0])
    assert stats["count"] == 3
    assert abs(stats["mean"] - 3.0) < 1e-12
    assert "stddev" in stats and stats["stddev"] > 0
