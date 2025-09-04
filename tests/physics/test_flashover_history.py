from dpf2.physics.flashover import FlashoverModel, FlashoverParameters
from dpf2.synthetic_diagnostics import flashover_jitter_stats


def test_flashover_model_conditioning_history():
    params = FlashoverParameters(field_threshold=10.0, sigma=0.0, conditioning=0.1, seed=1)
    model = FlashoverModel("mather", params)
    d0 = model.sample_delay(5.0)
    d1 = model.sample_delay(5.0)
    assert d1 < d0


def test_flashover_jitter_matches_reference():
    params = FlashoverParameters(field_threshold=10.0, sigma=0.2, conditioning=0.0, seed=123)
    model = FlashoverModel("mather", params)
    series = model.holdoff_series(20)
    stats = flashover_jitter_stats(series)
    assert abs(stats["stddev"] - 2.0) < 0.5
