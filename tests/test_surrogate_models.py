import warnings

from dpf2.ai import load_yield_surrogate, load_pinch_time_surrogate


def test_yield_surrogate_prediction():
    model = load_yield_surrogate()
    # Use a value within training domain
    val = 160.0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pred = model.predict(val)
    assert not w, "no warning expected inside training domain"
    assert pred > 0
    assert model.domain[0] <= val <= model.domain[1]
    assert model.error >= 0


def test_pinch_time_surrogate_warning():
    model = load_pinch_time_surrogate()
    # Deliberately exceed training range to trigger warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = model.predict(1000.0)
    assert any("outside training range" in str(item.message) for item in w)
