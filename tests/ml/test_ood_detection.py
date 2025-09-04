import warnings

import pytest

from dpf2.ai.simple_surrogates import LinearSurrogate
from dpf2.optimization import OptimizationWarning, enable_optimization_warning_as_error


def _surrogate():
    # Simple model y = x with unit variance around mean 0
    return LinearSurrogate(
        coeffs=(1.0, 0.0),
        domain=(-5.0, 5.0),
        error=0.5,
        mean=0.0,
        covariance=1.0,
        ood_threshold=2.0,
    )


def test_in_distribution_prediction_has_band():
    model = _surrogate()
    pred, (lo, hi) = model.predict_with_uncertainty(1.0)
    assert pytest.approx(pred) == 1.0
    assert lo < pred < hi
    # distance = 1 -> band scaled by (1 + 1)
    assert pytest.approx(hi - pred, rel=1e-6) == model.error * (1 + model._mahalanobis(1.0))


def test_out_of_distribution_warning_raised():
    model = _surrogate()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model.predict(10.0)
    assert any(issubclass(wi.category, OptimizationWarning) for wi in w)


def test_enable_warning_as_error_blocks():
    model = _surrogate()
    with warnings.catch_warnings():
        enable_optimization_warning_as_error()
        with pytest.raises(OptimizationWarning):
            model.predict(10.0)
