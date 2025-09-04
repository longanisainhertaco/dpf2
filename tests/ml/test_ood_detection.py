import pytest

from dpf2.ai.simple_surrogates import LinearSurrogate
from dpf2.exceptions import OutOfDomainError


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
    with pytest.raises(OutOfDomainError):
        model.predict(10.0)

def test_mahalanobis_threshold_blocks():
    model = _surrogate()
    with pytest.raises(OutOfDomainError):
        model.predict(3.0)
