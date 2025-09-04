import pytest

from dpf2.ai import load_yield_surrogate, load_pinch_time_surrogate
from dpf2.exceptions import OutOfDomainError


def test_yield_surrogate_prediction():
    model = load_yield_surrogate()
    # Use a value within training domain
    val = 160.0
    pred = model.predict(val)
    assert pred > 0
    assert model.domain[0] <= val <= model.domain[1]
    assert model.error >= 0


def test_pinch_time_surrogate_ood_error():
    model = load_pinch_time_surrogate()
    # Deliberately exceed training range to trigger OOD error
    with pytest.raises(OutOfDomainError):
        _ = model.predict(1000.0)
