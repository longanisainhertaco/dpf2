import json
from pathlib import Path

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dpf2.ai import SurrogateModel, OutOfDomainError


class _SimpleSurrogate(SurrogateModel):
    def __init__(self, model_path: Path, scale: float = 1.0) -> None:
        super().__init__(model_path)
        self.scale = scale

    def _predict(self, inputs: Any) -> Any:  # type: ignore[override]
        return inputs * self.scale


def _make_model(tmp_path: Path, threshold: float = 5.0) -> _SimpleSurrogate:
    model_path = tmp_path / "model.pkl"
    meta = {
        "feature_mean": [0.0],
        "feature_cov": [[1.0]],
        "mahalanobis_threshold": threshold,
        "toy": {"onnx": model_path.name, "quantile": 0.5},
    }
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    return _SimpleSurrogate(model_path)


def test_predict_with_guardrails_interval(tmp_path: Path) -> None:
    model = _make_model(tmp_path)
    pred, (lo, hi), dist = model.predict_with_guardrails(1.0)
    assert pred == pytest.approx(1.0)
    assert dist == pytest.approx(1.0)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(2.0)


def test_predict_with_guardrails_ood(tmp_path: Path) -> None:
    model = _make_model(tmp_path, threshold=0.5)
    with pytest.raises(OutOfDomainError):
        model.predict_with_guardrails(1.0)
