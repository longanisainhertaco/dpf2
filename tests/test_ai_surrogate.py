import json
import numpy as np
import pytest
from typing import Any

from dpf2.ai import (
    OutOfDomainError,
    SurrogateModel,
    TorchSurrogateModel,
    ONNXSurrogateModel,
)


class SimpleSurrogate(SurrogateModel):
    def __init__(self, model_path: str, scale: float = 1.0) -> None:
        super().__init__(model_path)
        self.scale = scale

    def _predict(self, inputs: Any) -> Any:
        return inputs * self.scale


class TrainableSurrogate(SimpleSurrogate):
    def _train(self, factor: float = 1.0) -> dict[str, float]:
        self.scale *= factor
        return {"scale": self.scale}


def test_surrogate_round_trip(tmp_path):
    x = np.array([1.0, 2.0])
    model = TrainableSurrogate(tmp_path / "model.pkl", scale=2.0)
    model.train(3.0)
    path = model.save()
    loaded = TrainableSurrogate.load(path)
    np.testing.assert_allclose(loaded.predict(x), x * 6.0)


def test_training_optional(tmp_path):
    model = SimpleSurrogate(tmp_path / "model.pkl", scale=5.0)
    metrics = model.train()
    assert metrics == {}
    assert model.scale == 5.0


def test_out_of_domain(tmp_path):
    model_path = tmp_path / "model.pkl"
    meta = {
        "feature_mean": [0.0, 0.0],
        "feature_cov": [[1.0, 0.0], [0.0, 1.0]],
        "feature_cov_inv": [[1.0, 0.0], [0.0, 1.0]],
        "mahalanobis_threshold": 1.0,
    }
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    model = SimpleSurrogate(model_path, scale=1.0)
    inside = np.array([[0.5, 0.5]])
    outside = np.array([[2.0, 2.0]])
    np.testing.assert_allclose(model.predict(inside), inside)
    with pytest.raises(OutOfDomainError):
        model.predict(outside)


def test_predict_with_uncertainty(tmp_path):
    model_path = tmp_path / "model.pkl"
    meta = {
        "feature_mean": [0.0],
        "feature_cov": [[1.0]],
        "mahalanobis_threshold": 5.0,
        "toy": {"onnx": model_path.name, "quantile": 0.5},
    }
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    model = SimpleSurrogate(model_path, scale=1.0)
    pred, (lo, hi) = model.predict_with_uncertainty(1.0)
    assert pred == pytest.approx(1.0)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(2.0)


def test_torch_surrogate_import_error(tmp_path):
    path = tmp_path / "model.pt"
    path.write_text("dummy")
    with pytest.raises(ImportError):
        TorchSurrogateModel(path)


def test_onnx_surrogate_import_error(tmp_path):
    path = tmp_path / "model.onnx"
    path.write_text("dummy")
    with pytest.raises(ImportError):
        ONNXSurrogateModel(path)
