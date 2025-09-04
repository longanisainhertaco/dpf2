import numpy as np
import pytest
from typing import Any

from dpf2.ai import SurrogateModel, TorchSurrogateModel, ONNXSurrogateModel


class SimpleSurrogate(SurrogateModel):
    def __init__(self, model_path: str, scale: float = 1.0) -> None:
        super().__init__(model_path)
        self.scale = scale

    def predict(self, inputs: Any) -> Any:
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
