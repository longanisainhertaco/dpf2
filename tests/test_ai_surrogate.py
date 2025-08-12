import numpy as np
import pytest

from dpf2.ai import SurrogateModel, TorchSurrogateModel, ONNXSurrogateModel


class DummySurrogate(SurrogateModel):
    def __init__(self):
        super().__init__("/tmp/model")

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return inputs * 2


def test_surrogate_base():
    s = DummySurrogate()
    x = np.array([1.0, 2.0])
    np.testing.assert_allclose(s.predict(x), x * 2)


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
