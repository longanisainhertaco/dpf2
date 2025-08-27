import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dpf2.ai import ONNXSurrogateModel, TorchSurrogateModel
from dpf2.ai.training import load_numpy_dataset, train_torch_model
from dpf2.metadata import Metadata


def _train_linear(tmp_path):
    x = np.linspace(-1, 1, 20, dtype=np.float32).reshape(-1, 1)
    y = 3 * x + 1
    loader = load_numpy_dataset(x, y, batch_size=5)
    net = torch.nn.Sequential(torch.nn.Linear(1, 1))
    meta = Metadata.with_defaults()
    train_torch_model(net, loader, epochs=200, lr=0.1, metadata=meta)
    model_path = tmp_path / "linear.pt"
    torch.jit.script(net).save(model_path)
    return x, y, model_path, meta, net


def test_torch_training_inference(tmp_path):
    x, y, path, meta, _ = _train_linear(tmp_path)
    sm = TorchSurrogateModel.load(path)
    preds = sm.predict(x)
    np.testing.assert_allclose(preds, y, atol=1e-1)
    assert meta.ml_metadata is not None
    assert meta.ml_result is not None


def test_onnx_inference(tmp_path):
    pytest.importorskip("onnxruntime")
    x, y, _, _, net = _train_linear(tmp_path)
    onnx_path = tmp_path / "linear.onnx"
    net.eval()
    torch.onnx.export(net, torch.as_tensor(x), onnx_path)
    sm = ONNXSurrogateModel.load(onnx_path)
    preds = sm.predict(x)
    np.testing.assert_allclose(preds, y, atol=1e-1)
