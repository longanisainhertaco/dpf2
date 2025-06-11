"""Abstractions for machine learning surrogate models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class SurrogateModel(ABC):
    """Base class for ML surrogate models."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)

    @abstractmethod
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Return model prediction for ``inputs``."""
        raise NotImplementedError


class TorchSurrogateModel(SurrogateModel):
    """Surrogate model backed by a PyTorch ``ScriptModule``."""

    def __init__(self, model_path: str | Path, device: str = "cpu") -> None:
        super().__init__(model_path)
        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("PyTorch is required for TorchSurrogateModel") from exc
        self._torch = torch
        self.device = device
        self.model = torch.jit.load(str(self.model_path))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        tensor = self._torch.as_tensor(inputs, device=self.device)
        with self._torch.no_grad():
            out = self.model(tensor).cpu().numpy()
        return out


class ONNXSurrogateModel(SurrogateModel):
    """Surrogate model using ``onnxruntime`` for inference."""

    def __init__(self, model_path: str | Path) -> None:
        super().__init__(model_path)
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("onnxruntime is required for ONNXSurrogateModel") from exc
        self.session = ort.InferenceSession(str(self.model_path))

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: inputs})
        return outputs[0]


__all__ = ["SurrogateModel", "TorchSurrogateModel", "ONNXSurrogateModel"]
