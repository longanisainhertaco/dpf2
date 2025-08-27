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

    # ------------------------------------------------------------------
    # Optional lifecycle helpers
    def train(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        """Train the surrogate model.

        Parameters and return values are model-specific. Sub-classes may
        override this method to provide training capabilities. The default
        implementation raises :class:`NotImplementedError`.
        """

        raise NotImplementedError("Training not implemented for this model")

    def save(self, path: str | Path | None = None) -> Path:
        """Persist the model to ``path``.

        Parameters
        ----------
        path:
            Destination for the serialized model. Implementations may fall
            back to ``self.model_path`` when ``path`` is ``None``.
        """

        raise NotImplementedError("Save not implemented for this model")

    @classmethod
    def load(cls, model_path: str | Path, *args: Any, **kwargs: Any) -> "SurrogateModel":
        """Load a serialized model from ``model_path``.

        Sub-classes should override this to return an initialized instance.
        """

        raise NotImplementedError("Load not implemented for this model")


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

    # ------------------------------------------------------------------
    def train(
        self,
        dataloader: Any,
        epochs: int = 10,
        lr: float = 1e-3,
        metadata: Any | None = None,
    ) -> dict[str, float]:
        """Train the underlying ``torch.nn.Module``.

        Parameters
        ----------
        dataloader:
            Iterable yielding ``(inputs, targets)`` tensors.
        epochs:
            Number of passes through the dataset.
        lr:
            Learning rate for the Adam optimizer.
        metadata:
            Optional :class:`~dpf2.metadata.Metadata` instance updated with
            ``ml_metadata`` and ``ml_result``.
        """

        try:
            from .training import train_torch_model
        except Exception as exc:  # pragma: no cover - optional
            raise ImportError("Training utilities require PyTorch") from exc

        self.model, metrics = train_torch_model(
            self.model, dataloader, epochs=epochs, lr=lr, metadata=metadata
        )
        return metrics

    def save(self, path: str | Path | None = None) -> Path:
        dest = Path(path or self.model_path)
        scripted = self._torch.jit.script(self.model)
        scripted.save(str(dest))
        return dest

    @classmethod
    def load(cls, model_path: str | Path, device: str = "cpu") -> "TorchSurrogateModel":
        return cls(model_path, device=device)


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

    # ------------------------------------------------------------------
    def train(self, *args: Any, **kwargs: Any) -> dict[str, float]:  # pragma: no cover - thin wrapper
        """ONNX models are inference-only and cannot be trained."""

        raise NotImplementedError("ONNX models do not support in-place training")

    def save(self, path: str | Path | None = None) -> Path:
        dest = Path(path or self.model_path)
        if dest != self.model_path:
            import shutil

            shutil.copy(self.model_path, dest)
        return dest

    @classmethod
    def load(cls, model_path: str | Path) -> "ONNXSurrogateModel":
        return cls(model_path)


__all__ = ["SurrogateModel", "TorchSurrogateModel", "ONNXSurrogateModel"]
