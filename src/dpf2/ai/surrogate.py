"""Abstractions for machine learning surrogate models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import json
import numpy as np
import pickle

from ..exceptions import OutOfDomainError


class SurrogateModel(ABC):
    """Base class for ML surrogate models.

    The class provides a minimal lifecycle consisting of ``train`` -> ``save``
    -> ``load``.  Sub-classes can override the underscored hook methods
    (``_train``, ``_save`` and ``_load``) to customise behaviour for a specific
    framework.  The public methods offer a working default implementation based
    on Python ``pickle`` serialization so that even trivial models can round
    trip through the lifecycle without additional code.
    """

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        metadata_path = self.model_path.with_name("metadata.json")
        self._feature_mean: list[float] | None = None
        self._inv_cov: list[list[float]] | None = None
        self._mahalanobis_threshold: float | None = None
        self._quantile: float | None = None
        if metadata_path.exists():
            with metadata_path.open() as fh:
                metadata = json.load(fh)
            mean = metadata.get("feature_mean")
            cov_inv = metadata.get("feature_cov_inv")
            cov = metadata.get("feature_cov")
            threshold = metadata.get("mahalanobis_threshold")
            # Extract per-model quantile if present
            for info in metadata.values():
                if isinstance(info, dict) and info.get("onnx") == self.model_path.name:
                    q = info.get("quantile")
                    if q is not None:
                        self._quantile = float(q)
                        break
            if mean is not None and threshold is not None:
                self._feature_mean = list(mean)
                inv = None
                if cov_inv is not None:
                    inv = [list(row) for row in cov_inv]
                elif cov is not None:
                    try:
                        import numpy as _np  # type: ignore

                        inv = _np.linalg.inv(_np.asarray(cov)).tolist()
                    except Exception:
                        if (
                            isinstance(cov, list)
                            and len(cov) == 1
                            and isinstance(cov[0], list)
                            and len(cov[0]) == 1
                            and cov[0][0] != 0
                        ):
                            inv = [[1.0 / float(cov[0][0])]]
                        else:
                            inv = None
                if inv is not None:
                    self._inv_cov = inv
                    self._mahalanobis_threshold = float(threshold)
                else:
                    self._feature_mean = None

    @abstractmethod
    def _predict(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Return model prediction for ``inputs`` with domain validation."""
        inputs_iter = [inputs] if getattr(inputs, "ndim", 1) == 1 else inputs
        self._check_domain(inputs_iter)
        return self._predict(inputs)

    def predict_with_guardrails(
        self, inputs: Any
    ) -> (
        tuple[float, tuple[float, float], float]
        | list[tuple[float, tuple[float, float], float]]
    ):
        """Return prediction, uncertainty band and OOD distance for ``inputs``."""

        if isinstance(inputs, (list, tuple)):
            arr = np.array([[float(v)] for v in inputs])
            preds = self.predict(arr)
            preds_list = [
                float(p[0]) if hasattr(p, "__getitem__") else float(p) for p in preds
            ]
            dists = [self._mahalanobis_distance([float(v)]) for v in inputs]
            bands = [
                0.0 if self._quantile is None else self._quantile * (1.0 + d)
                for d in dists
            ]
            return [(p, (p - b, p + b), d) for p, b, d in zip(preds_list, bands, dists)]
        else:
            val = float(inputs)
            arr = np.array([[val]])
            pred = float(self.predict(arr)[0][0])
            dist = self._mahalanobis_distance([val])
            band = 0.0 if self._quantile is None else self._quantile * (1.0 + dist)
            return pred, (pred - band, pred + band), dist

    def predict_with_uncertainty(
        self, inputs: Any
    ) -> tuple[float, tuple[float, float]] | list[tuple[float, tuple[float, float]]]:
        """Return prediction and conformal uncertainty band for ``inputs``."""

        res = self.predict_with_guardrails(inputs)
        if isinstance(res, list):
            return [(p, band) for p, band, _ in res]
        pred, band, _ = res
        return pred, band

    def _mahalanobis_distance(self, x: Any) -> float:
        if self._feature_mean is None or self._inv_cov is None:
            return 0.0
        vec = x.data if hasattr(x, "data") else x
        diff = [v - m for v, m in zip(vec, self._feature_mean)]
        tmp = [sum(row[j] * diff[j] for j in range(len(diff))) for row in self._inv_cov]
        return float(sum(diff[i] * tmp[i] for i in range(len(diff))))

    def _check_domain(self, inputs: Any) -> None:
        if (
            self._feature_mean is None
            or self._inv_cov is None
            or self._mahalanobis_threshold is None
        ):
            return
        distances = [self._mahalanobis_distance(x) for x in inputs]
        if any(d > self._mahalanobis_threshold for d in distances):
            raise OutOfDomainError(
                f"Mahalanobis distance {max(distances):.3f} exceeds "
                f"threshold {self._mahalanobis_threshold:.3f}"
            )

    # ------------------------------------------------------------------
    # Optional lifecycle helpers
    def train(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        """Train the surrogate model.

        Parameters and return values are model-specific. Sub-classes may
        override :meth:`_train` to provide training capabilities.  The default
        implementation simply returns an empty metrics dictionary.
        """

        return self._train(*args, **kwargs)

    def _train(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        return {}

    def save(self, path: str | Path | None = None) -> Path:
        """Persist the model to ``path``.

        Parameters
        ----------
        path:
            Destination for the serialized model. Implementations may fall
            back to ``self.model_path`` when ``path`` is ``None``.
        """

        dest = Path(path or self.model_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._save(dest)
        return dest

    def _save(self, path: Path) -> None:
        with path.open("wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(
        cls, model_path: str | Path, *args: Any, **kwargs: Any
    ) -> "SurrogateModel":
        """Load a serialized model from ``model_path``."""

        return cls._load(Path(model_path), *args, **kwargs)

    @classmethod
    def _load(cls, path: Path, *args: Any, **kwargs: Any) -> "SurrogateModel":
        with path.open("rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Serialized object is {type(obj)!r}, expected {cls!r}")
        return obj


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

    def _predict(self, inputs: np.ndarray) -> np.ndarray:
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

        This method delegates to :meth:`_train` which performs the actual
        optimisation.  Sub-classes may override :meth:`_train` to customise the
        process.
        """

        return super().train(dataloader, epochs=epochs, lr=lr, metadata=metadata)

    def _train(
        self,
        dataloader: Any,
        epochs: int = 10,
        lr: float = 1e-3,
        metadata: Any | None = None,
    ) -> dict[str, float]:
        try:
            from .training import train_torch_model
        except Exception as exc:  # pragma: no cover - optional
            raise ImportError("Training utilities require PyTorch") from exc

        self.model, metrics = train_torch_model(
            self.model, dataloader, epochs=epochs, lr=lr, metadata=metadata
        )
        return metrics

    def _save(self, path: Path) -> None:
        scripted = self._torch.jit.script(self.model)
        scripted.save(str(path))

    @classmethod
    def _load(cls, path: Path, device: str = "cpu") -> "TorchSurrogateModel":
        return cls(path, device=device)


class ONNXSurrogateModel(SurrogateModel):
    """Surrogate model using ``onnxruntime`` for inference."""

    def __init__(self, model_path: str | Path) -> None:
        super().__init__(model_path)
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("onnxruntime is required for ONNXSurrogateModel") from exc
        self.session = ort.InferenceSession(str(self.model_path))

    def _predict(self, inputs: np.ndarray) -> np.ndarray:
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: inputs})
        return outputs[0]

    # ------------------------------------------------------------------
    def _train(
        self, *args: Any, **kwargs: Any
    ) -> dict[str, float]:  # pragma: no cover - thin wrapper
        """ONNX models are inference-only and cannot be trained."""

        raise NotImplementedError("ONNX models do not support in-place training")

    def _save(self, path: Path) -> None:
        if path != self.model_path:
            import shutil

            shutil.copy(self.model_path, path)

    @classmethod
    def _load(cls, path: Path) -> "ONNXSurrogateModel":
        return cls(path)


__all__ = [
    "SurrogateModel",
    "TorchSurrogateModel",
    "ONNXSurrogateModel",
    "OutOfDomainError",
]
