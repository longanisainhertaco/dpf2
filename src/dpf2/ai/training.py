"""Utilities for training surrogate models."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore

from ..metadata import MLMetadata, MLResult, Metadata


def load_numpy_dataset(
    x: np.ndarray, y: np.ndarray, batch_size: int = 32
) -> "DataLoader":
    """Create a torch ``DataLoader`` from numpy arrays."""

    if torch is None:  # pragma: no cover - environment w/out torch
        raise ImportError("PyTorch is required for dataset loading")

    inputs = torch.as_tensor(x, dtype=torch.float32)
    targets = torch.as_tensor(y, dtype=torch.float32)
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_torch_model(
    model: "torch.nn.Module",
    dataloader: Iterable,
    *,
    epochs: int = 10,
    lr: float = 1e-3,
    metadata: Metadata | None = None,
) -> Tuple["torch.nn.Module", Dict[str, float]]:
    """Train ``model`` using data from ``dataloader``.

    The model is optimized with Adam and mean squared error loss. Evaluation
    metrics are returned as a dictionary. When ``metadata`` is supplied, its
    ``ml_metadata`` and ``ml_result`` fields are populated accordingly.
    """

    if torch is None:  # pragma: no cover - environment w/out torch
        raise ImportError("PyTorch is required for training")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for epoch in range(epochs):
        for xb, yb in dataloader:
            if epoch == 0:
                features.append(xb.detach().cpu().numpy())
                targets.append(yb.detach().cpu().numpy())
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

    # Evaluation ---------------------------------------------------------
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb in dataloader:
            pred = model(xb)
            total_loss += loss_fn(pred, yb).item() * len(xb)
            count += len(xb)
    mse = total_loss / max(count, 1)

    # Domain statistics --------------------------------------------------
    x_all = np.concatenate(features, axis=0) if features else np.empty((0, 1))
    y_all = np.concatenate(targets, axis=0) if targets else np.empty((0, 1))
    feature_mean = x_all.mean(axis=0) if x_all.size else np.zeros(1)
    feature_cov = np.cov(x_all, rowvar=False) if x_all.size else np.zeros((1, 1))
    cov_inv = np.linalg.pinv(feature_cov) if x_all.size else np.zeros((1, 1))
    diff = x_all - feature_mean
    distances = np.einsum("ij,jk,ik->i", diff, cov_inv, diff) if x_all.size else np.zeros(1)
    mahal_threshold = float(np.quantile(distances, 0.99)) if distances.size else 0.0

    with torch.no_grad():
        preds = model(torch.as_tensor(x_all, dtype=torch.float32)) if x_all.size else torch.zeros_like(torch.as_tensor(y_all))
    resid = np.abs(y_all - preds.cpu().numpy()) if x_all.size else np.zeros(1)
    quantile = float(np.quantile(resid, 0.95)) if resid.size else 0.0

    if metadata is not None:
        metadata.ml_metadata = MLMetadata(
            model=type(model).__name__,
            version=torch.__version__,
            engine="torch",
            optimizer="Adam",
        )
        metadata.ml_result = MLResult(model_error=mse)

    metrics = {
        "mse": mse,
        "feature_mean": feature_mean.tolist(),
        "feature_cov": feature_cov.tolist() if feature_cov.ndim == 2 else [float(feature_cov)],
        "mahalanobis_threshold": mahal_threshold,
        "quantile": quantile,
    }

    return model, metrics


__all__ = ["load_numpy_dataset", "train_torch_model"]
