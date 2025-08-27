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

    for _ in range(epochs):
        for xb, yb in dataloader:
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

    if metadata is not None:
        metadata.ml_metadata = MLMetadata(
            model=type(model).__name__,
            version=torch.__version__,
            engine="torch",
            optimizer="Adam",
        )
        metadata.ml_result = MLResult(model_error=mse)

    return model, {"mse": mse}


__all__ = ["load_numpy_dataset", "train_torch_model"]
