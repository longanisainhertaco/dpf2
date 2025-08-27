"""Surrogate model interfaces for AI integration."""

from .surrogate import SurrogateModel, TorchSurrogateModel, ONNXSurrogateModel
from .training import load_numpy_dataset, train_torch_model

__all__ = [
    "SurrogateModel",
    "TorchSurrogateModel",
    "ONNXSurrogateModel",
    "load_numpy_dataset",
    "train_torch_model",
]
