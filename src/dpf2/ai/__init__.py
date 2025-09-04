"""Surrogate model interfaces for AI integration."""

from .surrogate import SurrogateModel, TorchSurrogateModel, ONNXSurrogateModel
from ..exceptions import OutOfDomainError
from .training import load_numpy_dataset, train_torch_model
from .simple_surrogates import (
    LinearSurrogate,
    ONNXLinearSurrogate,
    load_yield_surrogate,
    load_pinch_time_surrogate,
)

__all__ = [
    "SurrogateModel",
    "TorchSurrogateModel",
    "ONNXSurrogateModel",
    "OutOfDomainError",
    "load_numpy_dataset",
    "train_torch_model",
    "LinearSurrogate",
    "ONNXLinearSurrogate",
    "load_yield_surrogate",
    "load_pinch_time_surrogate",
]
