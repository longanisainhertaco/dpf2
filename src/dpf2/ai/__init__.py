"""Surrogate model interfaces for AI integration."""

from .surrogate import SurrogateModel, TorchSurrogateModel, ONNXSurrogateModel

__all__ = [
    "SurrogateModel",
    "TorchSurrogateModel",
    "ONNXSurrogateModel",
]
