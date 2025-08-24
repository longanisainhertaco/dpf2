"""Custom exception hierarchy for the DPF2 project."""
from __future__ import annotations

class DPFError(Exception):
    """Base class for all custom exceptions raised by DPF2."""

class ConfigurationError(DPFError):
    """Raised when a simulation or server configuration is invalid."""

class SimulationError(DPFError):
    """Raised for errors occurring during simulation execution."""

class ServerError(DPFError):
    """Base class for server-related errors."""

class ExportError(ServerError):
    """Raised when exporting results from the server fails."""
