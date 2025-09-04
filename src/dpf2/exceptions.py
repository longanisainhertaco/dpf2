"""Custom exception hierarchy for the DPF2 project."""
from __future__ import annotations

class DPFError(Exception):
    """Base class for all custom exceptions raised by DPF2."""

class ConfigurationError(DPFError):
    """Raised when a simulation or server configuration is invalid.

    Parameters
    ----------
    message:
        Human readable error message.
    fields:
        Optional list of configuration field names that are related to the
        error.  These can be displayed by CLI helpers to guide the user
        towards the problematic parts of the configuration.
    hints:
        Optional mapping of configuration field names to contextual hint
        messages describing why validation failed for that field.
    """

    def __init__(
        self,
        message: str,
        fields: list[str] | None = None,
        hints: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.fields = fields or []
        self.hints = hints or {}

class SimulationRuntimeError(DPFError):
    """Raised for runtime errors occurring during simulation execution."""

class OutOfDomainError(DPFError):
    """Raised when model inputs are outside the training domain."""

class ServerError(DPFError):
    """Base class for server-related errors."""

class ExportError(ServerError):
    """Raised when exporting results from the server fails."""
