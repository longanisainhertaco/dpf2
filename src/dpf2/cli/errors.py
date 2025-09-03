"""Centralised error codes and remediation hints for the CLI."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CLIErrorInfo:
    """Definition of a CLI error with structured metadata."""

    code: str
    tip: str | None = None


ERRORS: dict[str, CLIErrorInfo] = {
    "CONFIG": CLIErrorInfo(
        "CLI001", "Check configuration path and values."
    ),
    "SIMULATION": CLIErrorInfo(
        "CLI002", "Verify simulation parameters."
    ),
    "VALIDATION": CLIErrorInfo(
        "CLI003", "Ensure dataset path and config are correct."
    ),
    "PLOT": CLIErrorInfo(
        "CLI004", "Confirm the input contains HDF5 outputs or install plotting backend."
    ),
    "DIAGNOSTICS": CLIErrorInfo(
        "CLI005", "Provide valid history and configuration JSON files."
    ),
    "NOTEBOOK": CLIErrorInfo(
        "CLI006", "Install Jupyter to use notebook mode."
    ),
    "UNEXPECTED": CLIErrorInfo(
        "CLI999", "Please report this issue."
    ),
}


def format_error(kind: str, message: str, tip: Optional[str] = None) -> str:
    """Format an error message with its code and remediation tip."""

    info = ERRORS.get(kind, ERRORS["UNEXPECTED"])
    text = f"[{info.code}] {message}"
    hint = tip or info.tip
    if hint:
        text += f"\nHint: {hint}"
    return text

