from dataclasses import dataclass
from typing import Optional


@dataclass
class CLIErrorInfo:
    """Information describing a CLI error."""

    code: str
    hint: str


ERRORS: dict[str, CLIErrorInfo] = {
    "CONFIG": CLIErrorInfo("E001", "Check configuration path and values."),
    "SIMULATION": CLIErrorInfo("E002", "Verify simulation parameters."),
    "VALIDATION": CLIErrorInfo("E003", "Ensure dataset path and config are correct."),
    "PLOT": CLIErrorInfo("E004", "Confirm the input contains HDF5 outputs."),
    "DIAGNOSTICS": CLIErrorInfo("E005", "Provide valid history and configuration JSON files."),
    "NOTEBOOK": CLIErrorInfo("E006", "Install Jupyter to use notebook mode."),
    "UNEXPECTED": CLIErrorInfo("E999", "Please report this issue."),
}


def format_error(kind: str, message: str, hint: Optional[str] = None) -> str:
    """Format an error message with code and remediation hint."""
    info = ERRORS.get(kind, ERRORS["UNEXPECTED"])
    msg = f"[{info.code}] {message}"
    hint_text = hint or info.hint
    if hint_text:
        msg += f"\nHint: {hint_text}"
    return msg
