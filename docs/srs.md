# Software Requirements Specification

Version 1.0

## Requirements

- **R1**: The system shall provide a command-line interface to run simulations from configuration files.
- **R2**: The system shall support a configuration schema that validates simulation parameters.
- **R3**: The system shall provide a simulation engine capable of running dense plasma focus simulations.
- **R4**: The system shall include pinch models for analytic, semi-analytic, and MHD simulations.
- **R5**: The system shall offer diagnostics utilities for validating simulation data.

## Traceability Matrix

| Requirement | Implementation | Tests |
|-------------|----------------|-------|
| R1 | `src/dpf2/cli/main.py` | `tests/test_cli_diag_frequency.py` |
| R2 | `src/dpf2/dpf_config.py` | `tests/test_dpf_config.py` |
| R3 | `src/dpf2/simulation_engine.py` | `tests/test_simulation_engine.py` |
| R4 | `src/dpf2/pinch_models.py` | `tests/test_pinch_models.py` |
| R5 | `src/dpf2/diagnostics/__init__.py` | `tests/test_diagnostics.py` |

## Change Control

Scope changes are rejected unless formally approved. Approved changes must be appended to this document with updated version numbers.

