"""Compatibility wrapper for the circuit solver.

This module re-exports :func:`dpf2.circuit_solver.run_circuit_simulation`
for legacy code that imported ``run_circuit_simulation`` from the top-level
``rlc_solver`` module.
"""

from dpf2.circuit_solver import run_circuit_simulation

__all__ = ["run_circuit_simulation"]
