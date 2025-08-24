"""Core components for simplified DPF simulations.

Base classes for solver components now live in :mod:`dpf2.core.bases` and
are no longer re-exported here.
"""

try:  # pragma: no cover - optional heavy dependencies
    from .config import DPFConfig
    from .simulation import DPFSimulation
    from .external_circuit import ExternalCircuit

    __all__ = ["DPFConfig", "DPFSimulation", "ExternalCircuit"]
except Exception:  # pragma: no cover - fallback when pydantic is unavailable
    __all__: list[str] = []
