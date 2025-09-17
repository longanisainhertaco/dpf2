import sys
import types
import importlib.util
from pathlib import Path

import pytest


def _load_distributed():
    """Load the distributed circuit module with minimal stubs."""

    pkg = types.ModuleType("dpf2")
    pkg.__path__ = []

    core_pkg = types.ModuleType("dpf2.core")
    bases_pkg = types.ModuleType("dpf2.core.bases")

    class PlasmaSolverBase:
        pass

    bases_pkg.PlasmaSolverBase = PlasmaSolverBase
    core_pkg.bases = bases_pkg

    sys.modules["dpf2"] = pkg
    sys.modules["dpf2.core"] = core_pkg
    sys.modules["dpf2.core.bases"] = bases_pkg

    module_path = (
        Path(__file__).resolve().parent.parent / "src/dpf2/circuit/distributed.py"
    )
    spec = importlib.util.spec_from_file_location(
        "dpf2.circuit.distributed", module_path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dpf2.circuit.distributed"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_warns_when_material_tables_missing():
    mod = _load_distributed()
    TransmissionLineSegment = mod.TransmissionLineSegment

    with pytest.warns(UserWarning) as record:
        TransmissionLineSegment(0, 1, 1.0, 0.0, 0.0, 0.0, material="copper")

    # Ensure warnings include the material name and default values
    assert record
    for w in record:
        msg = str(w.message)
        assert "copper" in msg
        assert "0.0" in msg

    for name in [
        "dpf2.circuit.distributed",
        "dpf2.core.bases",
        "dpf2.core",
        "dpf2",
    ]:
        sys.modules.pop(name, None)
