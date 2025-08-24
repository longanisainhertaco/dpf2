import json
import importlib.util
from pathlib import Path

# Load diagnostic functions without importing the package (avoids pydantic dependency)
ROOT = Path(__file__).resolve().parents[2]

def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"src/dpf2/diagnostics/{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

compute_neutron_yield = _load("neutron_yield").compute_neutron_yield
compute_xray_spectrum = _load("xray_spectra").compute_xray_spectrum
compute_scope_trace = _load("scope_trace").compute_scope_trace

from dpf2.io import RestartManager, StructuredOutputWriter


def test_restart_roundtrip(tmp_path: Path) -> None:
    state = {"time": 1.0, "current": [1.0, 2.0, 3.0]}
    mgr = RestartManager(tmp_path / "restart.json")
    mgr.save(state)
    loaded = mgr.load()
    assert loaded == state


def test_diagnostic_functions(tmp_path: Path) -> None:
    # neutron yield
    rate = [1.0, 2.0, 3.0, 4.0]
    dt = 0.5
    assert compute_neutron_yield(rate, dt) == 5.0

    # xray spectrum
    energies = [1.0, 2.0, 5.0, 7.0]
    intensities = [1.0, 1.0, 1.0, 1.0]
    bins = [0.0, 3.0, 6.0, 9.0]
    centers, counts = compute_xray_spectrum(energies, intensities, bins)
    assert counts == [2.0, 1.0, 1.0]
    assert centers == [1.5, 4.5, 7.5]

    # scope trace baseline subtraction
    times = [0.0, 1.0, 2.0, 3.0]
    values = [0.0, 1.0, 2.0, 3.0]
    t_out, v_out = compute_scope_trace(times, values)
    assert t_out == times
    assert abs(v_out[0] + 1.5) < 1e-12
    assert abs(v_out[-1] - 1.5) < 1e-12

    # structured output writing
    writer = StructuredOutputWriter(tmp_path)
    path = writer.write_json({"a": 1}, "diag")
    with path.open() as f:
        data = json.load(f)
    assert data["a"] == 1
