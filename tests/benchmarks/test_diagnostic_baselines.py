import json
from math import isclose
from pathlib import Path

import importlib.util

BASE = Path(__file__).resolve().parents[2] / "src" / "dpf2" / "diagnostics"

def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, BASE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module

compute_neutron_yield = _load("neutron_yield").compute_neutron_yield
compute_xray_spectrum = _load("xray_spectra").compute_xray_spectrum


def load_case() -> dict:
    case_path = Path(__file__).with_name('simple_case.json')
    with case_path.open() as fh:
        return json.load(fh)


def test_neutron_yield_baseline():
    data = load_case()
    rate = data['reaction_rate']
    yield_val = compute_neutron_yield(rate, data['dt'])
    assert isclose(yield_val, data['expected_yield'])


def test_xray_spectrum_baseline():
    data = load_case()
    centers, spectrum = compute_xray_spectrum(
        data['energies'], data['intensities'], data['bins']
    )
    expected = data['expected_spectrum']
    assert all(isclose(a, b) for a, b in zip(spectrum, expected))
    # ensure bin centers computed correctly for coverage
    assert all(isclose(c, e) for c, e in zip(centers, [1.5, 2.5, 3.5]))
