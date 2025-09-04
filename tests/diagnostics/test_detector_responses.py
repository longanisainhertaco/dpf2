from pathlib import Path

from dpf2.diagnostics import neutron, xray, iv_probes

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "detectors"


def test_neutron_gating_only():
    resp = neutron.load_response(DATA_DIR / "neutron_detector.json")
    times = [0, 1, 2, 3, 4]
    signal = [1, 1, 1, 1, 1]
    processed = neutron.apply_response(times, signal, resp)
    assert processed == [0.0, 1.0, 1.0, 1.0, 0.0]


def test_xray_dead_time():
    resp = xray.load_response(DATA_DIR / "xray_detector.json")
    times = [0.0, 0.1, 0.2, 0.5]
    signal = [1, 1, 1, 1]
    processed = xray.apply_response(times, signal, resp)
    assert processed == [1.0, 0.0, 0.0, 1.0]


def test_iv_probe_dispersion():
    resp = iv_probes.load_response(DATA_DIR / "iv_probe.json")
    times = [0.0, 1.0, 2.0, 3.0]
    signal = [1.0, 0.0, 0.0, 1.0]
    processed = iv_probes.apply_response(times, signal, resp)
    assert processed == [0.5, 0.5, 0.0, 0.5]
