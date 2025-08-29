import json
import math
import random
from pathlib import Path


def stats(values):
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    return mean, math.sqrt(var)


def run_ensemble(params):
    random.seed(params["seed"])
    samples = params.get("samples", 100)
    jitter = params.get("jitter_pct", 0.0) / 100.0
    scale = params["scale"]
    V0 = params["bank_voltage_kv"] * 1000.0
    currents = []
    yields = []
    for _ in range(samples):
        V = random.gauss(V0, V0 * jitter)
        I = scale * V
        Y = 1e-6 * I * I
        currents.append(I)
        yields.append(Y)
    c_stats = stats(currents)
    y_stats = stats(yields)
    return c_stats, y_stats


def test_monte_carlo_reference_shots():
    ref_dir = Path(__file__).resolve().parents[2] / "ReferenceMaterial"
    for name in ["shot_deuterium_20kV.json", "shot_argon_30kV.json"]:
        params = json.loads((ref_dir / name).read_text())
        (c_mean, c_std), (y_mean, y_std) = run_ensemble(params)
        assert math.isclose(c_mean, params["current_mean"], rel_tol=0.0, abs_tol=1e-12)
        assert math.isclose(c_std, params["current_std"], rel_tol=0.0, abs_tol=1e-12)
        assert math.isclose(y_mean, params["yield_mean"], rel_tol=0.0, abs_tol=1e-12)
        assert math.isclose(y_std, params["yield_std"], rel_tol=0.0, abs_tol=1e-12)
        low_c = c_mean - c_std * 2
        high_c = c_mean + c_std * 2
        assert low_c <= params["expected_current"] <= high_c
        low_y = y_mean - y_std * 2
        high_y = y_mean + y_std * 2
        assert low_y <= params["expected_yield"] <= high_y

