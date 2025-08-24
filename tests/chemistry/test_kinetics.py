from pathlib import Path
import importlib.util

import pytest

# Load modules directly to avoid heavy package imports
root = Path(__file__).resolve().parents[2]
kin_path = root / "src" / "dpf2" / "chemistry" / "kinetics.py"
spec_k = importlib.util.spec_from_file_location("kinetics", kin_path)
kinetics_mod = importlib.util.module_from_spec(spec_k)
import sys
sys.modules["kinetics"] = kinetics_mod
spec_k.loader.exec_module(kinetics_mod)  # type: ignore[attr-defined]

RateTable = kinetics_mod.RateTable
RateEquations = kinetics_mod.RateEquations


def data_path(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "data" / name


def test_rate_table_interpolation():
    table = RateTable.from_csv(data_path("crm_dummy.csv"))
    ion = table.ion_rate([10.0])[0]
    rec = table.rec_rate([10.0])[0]
    assert ion == pytest.approx(5.0)
    assert rec == pytest.approx(0.0)


def test_kinetics_converges_to_flychk():
    rates = RateTable.from_csv(data_path("crm_dummy.csv"))
    kinetics = RateEquations(rates, levels=2)

    n_total = 1.0
    T = 10.0
    n = [n_total - 1e-3, 1e-3]
    dt = 0.1
    for _ in range(200):
        n = kinetics.step(n, T, dt)
    zbar = kinetics.mean_charge(n)

    # Reference mean charge state from a FLYCHK table
    data = load_csv(data_path("flychk_dummy.csv"))
    T_ref = [row[0] for row in data]
    Z_ref = [row[1] for row in data]
    ref = interp(T, T_ref, Z_ref)
    assert abs(zbar - ref) <= 1e-2 * abs(ref)


def load_csv(path: Path) -> list[list[float]]:
    with open(path) as f:
        lines = f.read().strip().splitlines()[1:]
    return [list(map(float, line.split(","))) for line in lines]


def interp(x: float, xp: list[float], fp: list[float]) -> float:
    if x <= xp[0]:
        return fp[0]
    if x >= xp[-1]:
        return fp[-1]
    for i in range(1, len(xp)):
        if x < xp[i]:
            x0, x1 = xp[i - 1], xp[i]
            f0, f1 = fp[i - 1], fp[i]
            return f0 + (f1 - f0) * (x - x0) / (x1 - x0)
    return fp[-1]
