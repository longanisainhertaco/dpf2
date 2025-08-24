from pathlib import Path
import importlib.util

import numpy as np
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
    ion = table.ion_rate(np.array([10.0]))
    rec = table.rec_rate(np.array([10.0]))
    assert ion == pytest.approx(5.0)
    assert rec == pytest.approx(0.0)


def test_kinetics_converges_to_flychk():
    rates = RateTable.from_csv(data_path("crm_dummy.csv"))
    kinetics = RateEquations(rates)

    n_total = 1.0
    T = 10.0
    ne = 1e-3
    dt = 0.1
    for _ in range(200):
        ne = kinetics.step(ne, n_total, T, dt)
    zbar = ne / n_total

    # Reference mean charge state from a FLYCHK table
    flychk_data = np.loadtxt(data_path("flychk_dummy.csv"), delimiter=",", skiprows=1)
    T_ref, Z_ref = flychk_data[:, 0], flychk_data[:, 1]
    ref = np.interp(T, T_ref, Z_ref, left=Z_ref[0], right=Z_ref[-1])
    np.testing.assert_allclose(zbar, ref, rtol=1e-2)
