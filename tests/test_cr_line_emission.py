import csv
from pathlib import Path

import pytest

from dpf2.radiation.xray_emission_model import cr_line_emission


def test_cr_model_matches_reference():
    path = Path(__file__).resolve().parents[1] / "Validation" / "sxr_reference.csv"
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            Te = float(row["Te_eV"])
            ne = float(row["ne_cm3"])
            ref_ne = float(row["Ne_line"])
            ref_ar = float(row["Ar_line"])
            ne_em = cr_line_emission(Te, ne, "Ne")["Ne_X"]
            ar_em = cr_line_emission(Te, ne, "Ar")["Ar_Kalpha"]
            assert ne_em == pytest.approx(ref_ne, rel=1e-6)
            assert ar_em == pytest.approx(ref_ar, rel=1e-6)


def test_cr_model_requires_positive_inputs():
    with pytest.raises(ValueError):
        cr_line_emission(-1.0, 1e10, "Ne")
    with pytest.raises(ValueError):
        cr_line_emission(100.0, 0.0, "Ar")
    with pytest.raises(ValueError):
        cr_line_emission(100.0, 1e10, "Xe")
