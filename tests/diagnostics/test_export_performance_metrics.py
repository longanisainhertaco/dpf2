import csv

import pytest
import h5py

from dpf2.diagnostics import compute_performance_metrics, export_performance_metrics


def test_export_performance_metrics(tmp_path):
    pytest.importorskip("matplotlib")
    metrics = compute_performance_metrics(
        1e6,
        rep_rate_hz=1.0,
        energy_out_j=10.0,
        energy_in_j=20.0,
        electrode_mass_g=100.0,
        erosion_per_shot_g=0.01,
    )
    export_performance_metrics(metrics, tmp_path)

    csv_path = tmp_path / "performance_metrics.csv"
    h5_path = tmp_path / "performance_metrics.h5"
    md_path = tmp_path / "summary.md"
    png_path = tmp_path / "performance_metrics.png"

    assert csv_path.exists()
    assert h5_path.exists()
    assert md_path.exists()
    assert png_path.exists()

    with csv_path.open() as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["metric", "value"]

    with h5py.File(h5_path, "r") as h5:
        assert "yield_per_shot" in h5
