import h5py_stub as h5py

from dpf2.verification import VerificationPanel
from dpf2.diagnostics.quality_dashboard import QualityDashboard
from dpf2.ui.verification_panel import VerificationPanelUI


def test_brio_wu_hdf5(tmp_path):
    q = QualityDashboard(output_dir=tmp_path / "quality")
    panel = VerificationPanel(output_file=tmp_path / "verify.h5", quality=q)
    res = panel.run_brio_wu(sizes=(8, 16))
    assert res["passed"]
    with h5py.File(tmp_path / "verify.h5") as h5:
        assert "brio_wu" in h5
        grp = h5["brio_wu"]
        assert len(grp["l1_error"].data) == 2
        assert "observed_order" in grp


def test_quality_thresholds(tmp_path, caplog):
    q = QualityDashboard(output_dir=tmp_path / "quality")
    q.max_l1_error = 1e-6
    q.max_divB_norm = 1e-6
    q.max_energy_drift = 1e-6
    panel = VerificationPanel(output_file=tmp_path / "verify.h5", quality=q)
    res = panel.run_brio_wu(sizes=(8, 16))
    assert not res["passed"]
    text = caplog.text
    assert "L1 error above threshold" in text
    assert "∇·B norm above threshold" in text


def test_ui_summary(tmp_path):
    q = QualityDashboard(output_dir=tmp_path / "quality")
    ui = VerificationPanelUI(output_file=tmp_path / "verify.h5", quality=q)
    summary = ui.summarize()
    assert "Numerics verification results:" in summary
    assert "brio_wu" in summary
