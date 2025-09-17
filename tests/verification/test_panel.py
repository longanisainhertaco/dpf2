import h5py_stub as h5py
from dpf2.verification import VerificationPanel
from dpf2.diagnostics.quality_dashboard import QualityDashboard


def test_error_convergence(tmp_path):
    q = QualityDashboard(output_dir=tmp_path / "quality")
    panel = VerificationPanel(output_file=tmp_path / "verify.h5", quality=q)
    res = panel.run_brio_wu(sizes=(8, 16, 32))
    assert all(e1 > e2 for e1, e2 in zip(res["l1_error"], res["l1_error"][1:]))
    with h5py.File(tmp_path / "verify.h5") as h5:
        grp = h5["brio_wu"]
        assert "shock_count" in grp
        assert len(grp["l1_error"].data) == 3


def test_threshold_reporting(tmp_path, caplog):
    q = QualityDashboard(output_dir=tmp_path / "quality")
    q.max_l1_error = 1e-6
    q.max_divB_norm = 1e-6
    panel = VerificationPanel(output_file=tmp_path / "verify.h5", quality=q)
    res = panel.run_orszag_tang(sizes=(8, 16))
    assert not res["passed"]
    text = caplog.text
    assert "L1 error above threshold" in text
    assert "∇·B norm above threshold" in text
