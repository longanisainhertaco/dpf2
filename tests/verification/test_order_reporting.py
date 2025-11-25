import h5py_stub as h5py

from dpf2.verification import VerificationPanel


def test_brio_wu_reports_observed_order_and_divergence(tmp_path):
    panel = VerificationPanel(output_file=tmp_path / "verify.h5")
    res = panel.run_brio_wu(sizes=(4, 8, 16))
    assert len(res["observed_order"]) == 2
    assert all(val >= 0.0 for val in res["divB_norm"])
    with h5py.File(tmp_path / "verify.h5") as h5:
        grp = h5["brio_wu"]
        assert "divB_norm" in grp
        assert len(grp["observed_order"].data) == 2


def test_mms_and_orszag_tang_order_tracking(tmp_path):
    panel = VerificationPanel(output_file=tmp_path / "verify.h5")
    mms = panel.run_mms_ideal_mhd(sizes=(8, 16, 32))
    ot = panel.run_orszag_tang(sizes=(6, 12, 24))
    assert len(mms["observed_order"]) == 2
    assert len(ot["observed_order"]) == 2
    assert mms["divB_norm"][0] >= 0.0
    assert ot["divB_norm"][0] >= 0.0
