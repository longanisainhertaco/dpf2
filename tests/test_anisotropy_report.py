from dpf2.synthetic_diagnostics import autocorrelated_tof_iv_report
from dpf2.core.bases import CouplingState


def test_autocorrelated_report(tmp_path):
    history = [CouplingState(current=c, voltage=c) for c in [0.0, 1.0, 5.0, 1.0, 0.0]]
    dt = 1e-9
    out = autocorrelated_tof_iv_report(history, dt, 1.0, energies_mev=[2.45], output_dir=tmp_path)
    assert out.exists()
    first = out.read_text().splitlines()[0]
    assert first.split(',')[0] == 'lag_s'
