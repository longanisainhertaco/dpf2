import math
from pathlib import Path

from dpf2.diagnostics.regime_panel import RegimePanel


def _compute(n, T, B, v, eta, mfp, tau_e, L):
    mu_0 = 4e-7 * math.pi
    k_B = 1.380649e-23
    m_p = 1.67262192369e-27
    e = 1.602176634e-19
    m_e = 9.1093837015e-31
    sigma = 1.0 / eta
    rho = n * m_p
    v_a = B / math.sqrt(mu_0 * rho)
    S = mu_0 * sigma * v_a * L
    beta = 2 * mu_0 * n * k_B * T / (B * B)
    M_A = v / v_a
    R_m = mu_0 * sigma * v * L
    K_n = mfp / L
    omega = (e * B / m_e) * tau_e
    return {
        "S": S,
        "beta": beta,
        "M_A": M_A,
        "R_m": R_m,
        "K_n": K_n,
        "omega_ce_tau_e": omega,
    }


def test_regime_panel_flags_and_history(tmp_path):
    L = 1.0
    n = 1e20
    T = 1e3
    B = 0.1
    v = 1e5
    eta = 1e-6
    mfp = 0.1
    tau_e = 1e-9

    vals = _compute(n, T, B, v, eta, mfp, tau_e, L)
    thresholds = {
        "S": vals["S"] * 2,
        "beta": vals["beta"] / 2,
        "M_A": vals["M_A"] / 2,
        "R_m": vals["R_m"] * 2,
        "K_n": vals["K_n"] / 2,
        "omega_ce_tau_e": vals["omega_ce_tau_e"] * 2,
    }

    panel = RegimePanel(L=L, thresholds=thresholds)
    entry = panel.log(1, n, T, B, v, eta, mfp, tau_e)
    for key in thresholds:
        assert entry["violations"][key]

    panel.log(2, n, T, B, v, eta, mfp, tau_e)
    assert len(panel.history) == 2

    csv_path = panel.to_csv(tmp_path / "regime.csv")
    content = Path(csv_path).read_text().strip().splitlines()
    assert len(content) == 3  # header + 2 entries
    assert "S_violated" in content[0]
