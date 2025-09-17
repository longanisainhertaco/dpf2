from pathlib import Path


def test_regime_dashboard_websocket_subscription():
    content = Path("web/frontend/src/RegimeDashboard.jsx").read_text()
    assert "new WebSocket" in content
    assert "/ws/regime" in content
    for key in ["S", "beta", "M_A", "R_m", "K_n", "omega_ce_tau_e"]:
        assert key in content
    # client-side computation and plotting helpers
    assert "computeParams" in content
    assert "thresholds" in content
    assert "history" in content
    assert "warning" in content
    assert "Export" in content
    assert "svg" in content
