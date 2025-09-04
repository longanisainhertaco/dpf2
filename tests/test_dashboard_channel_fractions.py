import json
import pytest


def test_channel_fractions_displayed(tmp_path):
    flask = pytest.importorskip("flask")
    from dpf2.web.app import create_app

    cf = {"thermonuclear": 0.7, "beam_target": 0.3}
    (tmp_path / "channel_fractions.json").write_text(json.dumps(cf))
    app = create_app()
    client = app.test_client()
    resp = client.get("/diagnostics", query_string={"output": str(tmp_path)})
    assert resp.status_code == 200
    data = resp.data.decode("utf-8").lower()
    assert "thermonuclear" in data
    assert "0.7" in data
