import pytest
from fastapi.testclient import TestClient

from web.backend.main import app


@pytest.fixture
def client():
    return TestClient(app)


def _auth_headers(client):
    res = client.post('/token', data={'username': 'user', 'password': 'secret'})
    return {'Authorization': f"Bearer {res.json()['access_token']}"}


def test_snapshot_roundtrip(client):
    headers = _auth_headers(client)
    state = {'config': {'foo': 1}, 'voltage': 5.0}
    resp = client.post('/snapshot/save', json={'state': state}, headers=headers)
    assert resp.status_code == 200
    snap_id = resp.json()['id']
    resp2 = client.get(f'/snapshot/{snap_id}')
    assert resp2.json() == state
