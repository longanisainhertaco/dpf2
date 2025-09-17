import base64
import time
from werkzeug.security import generate_password_hash

from dpf2.simulation.dpf_simulator_server import app, simulation_manager


def _auth_headers():
    token = base64.b64encode(b"admin:secret").decode("utf-8")
    return {"Authorization": f"Basic {token}"}


def _setup_app():
    app.config["ADMIN_USERNAME"] = "admin"
    app.config["ADMIN_PASSWORD_HASH"] = generate_password_hash("secret")
    app.config["MAX_SIMULTANEOUS_SIMULATIONS"] = 2
    app.config["TELEMETRY_INTERVAL"] = 0.01


def test_start_stop_and_retrieve_state():
    _setup_app()
    client = app.test_client()
    config = {
        "dx": 1.0,
        "dy": 1.0,
        "dz": 1.0,
        "sim_time": 0.05,
        "dt_init": 0.01,
        "grid_shape": [1, 1, 1],
        "domain_lo": [0.0, 0.0, 0.0],
        "circuit": {
            "C": 1.0,
            "V0": 1.0,
            "L0": 1.0,
            "R0": 0.1,
            "anode_radius": 0.01,
            "cathode_radius": 0.02,
        },
    }
    resp = client.post("/api/simulate", json=config, headers=_auth_headers())
    assert resp.status_code == 202
    sim_id = resp.get_json()["sim_id"]

    # Allow the simulation thread to start
    time.sleep(0.05)
    stop_resp = client.post(f"/api/stop/{sim_id}", headers=_auth_headers())
    assert stop_resp.status_code == 204

    # Ensure we can access state and diagnostics through the interface
    sim = simulation_manager.get_simulation(sim_id)
    sim.get_state()
    sim.get_diagnostics()

    # Join background thread to avoid leaking resources
    thread = simulation_manager.get_simulation_thread(sim_id)
    thread.join(timeout=1)
