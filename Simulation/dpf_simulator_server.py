import os
import uuid
import json
import logging
import time
import threading
import argparse
import resource
from functools import wraps
from typing import Dict, Any

import sympy as sp
from flask import Flask, request, jsonify, send_file, abort
from flask_sock import Sock
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from dpf_simulation import DPFSimulation, ConfigurationError as SimConfigurationError
from config_schema import ServerConfig, FieldManagerConfig # Import FieldManagerConfig
from utils import FieldManager # Import FieldManager

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DPFSimulatorServer")

app = Flask(__name__)
sock = Sock(app)

# Custom Exceptions
class ServerError(Exception):
    pass

class ConfigurationError(ServerError):
    pass

class SimulationError(ServerError):
    pass

class ExportError(ServerError):
    pass

# ——— Simulation Interface ———
class SimulationInterface:
    def __init__(self, config: Dict[str, Any]):
        pass
    def run(self):
        raise NotImplementedError
    def stop(self):
        raise NotImplementedError
    def get_diagnostics(self):
        raise NotImplementedError
    def get_state(self):
        raise NotImplementedError

# ——— Simulation Manager ———
class SimulationManager:
    def __init__(self):
        self.simulations: Dict[str, SimulationInterface] = {}
        self.sim_threads: Dict[str, threading.Thread] = {}

    def create_simulation(self, config: Dict[str, Any]) -> str:
        sim_id = str(uuid.uuid4())
        try:
            # Create FieldManager
            field_manager_config = config.get('field_manager', {})
            field_manager = FieldManager(
                grid_shape=tuple(config['grid_shape']),
                dx=config['dx'],
                dy=config['dy'],
                dz=config['dz'],
                domain_lo=tuple(config['domain_lo']),
                boundary_conditions=field_manager_config.get('boundary_conditions', {})
            )
            sim = DPFSimulation(config, field_manager=field_manager) # Pass FieldManager to DPFSimulation
            self.simulations[sim_id] = sim
            logger.info(f"Simulation {sim_id} created with parameters: {config}")
            return sim_id
        except SimConfigurationError as e:
            raise ConfigurationError(f"Error creating simulation: {e}")

    def run_simulation(self, sim_id: str):
        sim = self.simulations.get(sim_id)
        if not sim:
            raise SimulationError(f"Simulation {sim_id} not found")
        thread = threading.Thread(target=sim.run, daemon=True)
        self.sim_threads[sim_id] = thread
        thread.start()

    def stop_simulation(self, sim_id: str):
        sim = self.simulations.get(sim_id)
        if not sim:
            raise SimulationError(f"Simulation {sim_id} not found")
        logger.info(f"Stopping simulation {sim_id}...")
        sim.sim_time = sim.current_time

    def get_simulation(self, sim_id: str) -> SimulationInterface:
        sim = self.simulations.get(sim_id)
        if not sim:
            raise SimulationError(f"Simulation {sim_id} not found")
        return sim

    def get_simulation_thread(self, sim_id: str) -> threading.Thread:
        thread = self.sim_threads.get(sim_id)
        if not thread:
            raise SimulationError(f"Simulation {sim_id} thread not found")
        return thread

    def get_telemetry(self, sim_id: str) -> str:
        sim = self.simulations.get(sim_id)
        if not sim:
            raise SimulationError(f"Simulation {sim_id} not found")
        # Gather telemetry data
        tel = {}
        # memory usage
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        tel['cpu_mem_usage_mb'] = mem
        return json.dumps(tel)

# ——— Configuration ———
def load_config(config_file: str) -> ServerConfig:
    try:
        with open(config_file, "r") as f:
            config_data = json.load(f)
            config = ServerConfig(**config_data)  # Validate the config
        logger.info(f"Loaded configuration from {config_file}")
        return config
    except FileNotFoundError:
        raise ConfigurationError(f"Configuration file not found: {config_file}")
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Error decoding JSON from {config_file}: {e}")
    except Exception as e:
        raise ConfigurationError(f"Error validating configuration: {e}")

# ——— Authentication ———
def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not (auth.username == app.config['ADMIN_USERNAME'] and \
                            check_password_hash(app.config['ADMIN_PASSWORD_HASH'], auth.password)):
            logger.warning("Authentication failed for user: %s", auth.username if auth else "None")
            return jsonify({'message': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated

# ——— Resource Management ———
def limit_simulations(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if len(simulation_manager.simulations) >= app.config['MAX_SIMULTANEOUS_SIMULATIONS']:
            logger.warning("Maximum simultaneous simulations reached.")
            return jsonify({'message': 'Server at capacity. Please try again later.'}), 429  # Too Many Requests
        return f(*args, **kwargs)
    return decorated

# ——— Helpers for input validation ———
_dx, _t = sp.symbols('dx sim_time', positive=True)

def _validate_sim_parameters(params):
    try:
        # Required
        dx       = float(params['dx'])
        sim_time = float(params['sim_time'])
        grid_shape = params['grid_shape']
        # Sympy check for positivity
        if not (_dx.subs(_dx, dx) > 0 and _t.subs(_t, sim_time) > 0):
            raise ValueError
        # grid_shape must be list of 3 positive ints
        if (
            not isinstance(grid_shape, list) or
            len(grid_shape) != 3 or
            any((not isinstance(n, int) or n <= 0) for n in grid_shape)
        ):
            raise ValueError
        return dx, sim_time, tuple(grid_shape)
    except Exception:
        abort(400, description="Invalid simulation parameters (dx, sim_time, grid_shape)")

# ——— API Endpoints ———
simulation_manager = SimulationManager()

@app.route("/api/simulate", methods=["POST"])
@requires_auth
@limit_simulations
def start_simulation():
    """
    Launch a new simulation. Expects JSON body:
      {
        "dx": float,
        "sim_time": float,
        "grid_shape": [nx, ny, nz],
        // optionally override default modules' params:
        "circuit_params": {...},
        "collision_params": {...},
        "radiation_params": {...},
        "pic_params": {...},
        "hybrid_params": {...},
        "diagnostics_params": {...},
        "dt_init": float,
        "checkpoint_interval": float,
        "full_output_interval_steps": int,
        "domain_lo": [x0,y0,z0],
        "domain_hi": [x1,y1,z1]
      }
    """
    try:
        params = request.get_json(force=True)
        dx, sim_time, grid_shape = _validate_sim_parameters(params)
        sim_id = simulation_manager.create_simulation(params)
        simulation_manager.run_simulation(sim_id)
        return jsonify({"sim_id": sim_id}), 202
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        abort(400, description=str(e))
    except SimulationError as e:
        logger.error(f"Simulation error: {e}")
        abort(500, description=str(e))
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        abort(500, description="An unexpected error occurred.")

@app.route("/api/stop/<sim_id>", methods=["POST"])
@requires_auth
def stop_simulation(sim_id):
    """
    Gracefully stop a running simulation by setting its end time to now.
    """
    try:
        simulation_manager.stop_simulation(sim_id)
        return "", 204
    except SimulationError as e:
        logger.error(f"Simulation error: {e}")
        abort(404, description=str(e))
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        abort(500, description="An unexpected error occurred.")

@app.route("/api/export/<sim_id>", methods=["GET"])
@requires_auth
def export_results(sim_id):
    """
    Export the HDF5 diagnostics for a completed simulation.
    """
    try:
        sim = simulation_manager.get_simulation(sim_id)
        h5path = sim.diagnostics.hdf5_filename
        # Ensure file exists
        if not os.path.exists(h5path):
            sim.diagnostics.to_hdf5()
        logger.info(f"Exporting diagnostics for simulation {sim_id} to {h5path}")
        # Return as downloadable attachment
        return send_file(
            h5path,
            as_attachment=True,
            download_name=f"{sim_id}_diagnostics.h5",
            mimetype="application/octet-stream"
        )
    except SimulationError as e:
        logger.error(f"Simulation error: {e}")
        abort(404, description=str(e))
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        abort(500, description="An unexpected error occurred.")

@sock.route("/api/simulation_updates/<sim_id>")
def simulation_updates(ws, sim_id):
    """
    WebSocket endpoint streaming summary diagnostics at ~10 Hz.
    Sends the latest diagnostics.data[-1] JSON each interval.
    """
    try:
        sim = simulation_manager.get_simulation(sim_id)
        thread = simulation_manager.get_simulation_thread(sim_id)
        send_interval = app.config['TELEMETRY_INTERVAL']
        last_sent = time.time()

        # Stream while simulation is running
        while thread.is_alive():
            now = time.time()
            if now - last_sent >= send_interval:
                last_sent = now
                # Build summary payload
                if hasattr(sim.diagnostics, "data") and sim.diagnostics.data:
                    payload = sim.diagnostics.to_json()
                else:
                    payload = {"time": sim.current_time, "status": "running"}
                # Add telemetry
                payload = json.loads(payload)
                payload.update(json.loads(simulation_manager.get_telemetry(sim_id)))
                try:
                    ws.send(json.dumps(payload))
                except Exception as e:
                    logger.warning(f"Failed to send update for simulation {sim_id}: {e}. Closing WebSocket.")
                    break
            # short sleep to avoid busy‐wait
            time.sleep(0.01)

        # Final send after completion
        if hasattr(sim.diagnostics, "data") and sim.diagnostics.data:
            final = sim.diagnostics.to_json()
        else:
            final = {"time": sim.current_time, "status": "completed"}
        try:
            ws.send(json.dumps(final))
            logger.info(f"Final update sent for simulation {sim_id}.")
        except Exception as e:
            logger.warning(f"Failed to send final update for simulation {sim_id}: {e}")
        ws.close()
    except SimulationError as e:
        logger.error(f"Simulation error: {e}")
        ws.close()
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        ws.close()

# ——— Main Execution ———
def main():
    parser = argparse.ArgumentParser(description="DPF Simulator Server")
    parser.add_argument("--config-file", type=str, default="server_config.json",
                        help="Path to the server configuration file")
    args = parser.parse_args()

    try:
        config = load_config(args.config_file)
        app.config.update(config.dict())
    except ConfigurationError as e:
        logger.error(e)
        sys.exit(1)

    # Run Flask app on port 5000, allow threaded requests
    app.run(host=app.config['host'],
            port=app.config['port'],
            threaded=True)

if __name__ == "__main__":
    main()
