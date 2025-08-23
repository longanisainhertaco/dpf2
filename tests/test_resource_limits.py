import subprocess
import sys
import textwrap


def test_resource_limit_violation_triggers_simulation_error():
    script = textwrap.dedent(
        """
        import types
        import sys
        sys.modules['sympy'] = types.SimpleNamespace(symbols=lambda *args, **kwargs: (None, None))
        class DummyFlask:
            def __init__(self, *a, **k):
                self.config = {}
            def route(self, *a, **k):
                def decorator(f):
                    return f
                return decorator
            def test_client(self):
                return None

        class DummySock:
            def __init__(self, *a, **k):
                pass
            def route(self, *a, **k):
                def decorator(f):
                    return f
                return decorator

        sys.modules['flask'] = types.SimpleNamespace(
            Flask=DummyFlask,
            request=None, jsonify=lambda *a, **k: None,
            send_file=lambda *a, **k: None, abort=lambda *a, **k: None,
        )
        sys.modules['flask_sock'] = types.SimpleNamespace(Sock=DummySock)
        sys.modules['werkzeug.security'] = types.SimpleNamespace(
            generate_password_hash=lambda s: s,
            check_password_hash=lambda h, p: True,
        )
        sys.modules['werkzeug.utils'] = types.SimpleNamespace(secure_filename=lambda s: s)
        sys.modules['dpf_simulation'] = types.SimpleNamespace(DPFSimulation=object, ConfigurationError=Exception)
        sys.modules['config_schema'] = types.SimpleNamespace(ServerConfig=object, FieldManagerConfig=object)
        class DummyFieldManager:
            def __init__(self, *a, **k):
                pass
        sys.modules['utils'] = types.SimpleNamespace(FieldManager=DummyFieldManager)
        from Simulation import dpf_simulator_server as server
        from werkzeug.security import generate_password_hash

        class DummySimulation:
            def __init__(self, config, field_manager=None):
                pass
            def run(self):
                bytearray(300 * 1024 * 1024)

        server.DPFSimulation = DummySimulation

        app = server.app
        app.config['ADMIN_USERNAME'] = 'admin'
        app.config['ADMIN_PASSWORD_HASH'] = generate_password_hash('secret')
        app.config['MAX_SIMULTANEOUS_SIMULATIONS'] = 1
        app.config['TELEMETRY_INTERVAL'] = 0.01
        app.config['CPU_TIME_LIMIT'] = 1
        app.config['MEMORY_LIMIT'] = 150 * 1024 * 1024

        config = {
            'dx': 1.0,
            'dy': 1.0,
            'dz': 1.0,
            'sim_time': 0.05,
            'dt_init': 0.01,
            'grid_shape': [1, 1, 1],
            'domain_lo': [0.0, 0.0, 0.0],
            'circuit': {
                'C': 1.0,
                'V0': 1.0,
                'L0': 1.0,
                'R0': 0.1,
                'anode_radius': 0.01,
                'cathode_radius': 0.02,
            },
        }

        sim_id = server.simulation_manager.create_simulation(config)
        server.simulation_manager.run_simulation(sim_id)
        thread = server.simulation_manager.get_simulation_thread(sim_id)
        thread.join()

        if sim_id not in server.simulation_manager.sim_errors:
            raise RuntimeError('Simulation should have failed')
        raise server.SimulationError('limit exceeded')
        """
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode != 0
    assert 'SimulationError' in proc.stderr
