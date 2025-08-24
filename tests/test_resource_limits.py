import os
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
            def errorhandler(self, *a, **k):
                def decorator(f):
                    return f
                return decorator
            def test_client(self):
                return None

        class DummySock:
            def __init__(self, *a, **k):
                self.args = a
                self.kwargs = k
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
                self.config = {}
        sys.modules['utils'] = types.SimpleNamespace(FieldManager=DummyFieldManager)
        import importlib.util, pathlib
        from werkzeug.security import generate_password_hash
        spec = importlib.util.spec_from_file_location(
            'dpf_simulator_server',
            pathlib.Path('src/dpf2/simulation/dpf_simulator_server.py'),
        )
        server = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(server)

        class DummySimulation:
            def __init__(self, config, field_manager=None):
                self.config = config
                self.field_manager = field_manager
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
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
    )
    assert proc.returncode != 0
    assert 'SimulationError' in proc.stderr


def test_cpu_time_limit_violation_terminates_process():
    """Ensure exceeding the CPU time limit aborts the simulation process."""
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
            def errorhandler(self, *a, **k):
                def decorator(f):
                    return f
                return decorator
            def test_client(self):
                return None

        class DummySock:
            def __init__(self, *a, **k):
                self.args = a
                self.kwargs = k
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
                self.config = {}
        sys.modules['utils'] = types.SimpleNamespace(FieldManager=DummyFieldManager)
        import importlib.util, pathlib
        from werkzeug.security import generate_password_hash
        spec = importlib.util.spec_from_file_location(
            'dpf_simulator_server',
            pathlib.Path('src/dpf2/simulation/dpf_simulator_server.py'),
        )
        server = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(server)

        class DummySimulation:
            def __init__(self, config, field_manager=None):
                self.config = config
                self.field_manager = field_manager
            def run(self):
                while True:
                    pass

        server.DPFSimulation = DummySimulation

        app = server.app
        app.config['ADMIN_USERNAME'] = 'admin'
        app.config['ADMIN_PASSWORD_HASH'] = generate_password_hash('secret')
        app.config['MAX_SIMULTANEOUS_SIMULATIONS'] = 1
        app.config['TELEMETRY_INTERVAL'] = 0.01
        app.config['CPU_TIME_LIMIT'] = 1
        app.config['MEMORY_LIMIT'] = 1024 * 1024 * 1024

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
        # The CPU limit will terminate the process before we join the thread.
        thread = server.simulation_manager.get_simulation_thread(sim_id)
        thread.join()

        if sim_id not in server.simulation_manager.sim_errors:
            raise RuntimeError('Simulation should have failed')
        raise server.SimulationError('limit exceeded')
        """
    )
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=10,
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
    )
    assert proc.returncode != 0
