import importlib
import types
import sys
import logging
import importlib.util
from pathlib import Path

import pytest


def test_invalid_sim_parameters_warn(monkeypatch, caplog):
    """_validate_sim_parameters warns and aborts on bad input."""
    # Stub external modules required for import
    flask_stub = types.ModuleType("flask")
    class DummyFlask:
        def __init__(self, *args, **kwargs):
            pass
        def route(self, *args, **kwargs):
            def decorator(f):
                return f
            return decorator
        config = {}
    def abort(code, description=None):
        raise RuntimeError(description or str(code))
    flask_stub.Flask = DummyFlask
    flask_stub.request = None
    flask_stub.jsonify = lambda *a, **k: {}
    flask_stub.send_file = lambda *a, **k: None
    flask_stub.abort = abort
    monkeypatch.setitem(sys.modules, "flask", flask_stub)

    flask_sock_stub = types.ModuleType("flask_sock")
    class Sock:
        def __init__(self, *args, **kwargs):
            pass
        def route(self, *args, **kwargs):
            def decorator(f):
                return f
            return decorator
    flask_sock_stub.Sock = Sock
    monkeypatch.setitem(sys.modules, "flask_sock", flask_sock_stub)

    dpf_sim_stub = types.ModuleType("dpf_simulation")
    dpf_sim_stub.DPFSimulation = type("DPFSimulation", (), {})
    class ConfigError(Exception):
        pass
    dpf_sim_stub.ConfigurationError = ConfigError
    monkeypatch.setitem(sys.modules, "dpf_simulation", dpf_sim_stub)

    config_schema_stub = types.ModuleType("config_schema")
    config_schema_stub.ServerConfig = type("ServerConfig", (), {})
    config_schema_stub.FieldManagerConfig = type("FieldManagerConfig", (), {})
    monkeypatch.setitem(sys.modules, "config_schema", config_schema_stub)

    pydantic_stub = types.ModuleType("pydantic")
    pd_dataclasses = types.ModuleType("pydantic.dataclasses")
    import dataclasses
    pd_dataclasses.dataclass = dataclasses.dataclass
    pydantic_stub.dataclasses = pd_dataclasses
    monkeypatch.setitem(sys.modules, "pydantic", pydantic_stub)
    monkeypatch.setitem(sys.modules, "pydantic.dataclasses", pd_dataclasses)

    werk_stub = types.ModuleType("werkzeug")
    werk_sec = types.ModuleType("werkzeug.security")
    werk_sec.generate_password_hash = lambda x: x
    werk_sec.check_password_hash = lambda h, p: True
    werk_utils = types.ModuleType("werkzeug.utils")
    werk_utils.secure_filename = lambda x: x
    werk_stub.security = werk_sec
    werk_stub.utils = werk_utils
    monkeypatch.setitem(sys.modules, "werkzeug", werk_stub)
    monkeypatch.setitem(sys.modules, "werkzeug.security", werk_sec)
    monkeypatch.setitem(sys.modules, "werkzeug.utils", werk_utils)

    sympy_stub = types.ModuleType("sympy")
    sympy_stub.symbols = lambda *args, **kwargs: (None, None)
    monkeypatch.setitem(sys.modules, "sympy", sympy_stub)

    utils_stub = types.ModuleType("utils")
    utils_stub.FieldManager = type("FieldManager", (), {})
    monkeypatch.setitem(sys.modules, "utils", utils_stub)

    module_path = Path(__file__).resolve().parent.parent / "src/dpf2/simulation/dpf_simulator_server.py"
    spec = importlib.util.spec_from_file_location("server_mod", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    bad = {"dx": "bad", "sim_time": "1", "grid_shape": [1, 2, 3]}
    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError):
            mod._validate_sim_parameters(bad)

    assert "Invalid simulation parameters" in caplog.text
