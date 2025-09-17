import sys
import types
import threading
import queue
import time
from unittest.mock import patch

# Stub optional dependencies required for import
sys.modules.setdefault("amrex", types.ModuleType("amrex"))
sys.modules.setdefault("adios2", types.ModuleType("adios2"))
import h5py_stub as h5py

sys.modules.setdefault("h5py", h5py)

numba_stub = types.ModuleType("numba")
numba_stub.njit = lambda *a, **k: (lambda f: f)
numba_stub.prange = range
sys.modules.setdefault("numba", numba_stub)

scipy_interp = types.ModuleType("scipy.interpolate")
scipy_interp.RegularGridInterpolator = lambda *a, **k: None
scipy_module = types.ModuleType("scipy")
sys.modules.setdefault("scipy", scipy_module)
sys.modules.setdefault("scipy.interpolate", scipy_interp)

from dpf2.simulation.radiation_model import RadiationModel


class DummyWriter:
    def __init__(self):
        self.closed = False

    def Close(self):
        self.closed = True


class DummySocket:
    def __init__(self):
        self.closed = False

    def sendall(self, data):
        pass

    def shutdown(self, how):
        pass

    def close(self):
        self.closed = True


def test_radiation_model_thread_cleanup():
    pre_threads = set(threading.enumerate())
    rm = RadiationModel.__new__(RadiationModel)
    rm._q = queue.Queue()
    rm.telemetry_port = 0
    rm.writer = DummyWriter()

    dummy_sock = DummySocket()
    with patch(
        "dpf2.simulation.radiation_model.socket.create_connection",
        return_value=dummy_sock,
    ):
        rm._t_thread = threading.Thread(target=rm._telemetry_loop)
        rm._t_thread.daemon = True
        rm._t_thread.start()
        for _ in range(100):
            if getattr(rm, "_telemetry_conn", None):
                break
            time.sleep(0.01)
        rm.finalize()

    post_threads = set(threading.enumerate())
    assert pre_threads == post_threads
    assert dummy_sock.closed
    assert rm.writer.closed
    assert not rm._t_thread.is_alive()
