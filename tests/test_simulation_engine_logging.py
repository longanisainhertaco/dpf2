import sys
import types
from unittest.mock import patch

import pytest
import numpy as np

from dpf2.dpf_config import DPFConfig
from dpf2.simulation_engine import SimulationEngine, logger


def _stub_matplotlib(monkeypatch):
    import types as _types

    matplotlib = _types.ModuleType("matplotlib")
    matplotlib.use = lambda *a, **k: None
    plt = _types.SimpleNamespace(
        figure=lambda *a, **k: None,
        loglog=lambda *a, **k: None,
        xlabel=lambda *a, **k: None,
        ylabel=lambda *a, **k: None,
        title=lambda *a, **k: None,
        savefig=lambda *a, **k: None,
        close=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", plt)


def test_generate_convergence_plot_logs_errors(monkeypatch):
    _stub_matplotlib(monkeypatch)
    cfg = DPFConfig.with_defaults()
    engine = SimulationEngine(cfg)

    class DummySolver:
        nx = ny = nz = 1

        def run_convergence_study(self, resolutions):
            raise ValueError("boom")

    solver = DummySolver()

    with patch.object(logger, "exception") as mock_exc:
        engine._generate_convergence_plot(solver)
        mock_exc.assert_called_once()


def test_threshold_logging(monkeypatch):
    cfg = DPFConfig.with_defaults()
    sc = cfg.simulation_control
    cfg = cfg.model_copy(
        update={
            "simulation_control": sc.model_copy(
                update={"time_end": sc.time_start + 1e-9}
            )
        }
    )
    engine = SimulationEngine(cfg)

    t_end = cfg.simulation_control.time_end

    class DummyCircuit:
        def __init__(self):
            self.voltages = [0.0]
            self.currents = [0.0]
            self.time = [t_end]
            self.circuit = types.SimpleNamespace(L=1.0, R=1.0, C=1.0, V0=1.0)

        def step(self, *args, **kwargs):
            return types.SimpleNamespace(current=0.0, voltage=0.0)

    monkeypatch.setattr(engine, "_setup_circuit", lambda: DummyCircuit())

    import dpf2.diagnostics.thresholds as thresholds
    import dpf2.pinch_models as pinch_models

    def bad_debye(*args, **kwargs):
        raise ValueError("bad")

    monkeypatch.setattr(thresholds, "compute_debye_length", bad_debye)

    monkeypatch.setattr(
        pinch_models.AnalyticPinchModel,
        "run",
        lambda self, t, current: types.SimpleNamespace(
            time=t,
            radius=np.zeros_like(t),
            temperature=np.zeros_like(t),
            pressure=np.zeros_like(t),
            neutron_yield=0.0,
            axial_position=np.zeros_like(t),
        ),
    )

    class DummyPICSolver:
        c = 1.0
        dt = 1e-9
        dx = dy = dz = 1.0
        nx = ny = nz = 1
        species = {}

        def step(self, state, dt, current, voltage):
            return state

        def coupling_interface(self):
            return types.SimpleNamespace(Lp=0.0)

    module = types.ModuleType("dpf2.simulation.pic_solver")
    module.PICSolver = DummyPICSolver
    monkeypatch.setitem(sys.modules, "dpf2.simulation.pic_solver", module)

    plasma_solver = DummyPICSolver()

    with patch.object(logger, "exception") as mock_exc:
        engine.run(plasma_solver=plasma_solver)
        mock_exc.assert_called_once()
