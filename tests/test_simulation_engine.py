import logging

import numpy as np
import pydantic

if not hasattr(pydantic.BaseModel, "parse_obj"):  # pragma: no cover - compatibility
    pydantic.BaseModel.parse_obj = classmethod(lambda cls, d: cls(**d))
if not hasattr(pydantic.BaseModel, "model_validate"):  # pragma: no cover - compatibility
    pydantic.BaseModel.model_validate = classmethod(lambda cls, d, **_: cls.parse_obj(d))

from dpf2.core_schema import RadiationModel
from dpf2.dpf_config import DPFConfig
from dpf2.simulation_engine import SimulationEngine, EnsembleResults
from dpf2.physics.energy import EnergyTracker
from dpf2.experimental_variability import ExperimentalVariabilityModel, MonteCarloVariability


SIM_TIME = 1e-8


def test_engine_runs():
    cfg = DPFConfig.with_defaults()
    sim_ctrl = cfg.simulation_control.model_copy(
        update={"time_end": SIM_TIME, "min_dt": SIM_TIME / 5, "max_steps": 10}
    )
    cfg = cfg.model_copy(update={"simulation_control": sim_ctrl})
    engine = SimulationEngine(cfg)
    results = engine.run()
    # basic shape checks
    assert results.current.size > 0
    assert results.time.shape == results.current.shape
    assert results.voltage.shape == results.current.shape
    assert results.radius.shape == results.current.shape
    assert results.temperature.shape == results.current.shape
    assert results.pressure.shape == results.current.shape

    # time should start at zero and be monotonically increasing
    assert np.isclose(results.time[0], 0.0)
    assert np.all(np.diff(results.time) > 0)
    assert results.time[-1] <= 1e-6

    # physical quantities should be finite and non-negative
    assert np.all(np.isfinite(results.temperature))
    assert np.all(results.temperature >= 0)
    assert np.all(results.pressure >= 0)
    assert np.all(results.radius >= 0)

    # neutron yield should be non-negative
    assert results.neutron_yield >= 0.0

    # simple oscillation check in current waveform when signal is non-zero
    if np.any(np.abs(results.current) > 0):
        assert np.any(np.diff(np.sign(results.current)) != 0)

    # ensure conversion to dictionary preserves keys
    as_dict = results.to_dict()
    for key in [
        "time",
        "current",
        "voltage",
        "pinch_radius",
        "temperature",
        "pressure",
        "neutron_yield",
    ]:
        assert key in as_dict


def test_engine_ensemble_statistics():
    cfg = DPFConfig.with_defaults()
    sim_ctrl = cfg.simulation_control.model_copy(
        update={"time_end": SIM_TIME, "min_dt": SIM_TIME / 5, "max_steps": 10}
    )
    cfg = cfg.model_copy(update={"simulation_control": sim_ctrl})
    var_cfg = ExperimentalVariabilityModel.with_defaults().model_copy(
        update={
            "pressure_jitter_pct": 5.0,
            "stochastic_run_id": 1,
            "per_field_distribution_params": {
                "capacitor_voltage": {"jitter_pct": 1.0},
            },
        }
    )
    variability = MonteCarloVariability(var_cfg, realizations=2)
    engine = SimulationEngine(cfg)
    results = engine.run(variability=variability)
    assert isinstance(results, EnsembleResults)
    assert results.current_mean.shape == results.time.shape
    assert results.current_std.shape == results.time.shape
    assert results.radius_mean.shape == results.time.shape
    assert results.radius_std.shape == results.time.shape
    assert results.temperature_mean.shape == results.time.shape
    assert results.temperature_std.shape == results.time.shape
    assert results.pressure_mean.shape == results.time.shape
    assert results.pressure_std.shape == results.time.shape
    assert results.neutron_yield_mean >= 0.0
    assert results.neutron_yield_std >= 0.0
    if results.axial_position_mean is not None:
        assert results.axial_position_mean.shape == results.time.shape
        assert results.axial_position_std is not None
        assert results.axial_position_std.shape == results.time.shape


def test_engine_progress_callback():
    cfg = DPFConfig.with_defaults()
    cfg = cfg.model_copy(
        update={
            "simulation_control": cfg.simulation_control.model_copy(
                update={"time_end": SIM_TIME, "min_dt": SIM_TIME / 5, "max_steps": 10}
            )
        }
    )
    engine = SimulationEngine(cfg)
    calls: list[float] = []

    def cb(step: int, time: float, current: float, voltage: float) -> None:
        calls.append(time)

    engine.run(progress_cb=cb)
    assert len(calls) > 0
    assert all(calls[i] <= calls[i + 1] for i in range(len(calls) - 1))


def test_engine_warns_when_radiation_requested(caplog):
    cfg = DPFConfig.with_defaults()
    cfg.physics_models.radiation_model = RadiationModel.BREMSSTRAHLUNG
    cfg.physics_models.sxr_bandpass_nm = (0.5, 1.0)
    engine = SimulationEngine(cfg)
    tracker = EnergyTracker()
    tracker.add()
    with caplog.at_level(logging.WARNING):
        engine._handle_radiation_coupling(tracker)
        engine._handle_radiation_coupling(tracker)
    warnings = [rec.message for rec in caplog.records if rec.levelno >= logging.WARNING]
    matches = [msg for msg in warnings if "Radiation coupling requested" in msg]
    assert len(matches) == 1
    assert tracker.radiative[-1] == 0.0
