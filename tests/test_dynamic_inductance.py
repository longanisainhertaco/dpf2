import numpy as np

from rlc_solver import dynamic_inductance, dynamic_resistance
from dpf_config import DPFConfig
from dpf2.simulation_engine import SimulationEngine


def test_dynamic_inductance_and_resistance_trends():
    r = np.array([1e-2, 5e-3])
    L = dynamic_inductance(r, cathode_radius=2e-2, length=0.1)
    # smaller radius -> larger inductance
    assert L[1] > L[0]

    radius = np.array([1e-3, 1e-3])
    density = np.array([1e-3, 2e-3])
    temperature = np.array([1e5, 1e5])
    R = dynamic_resistance(radius, density, temperature, length=0.1)
    assert R[1] < R[0]

    # higher temperature should lower resistance
    R_t = dynamic_resistance(radius, np.array([1e-3, 1e-3]), np.array([1e5, 2e5]), length=0.1)
    assert R_t[1] < R_t[0]


def test_simulation_engine_provides_density():
    cfg = DPFConfig.with_defaults()
    cfg = cfg.model_copy(update={
        "simulation_control": cfg.simulation_control.model_copy(update={"time_end": 1e-6})
    })
    engine = SimulationEngine(cfg)
    results = engine.run()
    assert results.density.shape == results.radius.shape
