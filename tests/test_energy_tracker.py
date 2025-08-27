import numpy as np
from types import SimpleNamespace

from dpf2.simulation_engine import SimulationEngine


def test_energy_conservation(tmp_path):
    cfg = SimpleNamespace()
    cfg.circuit_config = SimpleNamespace(L_ext=1e-6, R_ext=0.0, C_ext=1e-6, V0=20e3)
    cfg.simulation_control = SimpleNamespace(time_start=0.0, time_end=1e-6, min_dt=1e-8)

    def resolve_defaults():
        return cfg

    cfg.resolve_defaults = resolve_defaults

    engine = SimulationEngine(cfg)
    results = engine.run(energy_csv=tmp_path / "energy.csv", energy_tol=1e-3)
    assert (tmp_path / "energy.csv").exists()
    energies = results.energies
    assert energies is not None
    total = energies["total"]
    assert np.isclose(total[0], total[-1], rtol=1e-3)
