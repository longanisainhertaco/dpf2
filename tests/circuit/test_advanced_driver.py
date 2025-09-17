import random
import numpy as np
from types import SimpleNamespace

from dpf2.circuit_solver import run_circuit_simulation
from dpf2.circuit_config import (
    CircuitConfig,
    SegmentConfig,
    SwitchConfig,
    CrowbarStageConfig,
)


class DummyHallSolver:
    """Minimal plasma solver exposing ``plasma_inductance`` for feedback."""

    def __init__(self) -> None:
        self.calls = 0

    def step(self, state, dt, current, voltage):
        return state

    def coupling_interface(self):
        return SimpleNamespace(
            Lp=1e-6, emf=0.0, voltage=0.0, back_reaction=0.0, mutual_inductance=0.0
        )

    def plasma_inductance(self, state):
        self.calls += 1
        return 1e-6


def _make_config() -> CircuitConfig:
    base = CircuitConfig.with_defaults()
    return base.model_copy(
        update={
            "L_ext": 1.0,
            "R_ext": 1.0,
            "C_ext": 1.0,
            "V0": 1.0,
            "switch_delay": 0.0,
            "segments": [
                SegmentConfig(length=1.0, L=1.0, R=0.0, C=0.0, from_node=0, to_node=1)
            ],
            "switches": [
                SwitchConfig(
                    from_node=0,
                    to_node=1,
                    closed=False,
                    trigger_times=[50.0],
                    r_on=1.0,
                    r_off=1e6,
                )
            ],
            "crowbar_stages": [{"resistance": 0.1, "trigger": 100.0}],
            "trigger_jitter_stddev": 5.0,
        }
    )


def test_advanced_driver_features() -> None:
    cfg = _make_config()
    random.seed(0)
    solver1 = DummyHallSolver()
    t1, i1, _, _, _ = run_circuit_simulation(
        cfg, t_end=0.2, num_points=200, plasma_solver=solver1
    )
    assert solver1.calls > 0  # plasma inductance queried

    random.seed(1)
    solver2 = DummyHallSolver()
    t2, i2, _, _, _ = run_circuit_simulation(
        cfg, t_end=0.2, num_points=200, plasma_solver=solver2
    )

    # Different random seeds should yield distinct current traces due to jitter
    assert not np.allclose(i1, i2)
