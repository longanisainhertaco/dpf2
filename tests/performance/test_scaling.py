"""Basic performance regression tests for simulation scaling."""

from __future__ import annotations

import time

import numpy as np

from concurrent.futures import ThreadPoolExecutor

from dpf2.dpf_config import DPFConfig
from dpf2.simulation_engine import SimulationEngine


def _make_config(time_end: float) -> DPFConfig:
    cfg = DPFConfig.with_defaults()
    sc = cfg.simulation_control.model_copy(update={"time_end": time_end})
    return cfg.model_copy(update={"simulation_control": sc})


def test_problem_size_scaling():
    """Larger problems should take at least as long as smaller ones."""

    small_cfg = _make_config(1e-7)
    large_cfg = _make_config(5e-7)

    start = time.perf_counter()
    SimulationEngine(small_cfg).run()
    small_time = time.perf_counter() - start

    start = time.perf_counter()
    SimulationEngine(large_cfg).run()
    large_time = time.perf_counter() - start

    assert large_time >= small_time


def test_thread_scaling_consistency():
    """Threaded execution should produce the same result as serial."""

    cfg = _make_config(5e-7)
    serial = SimulationEngine(cfg, num_threads=1).run()
    threaded = SimulationEngine(cfg, num_threads=2).run()

    assert np.allclose(serial.current, threaded.current)


def test_parallel_speedup():
    """Running two simulations in parallel should not be slower than serial."""

    cfg = _make_config(5e-7)

    def run():
        SimulationEngine(cfg).run()

    start = time.perf_counter()
    for _ in range(2):
        run()
    serial_time = time.perf_counter() - start

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as ex:
        list(ex.map(lambda _: run(), range(2)))
    parallel_time = time.perf_counter() - start

    # Allow some leeway for scheduling noise
    assert parallel_time <= serial_time * 1.5

