import pytest


pytest.importorskip("pytest_benchmark")

from dpf2.physics.neutral_gas import NeutralGasFluid


@pytest.mark.benchmark
@pytest.mark.parametrize("steps", [100])
def test_neutral_transport_benchmark(benchmark, steps):
    fluid = NeutralGasFluid(rho=0.0, volume=1.0, mass_flow_rate=1e-6, puff_start=0.0, puff_end=1e-3)

    def run():
        t = 0.0
        for _ in range(steps):
            fluid.step(1e-5, t, ionization_rate=0.0)
            t += 1e-5
        return fluid.rho

    result = benchmark(run)
    assert result > 0.0
