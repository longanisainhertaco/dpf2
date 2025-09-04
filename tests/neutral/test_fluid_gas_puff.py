import pytest

from dpf2.physics.neutral_gas import NeutralGasFluid
from dpf2.physics.hooks import neutral_density_source


def test_gas_puff_increases_density_then_ionizes():
    fluid = NeutralGasFluid(rho=0.0, volume=1.0, mass_flow_rate=1e-6, puff_start=0.0, puff_end=1.0)
    # During puff, density rises linearly
    rho1 = fluid.step(1.0, t=0.5, ionization_rate=0.0)
    assert rho1 == pytest.approx(1e-6)

    # After puff ends, only ionisation acts and density decays
    rho2 = fluid.step(1.0, t=1.5, ionization_rate=1.0)
    assert rho2 < rho1
    assert rho2 == pytest.approx(rho1 - rho1)


def test_neutral_density_source_with_puff():
    # With a puff active the source should be positive even with ionisation
    src = neutral_density_source(
        rho_n=1e-6,
        ionization_rate=0.1,
        t=0.5,
        puff_start=0.0,
        puff_end=1.0,
        mass_flow_rate=1e-6,
        volume=1.0,
    )
    assert src > 0
