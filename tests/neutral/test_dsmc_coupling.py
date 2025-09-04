import numpy as np
import pytest

from dpf2.neutral.dsmc import DSMC
from dpf2.coupled_models import NeutralPlasmaCoupler


class _DummyPlasmaSolver:
    def __init__(self) -> None:
        self.received_density = None

    def run(self, *, neutral_density: float):
        self.received_density = neutral_density
        # Return a simple payload so the coupler can forward it
        return {"nd": neutral_density}


def test_dsmc_coupling_and_knudsen(tmp_path):
    # Load cross sections from the LXCat manifest entry
    dsmc = DSMC.from_lxcat("Ar", "dummy", knudsen_number=0.1, velocities=[1000.0, -1000.0])
    dsmc2 = DSMC.from_lxcat("Ar", "dummy", knudsen_number=0.2)

    n1 = dsmc.compute_neutral_density()
    n2 = dsmc2.compute_neutral_density()
    # Knudsen number enters inversely in the density
    assert n2 == pytest.approx(n1 / 2)

    plasma = _DummyPlasmaSolver()
    coupler = NeutralPlasmaCoupler(dsmc, plasma)
    result = coupler.run(dt=1e-6)

    assert plasma.received_density == pytest.approx(n1)
    assert result.neutral_density == pytest.approx(n1)
    assert result.plasma["nd"] == pytest.approx(n1)
