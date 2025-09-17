import pytest

from dpf2.neutral.dsmc import DSMC
from dpf2.materials.sputtering import Species, sputter_flux
from dpf2.materials.models import ImpurityState


def test_neutral_plasma_coupling_with_puff():
    dsmc = DSMC.from_lxcat(
        "Ar",
        "dummy",
        knudsen_number=0.1,
        velocities=[0.0],
        puff_start=0.0,
        puff_end=1.0,
        puff_rate=1.0,
    )
    baseline = dsmc.density
    n1 = dsmc.run(1.0, t=0.5, plasma_density=0.0)
    assert n1 == pytest.approx(baseline + 1.0)
    n2 = dsmc.run(1.0, t=1.5, plasma_density=0.5)
    assert n2 == pytest.approx(n1 - 0.5)


def test_sputtering_impurity_state():
    projectile = Species("D", 1, 2.0)
    target = Species("C", 6, 12.0)
    flux = sputter_flux(projectile, target, ion_flux=1.0, energy_eV=1000.0)
    assert target.name in flux and flux[target.name] > 0
    state = ImpurityState()
    state.apply_sources(flux, dt=1.0)
    assert state.densities[target.name] == pytest.approx(flux[target.name])
