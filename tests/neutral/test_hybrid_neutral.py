from pathlib import Path

from dpf2.neutral.hybrid import HybridNeutralModel
from dpf2.axial_sheath import AxialSheathModel


def test_hybrid_neutral_couples_to_sheath_and_validates():
    model = HybridNeutralModel.from_lxcat(
        Path("tests/neutral/lxcat_dummy.csv"),
        knudsen_number=0.2,
        volume=1.0,
        puff_rate=1e18,
    )
    sheath = AxialSheathModel(
        area=1.0,
        mass=1.0,
        length=0.05,
        upstream_density=0.0,
        upstream_pressure=0.0,
    )
    density = model.couple_sheath(sheath, dt=1e-6, plasma_density=1e18)
    assert density > 0.0
    assert sheath.upstream_density == density
    params = model.validate_swarm({"mobility": model.swarm.mobility, "diffusion": model.swarm.diffusion})
    assert params.mobility == model.swarm.mobility
