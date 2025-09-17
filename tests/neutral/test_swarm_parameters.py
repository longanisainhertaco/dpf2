from pathlib import Path

from dpf2.physics.neutral_gas import validate_swarm_parameters, compute_swarm_parameters
from dpf2.neutral.dsmc import load_lxcat_table


def test_swarm_parameter_validation():
    table = load_lxcat_table(Path("tests/neutral/lxcat_dummy.csv"))
    params = compute_swarm_parameters(table)
    # validate against the computed values to ensure routine works
    validated = validate_swarm_parameters(
        Path("tests/neutral/lxcat_dummy.csv"),
        {
            "mobility": params.mobility,
            "diffusion": params.diffusion,
        },
    )
    assert validated == params
