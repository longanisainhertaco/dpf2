from dpf2.simulation.dpf_simulator_amrex_backend import _parse_cli


def test_parse_cli():
    args = _parse_cli(["config.json", "--diag-frequency", "7"])
    assert args.diag_frequency == 7
    assert args.config == "config.json"
