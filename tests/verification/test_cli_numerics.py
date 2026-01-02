import sys

import h5py_stub as h5py
from click.testing import CliRunner

# Provide stub for optional dependency
sys.modules["h5py"] = h5py

from dpf2.cli import main as cli_main


def test_verify_numerics_cli(tmp_path):
    out_file = tmp_path / "verify.h5"
    runner = CliRunner()
    result = runner.invoke(cli_main.main, ["verify-numerics", "--output", str(out_file)])
    assert result.exit_code == 0
    assert "Numerics verification results" in result.output
    assert out_file.exists()


def test_verify_numerics_json(tmp_path):
    out_file = tmp_path / "verify.h5"
    runner = CliRunner()
    result = runner.invoke(
        cli_main.main,
        [
            "verify-numerics",
            "--json",
            "--sizes",
            "8",
            "--sizes",
            "16",
            "--output",
            str(out_file),
        ],
    )
    assert result.exit_code == 0
    assert "observed_order" in result.output
