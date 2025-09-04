import h5py_stub  # register h5py stub
from click.testing import CliRunner
from dpf2.core.config import DPFConfig
from dpf2.cli.main import main


def test_lab_mode_runs_multiple_shots(tmp_path):
    cfg = DPFConfig()
    cfg.to_file(tmp_path / "cfg.json")
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--lab-mode",
            "simulate",
            "--config",
            str(tmp_path / "cfg.json"),
            "--output",
            str(tmp_path / "out"),
            "--shots",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    shot0 = tmp_path / "out" / "shot_000"
    shot1 = tmp_path / "out" / "shot_001"
    assert (shot0 / "manifest.json").exists()
    assert (shot1 / "manifest.json").exists()
    assert any(p.suffix == ".h5" for p in shot0.iterdir())
