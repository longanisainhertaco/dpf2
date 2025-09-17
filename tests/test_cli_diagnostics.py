import json

try:  # pragma: no cover - prefer real h5py
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    import h5py_stub as h5py  # type: ignore
import json
from click.testing import CliRunner

from dpf2.cli.main import diagnostics


def _write_calibration(path, group):
    with h5py.File(path, "w") as fh:
        grp = fh.require_group(group)
        grp.create_dataset("time", data=[0.0, 1.0])
        grp.create_dataset("response", data=[1.0, 0.0])


def _write_history(path):
    history = [
        {"Lp": 0.0, "emf": 0.0, "current": 0.0, "voltage": 1.0},
        {"Lp": 0.0, "emf": 0.0, "current": 1.0, "voltage": 0.5},
        {"Lp": 0.0, "emf": 0.0, "current": 2.0, "voltage": 0.0},
    ]
    path.write_text(json.dumps(history))


def test_cli_outputs_current_and_voltage(tmp_path):
    hist_path = tmp_path / "history.json"
    _write_history(hist_path)

    runner = CliRunner()
    result = runner.invoke(
        diagnostics,
        ["--history", str(hist_path), "--current", "--voltage"],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["current"] == [0.0, 1.0, 2.0]
    assert data["voltage"] == [1.0, 0.5, 0.0]


def test_cli_accepts_calibration_files(tmp_path):
    hist_path = tmp_path / "history.json"
    _write_history(hist_path)
    rog_cal = tmp_path / "rog.h5"
    bdot_cal = tmp_path / "bdot.h5"
    sxr_cal = tmp_path / "sxr.h5"
    tof_cal = tmp_path / "tof.h5"
    _write_calibration(rog_cal, "rogowski")
    _write_calibration(bdot_cal, "bdot")
    _write_calibration(sxr_cal, "sxr")
    _write_calibration(tof_cal, "tof")

    runner = CliRunner()
    result = runner.invoke(
        diagnostics,
        [
            "--history",
            str(hist_path),
            "--rogowski-cal",
            str(rog_cal),
            "--bdot-cal",
            str(bdot_cal),
            "--sxr-cal",
            str(sxr_cal),
            "--tof-cal",
            str(tof_cal),
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data == {}


def test_cli_calibration_file_missing(tmp_path):
    hist_path = tmp_path / "history.json"
    _write_history(hist_path)

    runner = CliRunner()
    missing = tmp_path / "missing.h5"
    result = runner.invoke(
        diagnostics,
        ["--history", str(hist_path), "--rogowski", "--rogowski-cal", str(missing)],
    )
    assert result.exit_code != 0
    assert "does not exist" in result.output


def test_cli_help_shows_calibration_options():
    runner = CliRunner()
    result = runner.invoke(diagnostics, ["--help"])
    assert "--rogowski-cal" in result.output
    assert "--bdot-cal" in result.output
    assert "--sxr-cal" in result.output
    assert "--tof-cal" in result.output
    assert "--anisotropy-plot" in result.output


def test_cli_anisotropy_plot(tmp_path):
    hist_path = tmp_path / "history.json"
    _write_history(hist_path)
    runner = CliRunner()
    result = runner.invoke(
        diagnostics,
        ["--history", str(hist_path), "--anisotropy-plot"],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "anisotropy" in data
