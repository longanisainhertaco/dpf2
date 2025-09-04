from click.testing import CliRunner
import pydantic
import pytest

if not hasattr(pydantic.BaseModel, "parse_obj"):  # pragma: no cover - compatibility
    pydantic.BaseModel.parse_obj = classmethod(lambda cls, d: cls(**d))
if not hasattr(pydantic.BaseModel, "model_validate"):  # pragma: no cover - compatibility
    pydantic.BaseModel.model_validate = classmethod(lambda cls, d, **_: cls.parse_obj(d))

from dpf2.validation_suite import load_validation_dataset, score_simulation
from dpf2.cli.main import main


def test_load_validation_dataset():
    data = load_validation_dataset("PF1000")
    assert "current" in data
    assert data["neutron_yield"][1][2] == pytest.approx(1e11)


def test_score_simulation_pass(tmp_path):
    ref = load_validation_dataset("PF1000")
    sim = {k: (v[0], v[1]) for k, v in ref.items()}
    res = score_simulation(
        sim,
        "PF1000",
        {"current": 0.1, "voltage": 0.1, "neutron_yield": 0.2},
    )
    assert res["passed"]
    assert res["overall"] == pytest.approx(1.0)


def test_cli_validate_help():
    runner = CliRunner()
    result = runner.invoke(main, ["validate", "--help"])
    assert result.exit_code == 0
    assert "Run a validation simulation" in result.output
