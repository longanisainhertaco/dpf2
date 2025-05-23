import pytest

try:
    import yaml  # type: ignore
    YAML_AVAILABLE = True
except Exception:
    YAML_AVAILABLE = False

from metadata import Metadata


def test_surrogate_model_requires_metadata():
    data = Metadata.with_defaults().model_dump(by_alias=True)
    data["useSurrogateModel"] = "model.pt"
    with pytest.raises(ValueError):
        Metadata.model_validate(data)


def test_hash_and_summary_id_are_consistent():
    cfg = Metadata.with_defaults()
    assert cfg.summary_id == cfg.hash_metadata()[:6]


def test_parameter_bounds_are_valid():
    data = Metadata.with_defaults().model_dump(by_alias=True)
    data["mlParameterBounds"] = {"V0": [10.0, 5.0]}
    with pytest.raises(ValueError):
        Metadata.model_validate(data)


def test_restart_hash_format():
    data = Metadata.with_defaults().model_dump(by_alias=True)
    data["restartConfigHash"] = "zzzzzz"
    with pytest.raises(ValueError):
        Metadata.model_validate(data)


def test_yaml_round_trip_and_summary(tmp_path):
    cfg = Metadata.with_defaults()
    if not YAML_AVAILABLE:
        with pytest.raises(Exception):
            __import__("yaml")
        return

    import yaml  # type: ignore

    yaml_path = tmp_path / "m.yml"
    yaml.safe_dump({"metadata": cfg.model_dump(by_alias=True)}, open(yaml_path, "w"))
    loaded = yaml.safe_load(open(yaml_path))
    cfg2 = Metadata.model_validate(loaded["metadata"])
    assert cfg2.run_uuid == cfg.run_uuid
    assert cfg2.summary_id == cfg.summary_id
    summary = cfg.summarize()
    assert "Schema:" in summary and "Surrogate:" in summary
