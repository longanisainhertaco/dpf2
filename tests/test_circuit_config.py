import pytest
from pathlib import Path

from circuit_config import CircuitConfig


def test_waveform_path_validation_allows_uri():
    data = CircuitConfig.with_defaults().model_dump(by_alias=True)
    data["waveformProfile"] = None
    data["waveformProfilePath"] = "file:///tmp/test.csv"
    cfg = CircuitConfig.model_validate(data)
    assert str(cfg.waveform_profile_path).startswith("file:")


def test_missing_feedback_delay_for_multi_bank():
    data = CircuitConfig.with_defaults().model_dump(by_alias=True)
    data["switchingModel"] = "multi-bank"
    data["switchFeedbackDelayNs"] = None
    with pytest.raises(ValueError):
        CircuitConfig.model_validate(data)


def test_waveform_conflict_raises_without_resolution():
    data = CircuitConfig.with_defaults().model_dump(by_alias=True)
    data["waveformProfilePath"] = "/tmp/test.csv"
    data["waveformConflictResolution"] = "invalid"
    with pytest.raises(ValueError):
        CircuitConfig.model_validate(data)


def test_override_inductive_limit_bounds():
    data = CircuitConfig.with_defaults().model_dump(by_alias=True)
    data["overrideInductiveVoltageLimit"] = -1.0
    with pytest.raises(ValueError):
        CircuitConfig.model_validate(data)


def test_abort_without_failure_conditions_errors():
    data = CircuitConfig.with_defaults().model_dump(by_alias=True)
    data["abortOnNoCurrent"] = True
    data["failureConditions"] = {"voltage_max": 100.0}
    with pytest.raises(ValueError):
        CircuitConfig.model_validate(data)
