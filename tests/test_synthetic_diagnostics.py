import json
import pytest

from synthetic_diagnostics import SyntheticDiagnostics, SyntheticInstrument

try:
    import yaml  # type: ignore
    YAML_AVAILABLE = True
except Exception:
    YAML_AVAILABLE = False


def base_data():
    return SyntheticDiagnostics.with_defaults().model_dump(by_alias=True)


def test_filter_requires_parameters():
    data = base_data()
    data["applyElectricalFilter"] = True
    data["filterType"] = "RC"
    data["filterParameters"] = None
    with pytest.raises(ValueError):
        SyntheticDiagnostics.model_validate(data)


def test_noise_model_requires_parameters():
    data = base_data()
    data["includeDetectorNoise"] = True
    data["noiseModel"] = "gaussian"
    data["noiseParameters"] = None
    with pytest.raises(ValueError):
        SyntheticDiagnostics.model_validate(data)


def test_detector_file_required_if_response_applied():
    data = base_data()
    data["applyTimeResponse"] = True
    data["instrumentOverrides"] = {
        "TOF1": {"samplingOverrideNs": 0.5}
    }
    data["detectorIds"] = ["TOF1"]
    with pytest.raises(ValueError):
        SyntheticDiagnostics.model_validate(data)


def test_per_instrument_override_validation():
    data = base_data()
    data["instrumentOverrides"] = {
        "BAD": {"responseFile": "resp.csv"}
    }
    data["detectorIds"] = ["GOOD"]
    with pytest.raises(ValueError):
        SyntheticDiagnostics.model_validate(data)


def test_yaml_round_trip_and_summary_output(tmp_path):
    yaml_text = """
syntheticDiagnostics:
  outputDir: synthetic_diagnostics/
  outputFormat: csv
  samplingIntervalNs: 1.0
  runtimeSyntheticEnabled: true
  postprocessingOnly: false
  syntheticDiagnosticsConfigHash: null
  applyTimeResponse: true
  applyEnergyFilter: true
  applySpatialPsf: false
  syntheticCurrentWaveformEnabled: true
  syntheticVoltageWaveformEnabled: true
  syntheticRogowskiSignalEnabled: true
  syntheticBdotSignalEnabled: true
  syntheticNeutronTofEnabled: true
  syntheticXrayPinholeEnabled: true
  syntheticThomsonParabolaEnabled: false
  syntheticOpticalInterferogramEnabled: false
  detectorIds: [TOF_1, Bdot_X, XPinhole_1]
  diagnosticOutputType:
    TOF_1: time_series
    XPinhole_1: image
  detectorPositionsPath: instruments/detectors_xyz.csv
  diagnosticGeometryModel: 3D
  detectorDefinitionsPath: instruments/detectors.yml
  instrumentResponseDirectory: instruments/responses/
  applyElectricalFilter: true
  filterType: RC
  filterParameters:
    cutoff: 1e7
    order: 1
  includeDetectorNoise: true
  noiseModel: gaussian
  noiseParameters:
    mean: 0.0
    std: 0.01
  instrumentOverrides:
    TOF_1:
      responseFile: responses/tof1.csv
      samplingOverrideNs: 0.5
"""
    if not YAML_AVAILABLE:
        with pytest.raises(Exception):
            __import__("yaml")
        return
    path = tmp_path / "sd.yml"
    path.write_text(yaml_text)
    data = yaml.safe_load(path.read_text())
    cfg = SyntheticDiagnostics.model_validate(data["syntheticDiagnostics"])
    dumped = json.loads(cfg.model_dump_json(by_alias=True))
    cfg2 = SyntheticDiagnostics.model_validate(dumped)
    assert cfg == cfg2
    summary = cfg.summarize()
    assert "Synthetic Diagnostics" in summary
    assert "CSV" in summary


def test_hash_stability_on_toggle_change():
    cfg1 = SyntheticDiagnostics.with_defaults()
    cfg2 = SyntheticDiagnostics.with_defaults()
    assert cfg1.hash_synthetic_diagnostics_config() == cfg2.hash_synthetic_diagnostics_config()
    cfg2 = cfg2.model_copy(update={"synthetic_current_waveform_enabled": False})
    assert cfg1.hash_synthetic_diagnostics_config() != cfg2.hash_synthetic_diagnostics_config()
