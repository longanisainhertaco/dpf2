import json
from pathlib import Path

import pytest

from postprocessing_settings import PostprocessingSettings

try:
    import yaml  # type: ignore
    YAML_AVAILABLE = True
except Exception:
    YAML_AVAILABLE = False


def base_data():
    return PostprocessingSettings.with_defaults().model_dump(by_alias=True)


def test_yaml_round_trip_and_summary_output(tmp_path: Path):
    yaml_text = """
postprocessing:
  enabled: true
  postprocessingOutputDir: postprocessing/
  fileOutputFormat: OpenPMD
  outputFrequency: pinch_only
  customOutputTimesUs: [0.5, 1.0, 1.5]
  pinchDetectionMethod: Ti_peak
  pinchDetectionThreshold: 1.5
  postprocessingTaskTypes: [neutron, xray, field]
  generateSyntheticDiagnostics: true
  computeNeutronYieldBreakdown: true
  computeXraySpectrum: true
  extractMaxwellianFitParameters: true
  trackPlasmaCenterlineEvolution: true
  exportParticleEnergySpectra: true
  integrateSignalOverTime: [Ip, Bz]
  calculateFieldExtrema: true
  computeSpatialAverages: true
  integrateEnergyDensity: true
  outputFieldSlices: [z]
  fieldSlicePositionsCm: [0.0, 2.5]
  fieldSliceUnits: cm
  performFourierAnalysis: true
  fourierAxes: [z]
  frequencyWindowUs: [0.0, 5.0]
  rawDataSourceDir: results/
  externalFilterScriptPath: tools/filter_post.py
  externalFilterParameters:
    window: 0.1
    threshold: 0.5
  postprocessingTemplatePath: templates/post_template.yml
"""
    if not YAML_AVAILABLE:
        with pytest.raises(Exception):
            __import__("yaml")
        return
    p = tmp_path / "pp.yml"
    p.write_text(yaml_text)
    data = yaml.safe_load(p.read_text())
    cfg = PostprocessingSettings.model_validate(data["postprocessing"])
    dumped = json.loads(cfg.model_dump_json(by_alias=True))
    cfg2 = PostprocessingSettings.model_validate(dumped)
    assert cfg == cfg2
    summary = cfg.summarize()
    assert "Postprocessing" in summary
    assert "pinch_only" in summary


def test_custom_times_sorted_positive():
    data = base_data()
    data["outputFrequency"] = "custom"
    data["customOutputTimesUs"] = [1.0, 0.5]
    with pytest.raises(ValueError):
        PostprocessingSettings.model_validate(data)
    data["customOutputTimesUs"] = [-1.0, 0.5]
    with pytest.raises(ValueError):
        PostprocessingSettings.model_validate(data)


def test_script_path_and_parameters_checked(tmp_path: Path):
    data = base_data()
    data["externalFilterParameters"] = {"a": 1.0}
    with pytest.raises(ValueError):
        PostprocessingSettings.model_validate(data)
    data["externalFilterScriptPath"] = tmp_path / "missing.py"
    with pytest.raises(ValueError):
        PostprocessingSettings.model_validate(data)
    script = tmp_path / "f.py"
    script.write_text("pass")
    data["externalFilterScriptPath"] = script
    cfg = PostprocessingSettings.model_validate(data)
    assert cfg.external_filter_script_path == script


def test_pinch_detection_mode_requires_threshold():
    data = base_data()
    data["outputFrequency"] = "pinch_only"
    with pytest.raises(ValueError):
        PostprocessingSettings.model_validate(data)
    data["pinchDetectionThreshold"] = 1.0
    cfg = PostprocessingSettings.model_validate(data)
    assert cfg.pinch_detection_threshold == 1.0


def test_hash_stability_on_toggle_change():
    cfg1 = PostprocessingSettings.with_defaults()
    cfg2 = PostprocessingSettings.with_defaults()
    assert cfg1.hash_postprocessing_config() == cfg2.hash_postprocessing_config()
    cfg2 = cfg2.model_copy(update={"compute_neutron_yield_breakdown": False})
    assert cfg1.hash_postprocessing_config() != cfg2.hash_postprocessing_config()


def test_field_slice_axis_and_unit_sync():
    data = base_data()
    data["outputFieldSlices"] = ["x", "z"]
    data["fieldSlicePositionsCm"] = [0.0]
    with pytest.raises(ValueError):
        PostprocessingSettings.model_validate(data)
    data["fieldSlicePositionsCm"] = [0.0, 1.0]
    cfg = PostprocessingSettings.model_validate(data)
    assert cfg.output_field_slices == ["x", "z"]
