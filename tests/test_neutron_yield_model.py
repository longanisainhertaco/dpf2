import json
import pytest

from neutron_yield_model import NeutronYieldModel

try:
    import yaml  # type: ignore
    YAML_AVAILABLE = True
except Exception:
    YAML_AVAILABLE = False


def base_data():
    return NeutronYieldModel.with_defaults().model_dump(by_alias=True)


def test_requires_temp_density_for_analytic_thermal_model():
    data = base_data()
    data["reactivitySource"] = "analytic"
    data.pop("averageIonTemperatureKeV", None)
    data.pop("averageIonDensityCm3", None)
    with pytest.raises(ValueError):
        NeutronYieldModel.model_validate(data)


def test_missing_response_file_when_enabled():
    data = base_data()
    data["applyDetectorResponseFunction"] = True
    data["detectorResponseFile"] = None
    with pytest.raises(ValueError):
        NeutronYieldModel.model_validate(data)


def test_branching_ratio_bounds():
    data = base_data()
    data["ddBranchingRatio"] = 1.5
    with pytest.raises(ValueError):
        NeutronYieldModel.model_validate(data)
    data["ddBranchingRatio"] = -0.1
    with pytest.raises(ValueError):
        NeutronYieldModel.model_validate(data)


def test_hash_stability():
    cfg1 = NeutronYieldModel.with_defaults()
    cfg2 = NeutronYieldModel.with_defaults()
    assert cfg1.hash_neutron_yield_config() == cfg2.hash_neutron_yield_config()


def test_yaml_round_trip_and_summary_output(tmp_path):
    yaml_text = """
neutronYield:
  fusionFuelType: DD
  beamTargetModelEnabled: true
  thermonuclearModelEnabled: true
  separateYieldComponents: true
  yieldIntegrationWindowUs: [0.5, 1.5]
  beamIonSpecies: D+
  targetDensitySource: diagnostics
  targetDensityConstant: null
  iedfSource: diagnostics
  iedfUserPath: null
  iedfFormat: csv
  fusionCrossSectionModel: Bosch-Hale
  crossSectionTablePath: null
  crossSectionTableUnits:
    energy: MeV
    sigma: barn
  reactivitySource: analytic
  maxwellianAssumed: true
  averageIonTemperatureKeV: 3.0
  averageIonDensityCm3: 1e21
  ddBranchingRatio: 0.5
  reactivityTablePath: null
  reactivityTableUnits:
    Ti: keV
    reactivity: cm^3/s
  neutronSpectrumOutputEnabled: true
  spectrumEnergyBinsMeV: [1.0, 2.5, 14.1]
  spectrumOutputFormat: csv
  applyDetectorResponseFunction: true
  detectorResponseFile: detectors/n_response.csv
  detectorResponseNormalization: area
"""
    if not YAML_AVAILABLE:
        with pytest.raises(Exception):
            __import__("yaml")
        return
    path = tmp_path / "ny.yml"
    path.write_text(yaml_text)
    data = yaml.safe_load(path.read_text())
    cfg = NeutronYieldModel.model_validate(data["neutronYield"])
    dumped = json.loads(cfg.model_dump_json(by_alias=True))
    cfg2 = NeutronYieldModel.model_validate(dumped)
    assert cfg == cfg2
    summary = cfg.summarize()
    assert "Beam-target" in summary and "Thermonuclear" in summary
