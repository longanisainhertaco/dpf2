import json
from pathlib import Path

import pytest

from ml_model_config import MLModelConfig

try:
    import yaml  # type: ignore
    YAML_AVAILABLE = True
except Exception:
    YAML_AVAILABLE = False


def base_data(tmp_path: Path) -> dict:
    p = tmp_path / "train.csv"
    p.write_text("data")
    data = MLModelConfig.with_defaults().model_dump(by_alias=True)
    data["trainingDatasetPath"] = p
    data["inferenceOutputPath"] = tmp_path / "out.csv"
    data["inferenceInputPath"] = tmp_path / "X.csv"
    return data


def test_feature_and_target_lists_not_empty(tmp_path: Path):
    data = base_data(tmp_path)
    data["inputFeatures"] = []
    with pytest.raises(ValueError):
        MLModelConfig.model_validate(data)
    data = base_data(tmp_path)
    data["outputTargets"] = []
    with pytest.raises(ValueError):
        MLModelConfig.model_validate(data)


def test_model_type_matches_loss_function(tmp_path: Path):
    data = base_data(tmp_path)
    data["modelType"] = "classification"
    data["lossFunction"] = "mse"
    with pytest.raises(ValueError):
        MLModelConfig.model_validate(data)
    data["lossFunction"] = "crossentropy"
    cfg = MLModelConfig.model_validate(data)
    assert cfg.model_type == "classification"


def test_inference_paths_required_if_enabled(tmp_path: Path):
    data = base_data(tmp_path)
    data["inferenceEnabled"] = True
    data.pop("inferenceInputPath")
    with pytest.raises(ValueError):
        MLModelConfig.model_validate(data)
    data = base_data(tmp_path)
    data["inferenceEnabled"] = True
    cfg = MLModelConfig.model_validate(data)
    assert cfg.inference_output_path


def test_yaml_round_trip_and_summary_output(tmp_path: Path):
    yaml_text = """
mlModel:
  modelType: regression
  surrogateModelEnabled: true
  modelName: gpr_v4
  modelEngine: gpytorch
  modelArchitectureTemplate: MLP
  modelArchitecture: 3x64-relu
  trainingDatasetPath: datasets/train.csv
  inputFeatures: [V0, C_ext, L_ext]
  outputTargets: [neutron_yield, pinch_radius]
  normalizationMethod: zscore
  trainingEpochs: 200
  batchSize: 32
  optimizerType: adam
  lossFunction: mse
  validationSplit: 0.2
  earlyStoppingEnabled: true
  earlyStoppingPatience: 10
  splitMode: train_val
  cvFolds: 5
  saveBestModelEnabled: true
  bestModelOutputPath: models/gpr_best.pt
  inferenceEnabled: true
  loadExistingModel: true
  existingModelPath: models/gpr_best.pt
  inferenceInputPath: datasets/infer_X.csv
  inferenceOutputPath: results/predict_Y.csv
  confidenceThreshold: 0.05
  normalizeInputFeatures: true
  evaluationMetrics: [rmse, r2, f1]
  trackFeatureImportance: true
"""
    if not YAML_AVAILABLE:
        with pytest.raises(Exception):
            __import__("yaml")
        return
    p = tmp_path / "ml.yml"
    p.write_text(yaml_text)
    data = yaml.safe_load(p.read_text())
    cfg = MLModelConfig.model_validate(data["mlModel"])
    dumped = json.loads(cfg.model_dump_json(by_alias=True))
    cfg2 = MLModelConfig.model_validate(dumped)
    assert cfg == cfg2
    summary = cfg.summarize()
    assert "ML Model" in summary and "Inference" in summary


def test_hash_changes_with_model_name(tmp_path: Path):
    cfg1 = MLModelConfig.with_defaults()
    cfg2 = MLModelConfig.with_defaults().model_copy(update={"model_name": "other"})
    assert cfg1.hash_ml_model_config() != cfg2.hash_ml_model_config()
