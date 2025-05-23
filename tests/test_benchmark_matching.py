import pytest

from benchmark_matching import BenchmarkMatching


def test_single_vs_multi_trace_conflict():
    data = BenchmarkMatching.with_defaults().model_dump(by_alias=True)
    data["benchmarkTracePath"] = "a.csv"
    data["benchmarkTracePaths"] = ["b.csv", "c.csv"]
    with pytest.raises(ValueError):
        BenchmarkMatching.model_validate(data)


def test_compare_fields_requires_trace():
    data = BenchmarkMatching.with_defaults().model_dump(by_alias=True)
    data["compareFields"] = ["E"]
    with pytest.raises(ValueError):
        BenchmarkMatching.model_validate(data)


def test_benchmark_units_are_valid():
    data = BenchmarkMatching.with_defaults().model_dump(by_alias=True)
    data["benchmarkFields"] = ["I(t)"]
    data["benchmarkUnits"] = {"I(t)": "invalid"}
    with pytest.raises(ValueError):
        BenchmarkMatching.model_validate(data)


def test_match_window_bounds():
    data = BenchmarkMatching.with_defaults().model_dump(by_alias=True)
    data["matchRegionStartUs"] = 5.0
    data["matchRegionEndUs"] = 1.0
    with pytest.raises(ValueError):
        BenchmarkMatching.model_validate(data)


def test_feature_fields_required_for_alignment():
    data = BenchmarkMatching.with_defaults().model_dump(by_alias=True)
    data["matchWaveformFeatures"] = True
    data["benchmarkFields"] = []
    with pytest.raises(ValueError):
        BenchmarkMatching.model_validate(data)


def test_config_hash_consistency():
    data = BenchmarkMatching.with_defaults().model_dump(by_alias=True)
    data["benchmarkTracePath"] = "file:///tmp/trace.csv"
    cfg1 = BenchmarkMatching.model_validate(data)
    cfg2 = BenchmarkMatching.model_validate(data)
    assert cfg1.benchmark_config_hash == cfg2.benchmark_config_hash
