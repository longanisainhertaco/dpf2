import json
import pytest

from parallel_settings import ParallelSettings


def make_base_data():
    return {
        "mpiRanks": 2,
        "gpuBackend": "CUDA",
        "useMultithreading": True,
        "threadsPerRank": 2,
        "gpuDeviceIds": [0, 1],
    }


def test_resource_overcommit():
    data = make_base_data()
    data.update({"maxAllowedCores": 2})
    cfg = ParallelSettings.model_validate(data)
    summary = cfg.summarize()
    assert "exceeds max allowed cores" in summary


def test_gpu_backend_requires_ids():
    data = make_base_data()
    data["gpuDeviceIds"] = []
    with pytest.raises(ValueError):
        ParallelSettings.model_validate(data)


def test_hip_backend_blocks_pinned_memory():
    data = make_base_data()
    data["gpuBackend"] = "HIP"
    cfg = ParallelSettings.model_validate(data)
    assert "HIP backend" in cfg.summarize()


def test_manual_decomp_requires_manual_amr():
    data = make_base_data()
    data.update({"decompositionStrategy": "manual", "amrRefinementCriteria": "gradient"})
    with pytest.raises(ValueError):
        ParallelSettings.model_validate(data)


def test_invalid_gpu_fallback_raises():
    data = make_base_data()
    data["gpuFallbackMode"] = "invalid"
    with pytest.raises(ValueError):
        ParallelSettings.model_validate(data)


def test_tile_size_validates():
    data = make_base_data()
    data["tileSize"] = [32, -1, 1]
    with pytest.raises(ValueError):
        ParallelSettings.model_validate(data)


def test_summarize_outputs_expected_format():
    data = make_base_data()
    data.update({
        "decompositionStrategy": "auto",
        "amrRefinementCriteria": "gradient",
        "decompositionFailurePolicy": "auto_reduce",
        "schedulerContext": "slurm",
        "launcherType": "srun",
        "bufferPoolSizeMb": 256,
        "tileSize": [32, 32, 1],
        "gpuPartitionStrategy": "affinity_map",
    })
    cfg = ParallelSettings.model_validate(data)
    summary = cfg.summarize()
    assert "Parallelism" in summary
    assert "Tile size" in summary


def test_config_hash_changes_on_device_list():
    d1 = make_base_data()
    cfg1 = ParallelSettings.model_validate(d1)
    d2 = make_base_data()
    d2["gpuDeviceIds"] = [0]
    cfg2 = ParallelSettings.model_validate(d2)
    assert cfg1.parallel_config_hash != cfg2.parallel_config_hash
