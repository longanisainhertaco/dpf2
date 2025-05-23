import pytest

from diagnostics import Diagnostics, DetectorArrayGenerator, OutputField


def test_duplicate_detector_names_fail():
    data = Diagnostics.with_defaults().model_dump()
    data.update(
        {
            "enableSxrDetectors": True,
            "enableNeutronTofDetectors": True,
            "sxrDetectorModels": [{"name": "D1", "position": [0.0, 0.0, 0.0]}],
            "neutronTofDetectors": [{"name": "D1", "position": [1.0, 0.0, 0.0]}],
        }
    )
    with pytest.raises(ValueError):
        Diagnostics.model_validate(data)


def test_stream_backend_requires_path():
    data = Diagnostics.with_defaults().model_dump()
    data.update({"streamingBackend": "websocket", "enableRuntimeObservablesStream": True})
    with pytest.raises(ValueError):
        Diagnostics.model_validate(data)


def test_detector_array_requires_arc_fields():
    data = Diagnostics.with_defaults().model_dump()
    data["detectorArrayGenerator"] = {"type": "arc", "center": [0.0, 0.0, 0.0]}
    with pytest.raises(ValueError):
        Diagnostics.model_validate(data)


def test_output_format_compression_compat():
    data = Diagnostics.with_defaults().model_dump()
    data.update({"outputFormat": "HDF5", "compressionBackend": "blosc"})
    with pytest.raises(ValueError):
        Diagnostics.model_validate(data)


def test_dt_yield_requires_dt_gas():
    data = Diagnostics.with_defaults().model_dump()
    data.update({"enableDtYieldModeling": True})
    with pytest.raises(ValueError):
        Diagnostics.model_validate(data, context={"gas_type": "D2"})
