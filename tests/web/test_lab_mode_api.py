import json
import zipfile

from dpf2.web.lab_mode_api import (
    generate_manifest,
    generate_manifest_bundle,
    export_manifest_bundle,
)


def test_generate_manifest_bundle():
    configs = [{"a": 1}, {"a": 2}]
    seeds = [42, 43]
    diags = [{"note": "ok"}, {"note": "also ok"}]
    bundle = generate_manifest_bundle(configs, seeds=seeds, diagnostics=diags)
    assert bundle[0]["inputs"] == {"a": 1}
    assert bundle[1]["random_seeds"]["python"] == 43
    assert bundle[0]["code_hash"]
    assert len(bundle[0]["config_hash"]) == 64
    assert bundle[0]["diagnostics"]["note"] == "ok"


def test_generate_manifest():
    manifest = generate_manifest({"b": 3}, seed=99, diagnostics={"val": 1})
    assert manifest["inputs"]["b"] == 3
    assert manifest["random_seeds"]["python"] == 99
    assert manifest["diagnostics"]["val"] == 1


def test_export_manifest_bundle(tmp_path):
    configs = [{"x": 10}]
    zip_path = export_manifest_bundle(
        configs, tmp_path / "bundle.zip", seeds=[123], diagnostics=[{"d": 5}]
    )
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path, "r") as z:
        assert "inputs_0.json" in z.namelist()
        manifest = json.loads(z.read("run_manifest_0.json"))
        assert manifest["inputs"]["x"] == 10
        assert manifest["random_seeds"]["python"] == 123
        assert manifest["diagnostics"]["d"] == 5
        assert len(manifest["config_hash"]) == 64
