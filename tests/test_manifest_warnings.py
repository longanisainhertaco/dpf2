import json

from dpf2.cli.lab import write_manifest


def test_manifest_includes_warnings(tmp_path):
    path = write_manifest(tmp_path, warnings=["a", "b"])
    data = json.loads(path.read_text())
    assert data["warnings"] == ["a", "b"]
