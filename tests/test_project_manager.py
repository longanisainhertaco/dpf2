from pathlib import Path

import pytest

from dpf2.gui import ProjectManager


def test_export_and_overlay(tmp_path):
    pm = ProjectManager()
    pm.metrics["run1"] = {
        0.5: {"yield": 1.0, "efficiency": 0.1},
        1.0: {"yield": 2.0, "efficiency": 0.2},
    }
    pm.metrics["run2"] = {
        0.5: {"yield": 1.5, "efficiency": 0.15},
        1.0: {"yield": 2.5, "efficiency": 0.25},
    }

    out_csv = pm.export_metrics(tmp_path / "metrics.csv")
    assert out_csv.exists()
    content = out_csv.read_text()
    assert "run1" in content and "run2" in content

    pytest.importorskip("matplotlib")
    out_plot = pm.overlay_yield_pressure(tmp_path / "yield_pressure.png")
    assert out_plot.exists()


def test_scene_roundtrip(tmp_path):
    pm = ProjectManager(project="demo")
    pm.metrics["run"] = {0.5: {"yield": 1.0}}
    pm.params["run"] = "initial_pressure"
    scene = pm.export_scene(tmp_path / "scene.json")
    assert scene.exists()

    new_pm = ProjectManager()
    new_pm.import_scene(scene)
    assert new_pm.project == "demo"
    assert new_pm.metrics == {"run": {0.5: {"yield": 1.0}}}
    assert new_pm.params == {"run": "initial_pressure"}
