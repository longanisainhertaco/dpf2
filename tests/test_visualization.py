import pytest


matplotlib = pytest.importorskip("matplotlib")

matplotlib.use("Agg")


def test_animate_sheath_returns_animation():
    from dpf2.visualization import animate_sheath

    anim = animate_sheath(1.0, 0.1)
    assert anim is not None


def test_sheath_widget_returns_widget():
    ipywidgets = pytest.importorskip("ipywidgets")
    from dpf2.visualization import sheath_widget

    widget = sheath_widget()
    assert isinstance(widget, ipywidgets.Widget)

