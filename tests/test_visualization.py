import pytest


matplotlib = pytest.importorskip("matplotlib")

matplotlib.use("Agg")


def test_animate_sheath_returns_animation():
    from dpf2.visualization import animate_sheath

    anim = animate_sheath(1.0, 0.1)
    assert anim is not None


def test_discharge_phases_returns_animation():
    from dpf2.visualization import animate_discharge_phases

    anim = animate_discharge_phases(1.0, 0.1)
    assert anim is not None


def test_sheath_widget_returns_widget():
    ipywidgets = pytest.importorskip("ipywidgets")
    from dpf2.visualization import sheath_widget

    widget = sheath_widget()
    assert isinstance(widget, ipywidgets.Widget)


def test_jxb_field_cross_product():
    import numpy as np
    from dpf2.visualization.sheath import _sheath_field, jxb_field

    b = _sheath_field(1.0, 0.2, 0.0)
    jxb = jxb_field(1.0, 0.2, 0.0)
    assert np.allclose(jxb.u, -0.2 * b.v)
    assert np.allclose(jxb.v, 0.2 * b.u)


def test_sheath_velocity_field_matches_internal():
    import numpy as np
    from dpf2.visualization.sheath import _sheath_field, sheath_velocity_field

    b = _sheath_field(1.0, 0.2, 0.0)
    vel = sheath_velocity_field(1.0, 0.2, 0.0)
    assert np.allclose(vel.u, b.u)
    assert np.allclose(vel.v, b.v)

