import pytest


pytest.importorskip("matplotlib")

import matplotlib

matplotlib.use("Agg")


def test_animate_sheath_returns_animation():
    from dpf2.visualization import animate_sheath

    anim = animate_sheath(1.0, 0.1)
    assert anim is not None

