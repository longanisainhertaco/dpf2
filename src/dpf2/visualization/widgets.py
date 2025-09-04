"""Interactive widgets for visualization.

This module provides Jupyter widgets to manipulate the sheath
animation in real time.  The widgets are intentionally lightweight so
that they can be used in example notebooks without heavy dependencies
beyond ``ipywidgets``.
"""
from __future__ import annotations

from typing import Any

from .sheath import animate_sheath


def sheath_widget(initial_voltage: float = 1.0, initial_pressure: float = 0.1):
    """Return a widget controlling :func:`animate_sheath`.

    Parameters
    ----------
    initial_voltage:
        Starting voltage for the slider.
    initial_pressure:
        Starting pressure for the slider.

    The returned widget contains sliders for voltage and pressure along
    with an output area displaying the resulting animation.
    """
    try:  # pragma: no cover - ipywidgets optional at runtime
        import ipywidgets as widgets
        from IPython.display import clear_output, display
    except Exception as exc:  # pragma: no cover - ipywidgets may be absent
        raise RuntimeError("ipywidgets is required for sheath_widget") from exc

    voltage = widgets.FloatSlider(
        value=initial_voltage,
        min=0.0,
        max=5.0,
        step=0.1,
        description="Voltage (kV)",
    )
    pressure = widgets.FloatSlider(
        value=initial_pressure,
        min=0.0,
        max=1.0,
        step=0.01,
        description="Pressure (bar)",
    )
    out = widgets.Output()

    def _update(_: Any) -> None:
        with out:
            clear_output(wait=True)
            anim = animate_sheath(voltage.value, pressure.value)
            display(anim)

    voltage.observe(_update, names="value")
    pressure.observe(_update, names="value")
    _update(None)

    return widgets.VBox([voltage, pressure, out])


__all__ = ["sheath_widget"]
