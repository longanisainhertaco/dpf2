"""Sheath visualization utilities.

This module provides a very small helper for animating a synthetic
plasma sheath using either Matplotlib or, if available, VTK.  The
implementation is intentionally lightweight and purely illustrative;
it is used by the example notebooks and can run without the heavy
simulation dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

try:  # pragma: no cover - matplotlib optional at runtime
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
except Exception:  # pragma: no cover - matplotlib may be absent
    plt = None  # type: ignore
    FuncAnimation = None  # type: ignore

try:  # pragma: no cover - vtk is purely optional
    import vtk  # type: ignore
except Exception:  # pragma: no cover - vtk is optional
    vtk = None  # type: ignore


@dataclass
class SheathField:
    """Container for the vector field used in the animation."""

    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray


def _sheath_field(voltage: float, pressure: float, t: float) -> SheathField:
    """Generate a toy sheath vector field.

    Parameters
    ----------
    voltage:
        Driving voltage for the sheath.
    pressure:
        Background pressure affecting decay.
    t:
        Time-like parameter for the evolution.
    """

    grid = np.linspace(-1.0, 1.0, 20)
    x, y = np.meshgrid(grid, grid)
    sheath = np.sin(t * voltage) * np.exp(-pressure * t)
    u = -y * sheath
    v = x * sheath
    return SheathField(x, y, u, v)


def animate_sheath(
    voltage: float,
    pressure: float,
    *,
    use_vtk: bool = False,
    captions: Optional[Sequence[str]] = None,
):
    """Animate a simple sheath evolution.

    The function returns the underlying animation/renderer object so
    that callers (e.g. notebooks) can display it directly.

    Parameters
    ----------
    voltage:
        Driving voltage.
    pressure:
        Background pressure.
    use_vtk:
        Use a VTK pipeline instead of Matplotlib when ``True`` and VTK
        is installed.
    captions:
        Optional sequence of captions that will be displayed for each
        frame.  When provided, the ``i``\ th caption is shown along with
        the step number in the animation.
    """

    captions = list(captions or [])

    if use_vtk and vtk is not None:
        # Minimal VTK pipeline – render a coloured sphere that changes
        # with the sheath evolution.  This avoids heavy dependencies
        # while still exercising the rendering path.
        sphere = vtk.vtkSphereSource()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)
        text = vtk.vtkTextActor()
        text.SetPosition(10, 10)
        renderer.AddActor2D(text)
        window = vtk.vtkRenderWindow()
        window.AddRenderer(renderer)

        def _update(frame: int) -> None:
            colour = 0.5 + 0.5 * np.sin(frame * voltage)
            actor.GetProperty().SetColor(colour, 0.0, 1.0 - colour)
            if frame < len(captions):
                text.SetInput(f"Step {frame + 1}: {captions[frame]}")
            else:
                text.SetInput(f"Step {frame + 1}")
            window.Render()

        for i in range(20):
            _update(i)
        return window

    # Fallback to Matplotlib
    if plt is None or FuncAnimation is None:  # pragma: no cover - import guard
        raise RuntimeError("matplotlib is required for animate_sheath")

    fig, ax = plt.subplots()
    field0 = _sheath_field(voltage, pressure, 0.0)
    quiver = ax.quiver(field0.x, field0.y, field0.u, field0.v)
    ax.set_title("Sheath evolution")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    caption_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")

    def _frame(i: int):
        fld = _sheath_field(voltage, pressure, i * 0.1)
        quiver.set_UVC(fld.u, fld.v)
        if i < len(captions):
            caption_text.set_text(f"Step {i + 1}: {captions[i]}")
        else:
            caption_text.set_text(f"Step {i + 1}")
        return quiver, caption_text

    anim = FuncAnimation(fig, _frame, frames=40, interval=50, blit=True)
    return anim


__all__ = ["animate_sheath"]

