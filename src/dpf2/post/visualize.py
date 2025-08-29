"""Visualization utilities using VTK.

The functions in this module are intentionally lightweight wrappers around VTK
primitives.  They convert NumPy arrays into VTK data structures to write mesh
files and subsequently render those meshes to video.  ParaView can consume the
produced ``.vtp`` files for further interactive exploration.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:  # pragma: no cover - optional dependency
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
except Exception:  # pragma: no cover
    vtk = None
    numpy_to_vtk = None


def generate_mesh(points: np.ndarray, polys: np.ndarray, filename: Path) -> Path:
    """Create a VTK PolyData mesh from ``points`` and ``polys``.

    Parameters
    ----------
    points:
        Array of vertex coordinates with shape ``(N, 3)``.
    polys:
        Connectivity array defining triangular faces.
    filename:
        Output ``.vtp`` file.
    """

    if vtk is None:
        raise RuntimeError("VTK is required for mesh generation")

    poly_data = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points))
    poly_data.SetPoints(vtk_points)

    cells = vtk.vtkCellArray()
    flat = np.hstack([np.full((polys.shape[0], 1), 3), polys]).astype(np.int64)
    cells.SetCells(polys.shape[0], numpy_to_vtk(flat.flatten()))
    poly_data.SetPolys(cells)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(filename))
    writer.SetInputData(poly_data)
    writer.Write()
    return filename


def render_video(mesh_file: Path, output: Path, n_frames: int = 360) -> Path:
    """Render ``mesh_file`` to ``output`` video using VTK."""

    if vtk is None:
        raise RuntimeError("VTK is required for rendering")

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(mesh_file))
    reader.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)

    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    window.SetSize(800, 600)

    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(window)

    writer = vtk.vtkFFMPEGWriter()
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.SetFileName(str(output))
    writer.Start()

    for i in range(n_frames):
        window.Render()
        renderer.GetActiveCamera().Azimuth(360.0 / n_frames)
        w2i.Modified()
        writer.Write()

    writer.End()
    return output


__all__ = ["generate_mesh", "render_video"]
