from __future__ import annotations

"""Geometry ingestion helpers coupling CAD data, material models and inductance."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .inductance import reconstruct_dynamic_inductance
from .loaders import load_axisymmetric_mesh, load_cad_geometry, load_unstructured_mesh
from .parameterized import HollowGeometry, ReentrantGeometry, TaperedGeometry
from ..materials.library import MaterialLibrary


@dataclass
class ImportedGeometry:
    """Container for imported geometry metadata.

    The helper binds lightweight material models to either parametric or CAD
    geometry and exposes a utility to reconstruct inductance histories from
    transient plasma radii.  The intent is to minimise boilerplate in CLI/API
    entrypoints while keeping the coupling logic explicit and testable.
    """

    nodes: Sequence[Sequence[float]]
    elements: Sequence[Sequence[int]]
    materials: Sequence[str]
    features: Dict[str, List[int]] | None = None
    material_models: Dict[str, object] = field(default_factory=dict)

    def inductance_from_radius(self, radii: Iterable[float], length: float) -> List[float]:
        """Return a reconstructed inductance trace for the supplied ``radii``.

        The routine treats the imported geometry as coaxial and leverages
        :func:`~dpf2.geometry.inductance.reconstruct_dynamic_inductance` for the
        Biot–Savart integration.  ``length`` represents the effective plasma
        length used to normalise the reconstruction.
        """

        return reconstruct_dynamic_inductance(radii, length=length)


def _material_labels(materials: Sequence[object] | None, default: str) -> List[str]:
    if not materials:
        return [default]
    labels: List[str] = []
    for entry in materials:
        labels.append(str(entry))
    return labels


def load_geometry_with_materials(
    path: str | Path,
    *,
    default_material: str = "stainless_steel",
    material_overrides: Dict[str, str] | None = None,
) -> ImportedGeometry:
    """Load CAD or unstructured geometry and attach material models.

    ``material_overrides`` maps material identifiers embedded in the mesh to
    names understood by :class:`~dpf2.materials.library.MaterialLibrary`.
    Unknown entries fall back to ``default_material`` so that geometry import
    remains resilient for quick parametric sweeps.
    """

    p = Path(path)
    raw: Dict[str, object]
    if p.suffix.lower() in {".json", ".step", ".stp", ".iges", ".igs", ".stl", ".vtk"}:
        raw = load_cad_geometry(p)
    elif p.suffix.lower() in {".msh", ".txt"}:
        raw = load_unstructured_mesh(p)
    else:
        raw = load_axisymmetric_mesh(p)

    mat_labels = _material_labels(raw.get("materials"), default_material)  # type: ignore[arg-type]
    overrides = material_overrides or {}
    bound: Dict[str, object] = {}
    for label in mat_labels:
        name = overrides.get(str(label), str(label)) or default_material
        try:
            bound[str(label)] = MaterialLibrary.get(name)
        except Exception:
            bound[str(label)] = MaterialLibrary.get(default_material)

    return ImportedGeometry(
        nodes=raw["nodes"],
        elements=raw["elements"],
        materials=[overrides.get(str(m), str(m)) if material_overrides else str(m) for m in mat_labels],
        features=raw.get("features"),
        material_models=bound,
    )


def parametrized_geometry(
    shape: TaperedGeometry | HollowGeometry | ReentrantGeometry,
    *,
    material: str = "copper",
    axial_samples: int = 32,
) -> ImportedGeometry:
    """Generate a mesh-like description from simple parametric shapes.

    The output mirrors the dictionary structure of CAD imports so downstream
    routines (diagnostics, inductance reconstruction) can operate uniformly.
    """

    nodes: List[List[float]] = []
    elements: List[List[int]] = []

    if isinstance(shape, TaperedGeometry):
        profile = shape.radius_profile(axial_samples)
        for i, (z, r) in enumerate(profile):
            nodes.append([r, 0.0, z])
            if i > 0:
                elements.append([i - 1, i, i])
    elif isinstance(shape, HollowGeometry):
        nodes = [
            [shape.r_outer, 0.0, 0.0],
            [shape.r_outer, 0.0, shape.length],
            [shape.r_inner, 0.0, 0.0],
            [shape.r_inner, 0.0, shape.length],
        ]
        elements = [[0, 1, 2], [1, 2, 3]]
    elif isinstance(shape, ReentrantGeometry):
        for i, (z, r) in enumerate(shape.profile()):
            nodes.append([r, 0.0, z])
            if i > 0:
                elements.append([i - 1, i, i])
    else:  # pragma: no cover - defensive programming
        raise TypeError(f"Unsupported parametric shape: {type(shape)}")

    mat_model = MaterialLibrary.get(material)
    return ImportedGeometry(
        nodes=nodes,
        elements=elements,
        materials=[material] * (len(elements) or 1),
        material_models={material: mat_model},
    )


__all__ = [
    "ImportedGeometry",
    "load_geometry_with_materials",
    "parametrized_geometry",
]
