from __future__ import annotations

"""Geometry ingestion helpers coupling CAD data, material models and inductance."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Mapping
import math
from dataclasses import replace

from .inductance import reconstruct_dynamic_inductance
from .loaders import load_axisymmetric_mesh, load_cad_geometry, load_unstructured_mesh
from .parameterized import HollowGeometry, ReentrantGeometry, TaperedGeometry
from ..materials.library import MaterialLibrary, Material
from ..dpf_config import ElectrodeGeometry


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

    def lp_trace(self, radius_history: Sequence[tuple[float, float]], length: float) -> List[tuple[float, float]]:
        """Return ``(t, Lp)`` pairs reconstructed from ``radius_history``.

        ``radius_history`` is expected to be ``[(time, radius), ...]`` where time
        is in seconds and radius in metres.  The helper delegates to
        :func:`inductance_from_radius` for the heavy lifting, keeping the mapping
        of CAD/parametric geometry to circuit-facing inductance profiles in one
        place.
        """

        radii = [r for _, r in radius_history]
        inductances = self.inductance_from_radius(radii, length)
        return [(t, Lp) for (t, _), Lp in zip(radius_history, inductances)]


def _material_labels(materials: Sequence[object] | None, default: str) -> List[str]:
    if not materials:
        return [default]
    labels: List[str] = []
    for entry in materials:
        labels.append(str(entry))
    return labels


def _bind_material(
    label: str,
    default_material: str,
    overrides: Mapping[str, str] | None = None,
    material_properties: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[str, Material]:
    """Return a material instance honouring overrides and property tweaks."""

    overrides = overrides or {}
    target = overrides.get(label, label) or default_material
    try:
        base = MaterialLibrary.get(target)
    except Exception:
        base = MaterialLibrary.get(default_material)
        target = default_material

    props = (material_properties or {}).get(target) or (material_properties or {}).get(label)
    if props:
        base = replace(
            base,
            density=props.get("density", base.density),
            atomic_mass=props.get("atomic_mass", base.atomic_mass),
            sputter_yield=props.get("sputter_yield", base.sputter_yield),
            resistivity=props.get("resistivity", base.resistivity),
            frequency_ref=props.get("frequency_ref", base.frequency_ref),
            surface_conditioning=props.get("surface_conditioning", base.surface_conditioning),
        )
    return target, base


def load_geometry_with_materials(
    path: str | Path,
    *,
    default_material: str = "stainless_steel",
    material_overrides: Dict[str, str] | None = None,
    material_properties: Mapping[str, Mapping[str, float]] | None = None,
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
    bound: Dict[str, object] = {}
    for label in mat_labels:
        name, mat = _bind_material(
            str(label),
            default_material,
            material_overrides,
            material_properties,
        )
        bound[str(label)] = mat

    return ImportedGeometry(
        nodes=raw["nodes"],
        elements=raw["elements"],
        materials=[_bind_material(str(m), default_material, material_overrides, material_properties)[0] for m in mat_labels],
        features=raw.get("features"),
        material_models=bound,
    )


def parametrized_geometry(
    shape: TaperedGeometry | HollowGeometry | ReentrantGeometry,
    *,
    material: str = "copper",
    axial_samples: int = 32,
    material_properties: Mapping[str, Mapping[str, float]] | None = None,
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

    label, mat_model = _bind_material(material, material, None, material_properties)
    return ImportedGeometry(
        nodes=nodes,
        elements=elements,
        materials=[label] * (len(elements) or 1),
        material_models={label: mat_model},
    )


def ingest_electrode_geometry(
    geometry: ElectrodeGeometry,
    *,
    length_m: float = 0.1,
    outer_radius_m: float = 0.05,
    inner_radius_m: float | None = None,
    material_overrides: Mapping[str, str] | None = None,
    material_properties: Mapping[str, Mapping[str, float]] | None = None,
) -> ImportedGeometry:
    """Construct an :class:`ImportedGeometry` from config-style inputs.

    The helper understands both CAD/mesh inputs (via ``geometry.mesh_file``)
    and parametric electrode presets (tapered, hollow, re-entrant).  Material
    overrides and per-material property tweaks can be provided using the same
    keys as :class:`~dpf2.dpf_config.ElectrodeGeometry`.
    """

    default_material = getattr(geometry, "default_material", None) or "stainless_steel"
    overrides_raw = getattr(geometry, "material_overrides", None)
    if overrides_raw is None:
        try:
            overrides_raw = geometry.model_dump().get("material_overrides", {})  # type: ignore[union-attr]
        except Exception:
            overrides_raw = {}
    overrides = dict(overrides_raw or {})
    overrides.update(material_overrides or {})
    props_raw = getattr(geometry, "material_properties", None)
    if props_raw is None:
        try:
            props_raw = geometry.model_dump().get("material_properties", {})  # type: ignore[union-attr]
        except Exception:
            props_raw = {}
    properties = dict(props_raw or {})
    properties.update(material_properties or {})

    def _rebind_materials(geom: ImportedGeometry) -> ImportedGeometry:
        labels = list(geom.material_models.keys()) or [default_material]
        rebound: Dict[str, Material] = {}
        material_list: List[str] = []
        for lbl in labels:
            name, mat = _bind_material(lbl, default_material, overrides, properties)
            rebound[lbl] = mat
            material_list.append(name)
        return replace(geom, material_models=rebound, materials=material_list)

    if geometry.mesh_file is not None:
        imported = load_geometry_with_materials(
            geometry.mesh_file,
            default_material=default_material,
            material_overrides=overrides,
            material_properties=properties,
        )
        return _rebind_materials(imported)

    shape_name = getattr(geometry, "anode_shape", "cylinder")
    if shape_name == "tapered":
        taper = getattr(geometry, "taper_angle", 5.0) or 5.0
        slope = math.tan(math.radians(taper))
        r_top = max(outer_radius_m * 0.1, outer_radius_m - slope * length_m)
        shape: TaperedGeometry | HollowGeometry | ReentrantGeometry = TaperedGeometry(
            length=length_m,
            r_base=outer_radius_m,
            r_top=r_top,
        )
    elif shape_name == "hollow":
        bore = inner_radius_m or getattr(geometry, "inner_radius", None) or outer_radius_m * 0.5
        shape = HollowGeometry(length=length_m, r_outer=outer_radius_m, r_inner=bore)
    elif shape_name == "reentrant":
        depth = getattr(geometry, "reentrant_depth", 0.0) or 0.0
        shape = ReentrantGeometry([(0.0, outer_radius_m), (depth, outer_radius_m * 0.5)])
    else:
        shape = TaperedGeometry(length=length_m, r_base=outer_radius_m, r_top=outer_radius_m)

    material_name = overrides.get("electrode", default_material)
    imported = parametrized_geometry(
        shape,
        material=material_name,
        material_properties=properties,
        axial_samples=64 if isinstance(shape, TaperedGeometry) else 4,
    )
    imported = _rebind_materials(imported)
    if isinstance(shape, HollowGeometry):
        imported = replace(imported, features={"outer": [0], "inner": [1]})
    return imported


__all__ = [
    "ImportedGeometry",
    "load_geometry_with_materials",
    "parametrized_geometry",
    "ingest_electrode_geometry",
]
