from pathlib import Path

from dpf2.dpf_config import ElectrodeGeometry
from dpf2.geometry.importer import ingest_electrode_geometry, load_geometry_with_materials


def test_parameterized_ingest_applies_material_overrides():
    geom = ElectrodeGeometry.tapered(
        geometry_preset="tapered",
        cathode_type="bar",
        cathode_bar_count=8,
        cathode_gap_degrees=45.0,
        taper_angle=8.0,
    )
    geom = geom.model_copy(
        update={
            "material_overrides": {"electrode": "copper"},
            "material_properties": {"copper": {"surface_conditioning": 1.1}},
        }
    )
    imported = ingest_electrode_geometry(geom, length_m=0.2, outer_radius_m=0.05)
    assert imported.materials[0] == "copper"
    assert list(imported.material_models.values())[0].surface_conditioning == 1.1


def test_mesh_ingest_respects_material_properties():
    path = Path(__file__).with_name("hollow.step")
    geom = ElectrodeGeometry.hollow(
        geometry_preset="hollow",
        cathode_type="bar",
        cathode_bar_count=10,
        cathode_gap_degrees=36.0,
        mesh_file=path,
    ).model_copy(
        update={
            "material_overrides": {"steel": "quartz"},
            "material_properties": {"quartz": {"resistivity": 9.0}},
        }
    )
    imported = ingest_electrode_geometry(geom)
    assert imported.materials[0] == "quartz"
    assert imported.materials[1] == "stainless_steel"
    assert imported.material_models["steel"].resistivity == 9.0
