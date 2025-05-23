from amrex_settings import AmrexSettings, ElectrodeGeometry
import pytest


def base_data():
    return {
        "amr_levels": 2,
        "amr_coarsening_ratio": 2,
        "integrator": "RK4",
        "stencil_order": 2,
        "electrode_geometry": {
            "cathode_type": "bars",
            "cathode_bar_count": 16,
            "cathode_gap_degrees": 15.0,
            "anode_shape": "cone",
            "knife_edge_enabled": True,
        },
        "electrode_material": "Cu",
        "erosion_mechanisms_enabled": ["thermal"],
    }


def test_tile_size_vs_grid_size_check():
    data = base_data()
    data.update({"max_grid_size": 32, "tile_size_override": [64, 32, 1]})
    with pytest.raises(ValueError):
        AmrexSettings.model_validate(data)


def test_invalid_material_override_key():
    data = base_data()
    data["material_properties_override"] = {"bad": 1.0}
    with pytest.raises(ValueError):
        AmrexSettings.model_validate(data)


def test_mesh_file_units_scale():
    data = base_data()
    data["electrode_geometry"]["mesh_file"] = "mesh.stl"
    data["electrode_geometry"]["mesh_file_units"] = "cm"
    cfg = AmrexSettings.model_validate(data)
    scaled = cfg.normalize_units("m")
    assert scaled.electrode_geometry.mesh_file_units == "m"


def test_hash_stability_on_geometry_change():
    d1 = base_data()
    cfg1 = AmrexSettings.model_validate(d1)
    h1 = cfg1.hash_amrex_config()
    d2 = base_data()
    d2["electrode_geometry"]["cathode_bar_count"] = 8
    cfg2 = AmrexSettings.model_validate(d2)
    h2 = cfg2.hash_amrex_config()
    assert h1 != h2


def test_refinement_ratio_applied_correctly():
    data = base_data()
    data.update({"tile_size_override": [16, 16, 8], "coarse_block_size": [32, 32, 16]})
    cfg = AmrexSettings.model_validate(data)
    assert cfg.coarse_block_size == (32, 32, 16)
    data_bad = base_data()
    data_bad.update({"tile_size_override": [16, 16, 8], "coarse_block_size": [30, 32, 16]})
    with pytest.raises(ValueError):
        AmrexSettings.model_validate(data_bad)


def test_summary_outputs_expected_fields():
    data = base_data()
    data.update({
        "embedded_boundary": True,
        "embedded_boundary_extrapolation": "linear",
        "flux_limiter_enabled": True,
        "tile_size_override": [32, 32, 1],
        "coarse_block_size": [64, 64, 32],
        "max_grid_size": 128,
        "erosion_mechanisms_enabled": ["thermal", "sputtering"],
    })
    cfg = AmrexSettings.model_validate(data)
    summary = cfg.summarize()
    assert "AMReX:" in summary
    assert "EB:" in summary
    assert "Tile:" in summary
    assert "Electrodes:" in summary
