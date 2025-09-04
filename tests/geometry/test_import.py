from pathlib import Path

from dpf2.dpf_config import DPFConfig
from dpf2.grid_resolution import GridResolution


def _make_config(geometry: str, mesh: Path) -> DPFConfig:
    cfg = DPFConfig.with_defaults()
    sc = cfg.simulation_control.model_copy(update={"geometry": geometry})
    gr = GridResolution.with_defaults(geometry)
    eg = cfg.amrex_settings.electrode_geometry.model_copy(update={"mesh_file": mesh})
    amr = cfg.amrex_settings.model_copy(update={"electrode_geometry": eg})
    cfg = cfg.model_copy(update={"simulation_control": sc, "grid_resolution": gr, "amrex_settings": amr})
    return cfg


def test_mesh_import_valid():
    path = Path(__file__).with_name("sample.step")
    cfg = _make_config("3D_Cartesian", path)
    DPFConfig.validate_cross_fields(DPFConfig, cfg)


def test_axisymmetric_stl_import():
    path = Path(__file__).with_name("axisymmetric.stl")
    cfg = _make_config("2D_RZ", path)
    DPFConfig.validate_cross_fields(DPFConfig, cfg)
