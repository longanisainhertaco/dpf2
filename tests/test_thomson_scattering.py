import numpy as np
import h5py
from pathlib import Path
import importlib.util
import sys

SIM_DIR = Path(__file__).resolve().parents[1] / "Simulation"
sys.path.insert(0, str(SIM_DIR))

# Dynamically load modules from the Simulation directory to avoid conflicts
diag_spec = importlib.util.spec_from_file_location("sim_diag", SIM_DIR / "diagnostics.py")
diag_module = importlib.util.module_from_spec(diag_spec)
diag_spec.loader.exec_module(diag_module)

utils_spec = importlib.util.spec_from_file_location("sim_utils", SIM_DIR / "utils.py")
utils_module = importlib.util.module_from_spec(utils_spec)
utils_spec.loader.exec_module(utils_module)

ThomsonScattering = diag_module.ThomsonScattering
FieldManager = utils_module.FieldManager
SimulationState = utils_module.SimulationState


def make_state(ne, Te):
    """Create a minimal SimulationState with uniform fields."""
    grid_shape = (1, 1, 1)
    dx = dy = dz = 1.0
    domain_lo = (0.0, 0.0, 0.0)
    fm = FieldManager(grid_shape, dx, dy, dz, domain_lo, {})
    state = SimulationState(
        grid_shape,
        dx,
        dy,
        dz,
        domain_lo,
        {},
        electron_density=np.full(grid_shape, ne),
        electron_temperature=np.full(grid_shape, Te),
    )
    return fm, state


def test_thomson_scattering_spectrum(tmp_path):
    fm, state = make_state(1e20, 1e5)
    diag = ThomsonScattering(
        "TS",
        laser_wavelength=532e-9,
        scattering_angle=np.pi / 2,
        position=(0.5, 0.5, 0.5),
        field_manager=fm,
    )

    diag.record(0.0, None, None, state=state)

    assert diag.data, "No diagnostic data recorded"
    entry = diag.data[0]
    assert entry["wavelength"].shape == entry["spectrum"].shape
    assert np.all(entry["spectrum"] >= 0)

    with h5py.File(tmp_path / "out.h5", "w") as h5:
        diag.to_hdf5(h5)
        grp = h5["TS"]
        assert "wavelength" in grp
        assert "spectrum" in grp
        spec = grp["spectrum"][:]
        assert spec.shape[0] == 1
