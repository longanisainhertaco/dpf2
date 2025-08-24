import numpy as np
import h5py
from pathlib import Path

from dpf2.simulation.diagnostics import ThomsonScattering
from dpf2.simulation.utils import FieldManager, SimulationState


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
