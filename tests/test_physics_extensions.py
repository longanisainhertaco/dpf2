try:  # pragma: no cover - fallback when h5py missing
    import h5py  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from tests.h5py_stub import h5py  # type: ignore

import numpy as np
import pytest
import importlib.util
import types
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "src"
pkg = types.ModuleType("dpf2")
pkg.__path__ = [str(root / "dpf2")]
sys.modules.setdefault("dpf2", pkg)

class _StubSimTabulated:
    def __init__(self, filename, mixture_fractions=None):
        import numpy as np, h5py
        def load(path):
            with h5py.File(path, "r") as f:
                return f["rho"][:], f["T"][:], f["p"][:], f["e"][:]
        if mixture_fractions:
            if isinstance(filename, (str, Path)):
                base = Path(filename)
                files = {sp: base / f"{sp}.h5" for sp in mixture_fractions}
            else:
                files = {sp: Path(filename[sp]) for sp in mixture_fractions}
            for i, (sp, path) in enumerate(files.items()):
                rho, T, p, e = load(path)
                w = mixture_fractions[sp]
                if i == 0:
                    self.rho_grid, self.T_grid = rho, T
                    self.p_val = w * p[0, 0]
                    self.e_val = w * e[0, 0]
                else:
                    self.p_val += w * p[0, 0]
                    self.e_val += w * e[0, 0]
        else:
            rho, T, p, e = load(filename)
            self.rho_grid, self.T_grid = rho, T
            self.p_val = p[0, 0]
            self.e_val = e[0, 0]
        self.p_interp = lambda pts: np.full(len(pts), self.p_val)
        self.e_interp = lambda pts: np.full(len(pts), self.e_val)

sys.modules.setdefault("dpf2.simulation.eos", types.ModuleType("dpf2.simulation.eos"))
sys.modules["dpf2.simulation.eos"].TabulatedEOS = _StubSimTabulated

def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

eos_mod = _load("dpf2.eos", root / "dpf2" / "eos" / "__init__.py")
fusion_mod = _load("dpf2.fusion", root / "dpf2" / "fusion.py")
pinch_mod = _load("dpf2.pinch_models", root / "dpf2" / "pinch_models.py")

RealGasEOS = eos_mod.RealGasEOS
bosch_hale_dd = fusion_mod.bosch_hale_dd
SemiAnalyticPinchModel = pinch_mod.SemiAnalyticPinchModel


def _create_species_file(tmp_path, name, p_val):
    rho = np.array([1.0, 2.0])
    T = np.array([10.0, 20.0])
    path = tmp_path / f"{name}.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("rho", data=rho)
        f.create_dataset("T", data=T)
        f.create_dataset("p", data=np.full((2, 2), p_val))
        f.create_dataset("e", data=np.full((2, 2), 1.0))
    return path


def test_real_gas_pressure(tmp_path):
    a = _create_species_file(tmp_path, "A", 1.0)
    b = _create_species_file(tmp_path, "B", 2.0)
    eos = RealGasEOS({"A": a, "B": b}, mixture_fractions={"A": 0.5, "B": 0.5})
    p = eos.pressure(np.array([1.0]), np.array([10.0]))
    assert p == pytest.approx([1.5])


def test_bosch_hale_positive():
    r = bosch_hale_dd(10.0)
    assert r > 0.0


def test_semi_analytic_yield_positive():
    model = SemiAnalyticPinchModel()
    t = np.linspace(0, 1e-6, 10)
    I = np.ones_like(t) * 1e4
    res = model.run(t, I)
    assert res.neutron_yield >= 0.0
