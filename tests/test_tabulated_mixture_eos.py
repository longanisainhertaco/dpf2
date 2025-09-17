import h5py_stub as h5py

import numpy as np
import pytest
from pathlib import Path
import importlib.util
import types
import sys

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
spec = importlib.util.spec_from_file_location(
    "dpf2.eos", root / "dpf2" / "eos" / "__init__.py"
)
eos_mod = importlib.util.module_from_spec(spec)
sys.modules["dpf2.eos"] = eos_mod
spec.loader.exec_module(eos_mod)  # type: ignore[attr-defined]

TabulatedEOS = eos_mod.TabulatedEOS


def _create_species_file(
    tmp_path: Path, species: str, p_val: float, e_val: float
) -> Path:
    """Create a simple EOS table for a single species."""

    path = tmp_path / f"{species}.h5"
    rho = np.array([1.0, 2.0])
    T = np.array([10.0, 20.0])
    with h5py.File(path, "w") as f:
        f.create_dataset("rho", data=rho)
        f.create_dataset("T", data=T)
        f.create_dataset("p", data=np.full((2, 2), p_val))
        f.create_dataset("e", data=np.full((2, 2), e_val))
    return path


def test_mixed_eos_weighted_combination(tmp_path):
    path_a = _create_species_file(tmp_path, "A", p_val=1.0, e_val=100.0)
    path_b = _create_species_file(tmp_path, "B", p_val=2.0, e_val=200.0)
    fractions = {"A": 0.25, "B": 0.75}

    mix_eos = TabulatedEOS(
        filename={"A": str(path_a), "B": str(path_b)}, mixture_fractions=fractions
    )
    eos_a = TabulatedEOS(str(path_a))
    eos_b = TabulatedEOS(str(path_b))

    rho = np.array([1.0])
    T_val = np.array([10.0])

    expected_p = fractions["A"] * eos_a.pressure(rho, T_val) + fractions[
        "B"
    ] * eos_b.pressure(rho, T_val)
    expected_e = fractions["A"] * eos_a.energy(rho, T_val) + fractions[
        "B"
    ] * eos_b.energy(rho, T_val)

    np.testing.assert_allclose(mix_eos.pressure(rho, T_val), expected_p)
    np.testing.assert_allclose(mix_eos.energy(rho, T_val), expected_e)


def test_mixed_eos_invalid_fraction_sum(tmp_path):
    path_a = _create_species_file(tmp_path, "A", p_val=1.0, e_val=100.0)
    path_b = _create_species_file(tmp_path, "B", p_val=2.0, e_val=200.0)
    fractions = {"A": 0.2, "B": 0.2}

    with pytest.raises(ValueError):
        TabulatedEOS(
            filename={"A": str(path_a), "B": str(path_b)},
            mixture_fractions=fractions,
        )


def test_mixed_eos_missing_species_data(tmp_path):
    path_a = _create_species_file(tmp_path, "A", p_val=1.0, e_val=100.0)
    fractions = {"A": 0.5, "B": 0.5}

    with pytest.raises(ValueError):
        TabulatedEOS(filename=str(tmp_path), mixture_fractions=fractions)


def test_mixed_eos_string_fractions(tmp_path):
    path_a = _create_species_file(tmp_path, "A", p_val=1.0, e_val=100.0)
    path_b = _create_species_file(tmp_path, "B", p_val=2.0, e_val=200.0)
    mix_eos = TabulatedEOS(
        filename=str(tmp_path),
        mixture_fractions="A:0.5,B:0.5",
    )
    eos_a = TabulatedEOS(str(path_a))
    eos_b = TabulatedEOS(str(path_b))

    rho = np.array([1.0])
    T_val = np.array([10.0])

    expected_p = 0.5 * eos_a.pressure(rho, T_val) + 0.5 * eos_b.pressure(rho, T_val)
    expected_e = 0.5 * eos_a.energy(rho, T_val) + 0.5 * eos_b.energy(rho, T_val)

    np.testing.assert_allclose(mix_eos.pressure(rho, T_val), expected_p)
    np.testing.assert_allclose(mix_eos.energy(rho, T_val), expected_e)


def test_mixed_eos_negative_fraction(tmp_path):
    path_a = _create_species_file(tmp_path, "A", p_val=1.0, e_val=100.0)
    path_b = _create_species_file(tmp_path, "B", p_val=2.0, e_val=200.0)

    with pytest.raises(ValueError):
        TabulatedEOS(
            filename={"A": str(path_a), "B": str(path_b)},
            mixture_fractions={"A": -0.1, "B": 1.1},
        )
