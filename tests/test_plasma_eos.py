import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

# Ensure Simulation modules are importable
sys.path.append(str(Path(__file__).resolve().parents[1] / "Simulation"))
from eos import TabulatedEOS  # type: ignore


def _create_species_eos_file(
    tmp_path: Path, species: str, p_val: float = 1.0, e_val: float = 1.0
) -> Path:
    file_path = tmp_path / f"{species}.h5"
    with h5py.File(file_path, "w") as f:
        f.create_dataset("rho", data=np.array([1.0, 2.0]))
        f.create_dataset("T", data=np.array([3.0, 4.0]))
        f.create_dataset("p", data=np.full((2, 2), p_val))
        f.create_dataset("e", data=np.full((2, 2), e_val))
    return file_path


def test_tabulated_eos_missing_dataset(tmp_path):
    file_path = tmp_path / "eos_missing.h5"
    with h5py.File(file_path, "w") as f:
        f.create_dataset("rho", data=[1.0, 2.0])
        f.create_dataset("T", data=[3.0, 4.0])
        # Intentionally omit the 'p' dataset
        f.create_dataset("e", data=[[5.0, 6.0], [7.0, 8.0]])
    with pytest.raises(ValueError, match="missing required datasets"):
        TabulatedEOS(str(file_path))


def test_tabulated_eos_inconsistent_dimensions(tmp_path):
    file_path = tmp_path / "eos_bad.h5"
    with h5py.File(file_path, "w") as f:
        f.create_dataset("rho", data=[1.0, 2.0])
        f.create_dataset("T", data=[3.0, 4.0])
        f.create_dataset("p", data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        f.create_dataset("e", data=[[7.0, 8.0], [9.0, 10.0]])
    with pytest.raises(ValueError, match="inconsistent dimensions"):
        TabulatedEOS(str(file_path))


def test_tabulated_eos_mixture_combination(tmp_path):
    _create_species_eos_file(tmp_path, "A", p_val=1.0, e_val=10.0)
    _create_species_eos_file(tmp_path, "B", p_val=2.0, e_val=20.0)
    mix = TabulatedEOS(str(tmp_path), mixture_fractions="A:0.25,B:0.75")
    assert np.allclose(mix.p_table, 0.25 * 1.0 + 0.75 * 2.0)
    assert np.allclose(mix.e_table, 0.25 * 10.0 + 0.75 * 20.0)


def test_tabulated_eos_invalid_mixture(tmp_path):
    _create_species_eos_file(tmp_path, "A")
    _create_species_eos_file(tmp_path, "B")
    with pytest.raises(ValueError):
        TabulatedEOS(str(tmp_path), mixture_fractions="A:0.6,B:0.5")
