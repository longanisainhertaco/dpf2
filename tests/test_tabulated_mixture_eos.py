import sys
from pathlib import Path

import h5py
import numpy as np

# Ensure Simulation modules are importable
sys.path.append(str(Path(__file__).resolve().parents[1] / "Simulation"))
from eos import TabulatedEOS  # type: ignore


def _create_species_file(tmp_path: Path, species: str, p_val: float, e_val: float) -> Path:
    path = tmp_path / f"{species}.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("rho", data=np.array([1.0, 2.0]))
        f.create_dataset("T", data=np.array([3.0, 4.0]))
        f.create_dataset("p", data=np.full((2, 2), p_val))
        f.create_dataset("e", data=np.full((2, 2), e_val))
    return path


def test_mixed_eos_weighted_combination(tmp_path):
    path_a = _create_species_file(tmp_path, "A", p_val=1.0, e_val=10.0)
    path_b = _create_species_file(tmp_path, "B", p_val=2.0, e_val=20.0)
    fractions = {"A": 0.25, "B": 0.75}

    mix_eos = TabulatedEOS(
        filename={"A": str(path_a), "B": str(path_b)},
        mixture_fractions=fractions,
    )
    eos_a = TabulatedEOS(str(path_a))
    eos_b = TabulatedEOS(str(path_b))

    rho = np.array([1.0])
    T = np.array([3.0])

    expected_p = fractions["A"] * eos_a.ion_pressure(rho, T) + fractions["B"] * eos_b.ion_pressure(rho, T)
    expected_e = fractions["A"] * eos_a.ion_energy(rho, T) + fractions["B"] * eos_b.ion_energy(rho, T)

    np.testing.assert_allclose(mix_eos.ion_pressure(rho, T), expected_p)
    np.testing.assert_allclose(mix_eos.ion_energy(rho, T), expected_e)
