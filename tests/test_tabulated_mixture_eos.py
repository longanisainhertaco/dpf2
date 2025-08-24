import h5py
import numpy as np
from pathlib import Path

from dpf2.eos import TabulatedEOS


def _create_species_file(tmp_path: Path, species: str, p_val: float, e_val: float) -> Path:
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

    expected_p = fractions["A"] * eos_a.pressure(rho, T_val) + fractions["B"] * eos_b.pressure(
        rho, T_val
    )
    expected_e = fractions["A"] * eos_a.energy(rho, T_val) + fractions["B"] * eos_b.energy(
        rho, T_val
    )

    np.testing.assert_allclose(mix_eos.pressure(rho, T_val), expected_p)
    np.testing.assert_allclose(mix_eos.energy(rho, T_val), expected_e)

