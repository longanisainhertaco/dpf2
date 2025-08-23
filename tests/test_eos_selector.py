import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

# Ensure Simulation modules are importable
sys.path.append(str(Path(__file__).resolve().parents[1] / "Simulation"))
from eos_selector import select_eos  # type: ignore


def _create_dummy_eos_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "eos.h5"
    with h5py.File(file_path, "w") as f:
        f.create_dataset("rho", data=np.array([1.0, 2.0]))
        f.create_dataset("T", data=np.array([3.0, 4.0]))
        f.create_dataset("p", data=np.ones((2, 2)))
        f.create_dataset("e", data=np.ones((2, 2)))
    return file_path


def test_select_eos_valid_mixture(tmp_path):
    file_path = _create_dummy_eos_file(tmp_path)
    with pytest.raises(NotImplementedError):
        select_eos(
            "tabulated",
            table_file=str(file_path),
            mixture_fractions="A:0.5,B:0.5",
        )


def test_select_eos_mixture_missing_data(tmp_path):
    file_path = _create_dummy_eos_file(tmp_path)
    with pytest.raises(ValueError):
        select_eos(
            "tabulated",
            table_file=str(file_path),
            mixture_fractions="A:0.5",
        )
