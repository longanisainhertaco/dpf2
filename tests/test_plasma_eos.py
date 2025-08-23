import sys
from pathlib import Path

import pytest
import h5py

# Ensure Simulation modules are importable
sys.path.append(str(Path(__file__).resolve().parents[1] / "Simulation"))
from eos import TabulatedEOS  # type: ignore


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
