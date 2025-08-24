"""Tests for parsing mixture fractions in ``select_eos``."""

import sys
from pathlib import Path

try:  # pragma: no cover - fallback when h5py unavailable
    import h5py  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from tests.h5py_stub import h5py  # type: ignore

import numpy as np
import pytest

from dpf2.simulation.eos_selector import select_eos  # type: ignore
from dpf2.simulation.eos import TabulatedEOS  # type: ignore


def _create_species_eos_file(tmp_path: Path, species: str, p_val: float = 1.0, e_val: float = 1.0) -> Path:
    file_path = tmp_path / f"{species}.h5"
    with h5py.File(file_path, "w") as f:
        f.create_dataset("rho", data=np.array([1.0, 2.0]))
        f.create_dataset("T", data=np.array([3.0, 4.0]))
        f.create_dataset("p", data=np.full((2, 2), p_val))
        f.create_dataset("e", data=np.full((2, 2), e_val))
    return file_path


def test_select_eos_valid_mixture(tmp_path):
    _create_species_eos_file(tmp_path, "A")
    _create_species_eos_file(tmp_path, "B")
    eos = select_eos(
        "tabulated",
        table_file=str(tmp_path),
        mixture_fractions="A:0.5,B:0.5",
    )
    assert isinstance(eos, TabulatedEOS)


def test_select_eos_valid_mixture_dict(tmp_path):
    _create_species_eos_file(tmp_path, "A")
    _create_species_eos_file(tmp_path, "B")
    eos = select_eos(
        "tabulated",
        table_file=str(tmp_path),
        mixture_fractions={"A": 0.5, "B": 0.5},
    )
    assert isinstance(eos, TabulatedEOS)


def test_select_eos_mixture_missing_data(tmp_path):
    _create_species_eos_file(tmp_path, "A")
    with pytest.raises(ValueError):
        select_eos(
            "tabulated",
            table_file=str(tmp_path),
            mixture_fractions="A:0.5,B:0.5",
        )


def test_select_eos_invalid_fractions(tmp_path):
    _create_species_eos_file(tmp_path, "A")
    _create_species_eos_file(tmp_path, "B")
    with pytest.raises(ValueError):
        select_eos(
            "tabulated",
            table_file=str(tmp_path),
            mixture_fractions="A:0.3,B:0.3",
        )
