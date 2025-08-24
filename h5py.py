"""Lightweight ``h5py`` stub for environments without the real package.

The project only requires a tiny subset of :mod:`h5py` for reading and
writing very small datasets inside the unit tests.  This stub mirrors the
behaviour of the real library sufficiently for the tests to exercise the
EOS table handling without pulling in the heavy dependency.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


class _FakeDataset:
    def __init__(self, array: Any):
        self._array = array

    def __getitem__(self, key: Any) -> Any:  # pragma: no cover - trivial
        return self._array


class _FakeFile:
    def __init__(self, path: str | Path, mode: str = "r") -> None:
        self.path = Path(path)
        self.mode = mode
        if "r" in mode:
            try:
                with open(self.path, "rb") as fh:
                    self._data = pickle.load(fh)
            except FileNotFoundError:  # pragma: no cover - empty initial file
                self._data = {}
        else:
            self._data = {}

    # Context manager protocol -------------------------------------------------
    def __enter__(self) -> "_FakeFile":  # pragma: no cover - trivial
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial
        if "w" in self.mode or "a" in self.mode:
            with open(self.path, "wb") as fh:
                pickle.dump(self._data, fh)

    # Minimal API --------------------------------------------------------------
    def create_dataset(self, name: str, data: Any) -> None:
        self._data[name] = getattr(data, "data", data)

    def __getitem__(self, key: str) -> _FakeDataset:  # pragma: no cover - trivial
        return _FakeDataset(self._data[key])


# Public module-like object ----------------------------------------------------
File = _FakeFile


__all__ = ["File"]

