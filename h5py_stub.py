"""Lightweight ``h5py`` stub for test environments."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np


class _FakeDataset:
    def __init__(self, array: Any):
        arr = np.array(array)
        try:
            self.data = arr.tolist()
        except Exception:  # pragma: no cover - fallback for unusual types
            self.data = getattr(arr, "data", arr)
        self.attrs: Dict[str, Any] = {}

    def __getitem__(self, key: Any) -> Any:  # pragma: no cover - trivial
        return np.array(self.data)


class _FakeGroup:
    def __init__(self) -> None:
        self._items: Dict[str, Any] = {}
        self.attrs: Dict[str, Any] = {}

    def create_dataset(self, name: str, data: Any) -> _FakeDataset:
        ds = _FakeDataset(data)
        self._items[name] = ds
        return ds

    def require_group(self, name: str) -> "_FakeGroup":
        grp = self
        for part in name.split("/"):
            node = grp._items.get(part)
            if not isinstance(node, _FakeGroup):
                node = _FakeGroup()
                grp._items[part] = node
            grp = node
        return grp

    def __getitem__(self, key: str) -> Any:
        node: Any = self
        for part in key.split("/"):
            node = node._items[part]
        return node

    def __contains__(self, key: str) -> bool:
        try:
            self.__getitem__(key)
        except KeyError:
            return False
        return True


class _FakeFile(_FakeGroup):
    def __init__(self, path: str | Path, mode: str = "r") -> None:
        super().__init__()
        self.path = Path(path)
        self.mode = mode
        if "r" in mode:
            try:
                with open(self.path, "rb") as fh:
                    obj = pickle.load(fh)
                    self.attrs = obj.attrs
                    self._items = obj._items
            except FileNotFoundError:  # pragma: no cover - empty file
                pass

    def flush(self) -> None:  # pragma: no cover - no-op
        pass

    def close(self) -> None:  # pragma: no cover - write on close
        if "w" in self.mode or "a" in self.mode:
            with open(self.path, "wb") as fh:
                pickle.dump(self, fh)

    # Context manager protocol -------------------------------------------------
    def __enter__(self) -> "_FakeFile":  # pragma: no cover - trivial
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial
        self.close()


File = _FakeFile

__all__ = ["File"]

# Register under the expected package name so ``import h5py`` works
sys.modules.setdefault("h5py", sys.modules[__name__])
