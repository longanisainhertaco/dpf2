try:  # optional dependency
    import h5py
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "h5py is required; install dpf2[warpx]"
    ) from exc
import numpy as np
from typing import Dict, Any


class OpenPMDWriter:
    """Minimal openPMD-compliant writer for field and particle data."""

    def __init__(self, filename: str):
        self.filename = str(filename)
        self._file = h5py.File(self.filename, "w")
        self._file.attrs.update(
            {
                "openPMD": "1.1.0",
                "basePath": "/data/%T/",
                "iterationEncoding": "groupBased",
                "iterationFormat": "%T",
                "software": "dpf-simulator",
            }
        )

    def write_fields(self, iteration: int, fields: Dict[str, np.ndarray]):
        grp = self._file.require_group(f"data/{iteration}")
        for name, arr in fields.items():
            ds = grp.create_dataset(name, data=np.asarray(arr))
            ds.attrs["unitSI"] = 1.0
        self._file.flush()

    def write_particles(self, iteration: int, particles: Dict[str, Dict[str, Any]]):
        grp = self._file.require_group(f"data/{iteration}/particles")
        for species, comps in particles.items():
            sgrp = grp.require_group(species)
            for comp, arr in comps.items():
                ds = sgrp.create_dataset(comp, data=np.asarray(arr))
                ds.attrs["unitSI"] = 1.0
        self._file.flush()

    def close(self):
        if self._file:
            self._file.close()
            self._file = None
