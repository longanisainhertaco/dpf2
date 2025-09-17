"""GPU-accelerated diagnostic helpers."""

import numpy as np
import types

try:  # optional dependency
    from numba import cuda
except Exception:  # pragma: no cover - fallback when CUDA unavailable
    cuda = types.SimpleNamespace(
        is_available=lambda: False,
        jit=lambda f=None, *a, **k: (
            lambda *args, **kwargs: f(*args, **kwargs) if f else None
        ),
        to_device=lambda arr: arr,
        device_array=lambda n, dtype=None: np.zeros(n),
        grid=lambda x: 0,
        synchronize=lambda: None,
    )


if getattr(cuda, "is_available", lambda: False)():

    @cuda.jit
    def _kinetic_energy_kernel(vel, mass, out):  # pragma: no cover - device code
        i = cuda.grid(1)
        if i < vel.shape[0]:
            vx, vy, vz = vel[i, 0], vel[i, 1], vel[i, 2]
            out[i] = 0.5 * mass * (vx * vx + vy * vy + vz * vz)

else:  # pragma: no cover - CPU fallback when CUDA missing

    def _kinetic_energy_kernel(vel, mass, out):
        pass


def kinetic_energy(vel, mass):
    """Return total kinetic energy using CUDA when available."""
    if getattr(cuda, "is_available", lambda: False)():
        n = vel.shape[0]
        d_vel = cuda.to_device(vel)
        d_out = cuda.device_array(n, dtype=np.float64)
        threads = 128
        blocks = (n + threads - 1) // threads
        _kinetic_energy_kernel[blocks, threads](d_vel, mass, d_out)
        cuda.synchronize()
        return float(d_out.copy_to_host().sum())
    else:
        return float(0.5 * mass * np.sum(vel**2))


class GPUKineticEnergyDiagnostic:
    """Minimal diagnostic recording kinetic energy for a species."""

    def __init__(self, species: str):
        self.species = species
        self.data = []

    def record(self, state):
        if not hasattr(state, "species"):
            return
        sp = state.species.get(self.species)
        if sp is None or "vel" not in sp:
            return
        ke = kinetic_energy(sp["vel"], sp.get("m", 1.0))
        self.data.append({"ke": ke})
