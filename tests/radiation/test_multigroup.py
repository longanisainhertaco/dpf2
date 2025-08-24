import sys
import types
import math
import importlib.util
import pathlib

np_stub = types.SimpleNamespace(
    array=lambda x, dtype=None: ([float(v) for v in x] if isinstance(x, (list, tuple)) else float(x)),
    ndarray=list,
    zeros=lambda n: [0.0] * n if isinstance(n, int) else [[0.0] * n[1] for _ in range(n[0])],
    power=lambda a, b: a ** b,
    allclose=lambda a, b, rtol=1e-5, atol=1e-8: abs(a - b) <= (atol + rtol * abs(b)),
    pi=3.141592653589793,
)
sys.modules.setdefault("numpy", np_stub)

# Stub out the package structure required for relative imports inside the
# modules we load below.
dpf2_stub = types.ModuleType("dpf2")
physics_stub = types.ModuleType("dpf2.physics")
radiation_stub = types.ModuleType("dpf2.radiation")
dpf2_stub.physics = physics_stub
dpf2_stub.radiation = radiation_stub
sys.modules.setdefault("dpf2", dpf2_stub)
sys.modules.setdefault("dpf2.physics", physics_stub)
sys.modules.setdefault("dpf2.radiation", radiation_stub)

base = pathlib.Path(__file__).resolve().parents[2] / "src" / "dpf2"

spec = importlib.util.spec_from_file_location("dpf2.radiation.multigroup", base / "radiation" / "multigroup.py")
multigroup = importlib.util.module_from_spec(spec)
sys.modules["dpf2.radiation.multigroup"] = multigroup
spec.loader.exec_module(multigroup)

spec2 = importlib.util.spec_from_file_location("dpf2.physics.mhd", base / "physics" / "mhd.py")
mhd_mod = importlib.util.module_from_spec(spec2)
sys.modules["dpf2.physics.mhd"] = mhd_mod
spec2.loader.exec_module(mhd_mod)

MultiGroupDiffusion = multigroup.MultiGroupDiffusion
ResistiveMHD = mhd_mod.ResistiveMHD


def test_energy_conservation():
    rad = MultiGroupDiffusion(opacities=[0.1, 0.2])
    mhd = ResistiveMHD()
    U = [0.0] * 9
    U[0] = 1.0  # density
    U[4] = 10.0  # total energy
    total_before = U[4] + sum(sum(g) for g in rad.energy)

    mhd.apply_radiation(U, rad, dt=1.0)

    total_after = U[4] + sum(sum(g) for g in rad.energy)
    assert math.isclose(total_before, total_after)


def test_group_coupling_distribution():
    rad = MultiGroupDiffusion(opacities=[0.1, 0.2])
    mhd = ResistiveMHD()
    U = [0.0] * 9
    U[0] = 1.0
    U[4] = 9.0

    mhd.apply_radiation(U, rad, dt=1.0)

    expected = [0.1 * 9.0, 0.2 * 9.0]
    assert all(math.isclose(rad.energy[g][0], expected[g]) for g in range(2))
    assert math.isclose(U[4], 9.0 - sum(expected))
