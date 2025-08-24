import sys
import types
import pytest

# Provide a minimal numpy stub for the methods used in these tests
np_stub = types.SimpleNamespace(
    array=lambda x: x,
    power=lambda a, b: a ** b,
    allclose=lambda a, b, rtol=1e-5, atol=1e-8: abs(a - b) <= (atol + rtol * abs(b)),
    pi=3.141592653589793,
    ndarray=object,
)
sys.modules.setdefault("numpy", np_stub)
np = np_stub

# Stub out dependencies not needed for opacity calculations
sys.modules.setdefault("amrex", types.ModuleType("amrex"))
sys.modules.setdefault("h5py", types.ModuleType("h5py"))
sys.modules.setdefault("adios2", types.ModuleType("adios2"))
numba_stub = types.ModuleType("numba")
numba_stub.njit = lambda *a, **k: (lambda f: f)
numba_stub.prange = range
sys.modules.setdefault("numba", numba_stub)
scipy_interp = types.ModuleType("scipy.interpolate")
scipy_interp.RegularGridInterpolator = lambda *a, **k: None
sys.modules.setdefault("scipy", types.ModuleType("scipy"))
sys.modules.setdefault("scipy.interpolate", scipy_interp)
scipy_integrate = types.ModuleType("scipy.integrate")
scipy_integrate.solve_ivp = lambda *a, **k: None
sys.modules.setdefault("scipy.integrate", scipy_integrate)
models_stub = types.ModuleType("models")
models_stub.PhysicsModule = object
models_stub.SimulationState = object
sys.modules.setdefault("models", models_stub)
config_stub = types.ModuleType("config_schema")
config_stub.RadiationConfig = object
sys.modules.setdefault("config_schema", config_stub)

# Minimal pydantic stub to satisfy package imports
pydantic_stub = types.ModuleType("pydantic")
pydantic_stub.BaseModel = object
pydantic_stub.ConfigDict = dict
pydantic_stub.Field = lambda default=None, **kwargs: default
pydantic_stub.root_validator = lambda *a, **k: (lambda f: f)
sys.modules.setdefault("pydantic", pydantic_stub)

pydantic_dc_stub = types.ModuleType("pydantic.dataclasses")
def _dc_decorator(cls=None, **kwargs):
    if cls is None:
        return lambda c: c
    return cls
pydantic_dc_stub.dataclass = _dc_decorator
sys.modules.setdefault("pydantic.dataclasses", pydantic_dc_stub)

# Import RadiationModel directly without triggering package-level side effects
import importlib.util, pathlib

dpf2_stub = types.ModuleType("dpf2")
sim_stub = types.ModuleType("dpf2.simulation")
dpf2_stub.simulation = sim_stub
sys.modules.setdefault("dpf2", dpf2_stub)
sys.modules.setdefault("dpf2.simulation", sim_stub)

rm_path = pathlib.Path(__file__).resolve().parents[1] / "src" / "dpf2" / "simulation" / "radiation_model.py"
spec = importlib.util.spec_from_file_location("dpf2.simulation.radiation_model", rm_path)
rad_mod = importlib.util.module_from_spec(spec)
sys.modules["dpf2.simulation.radiation_model"] = rad_mod
spec.loader.exec_module(rad_mod)

RadiationModel = rad_mod.RadiationModel


def _make_model(model, params):
    rm = RadiationModel.__new__(RadiationModel)
    rm.opacity_model = model
    rm.opacity_params = params
    return rm


def test_constant_opacity():
    rm = _make_model("constant", {"constant_opacity": 2.5})
    out = rm._compute_opacity(Te=0.0, ne=0.0, Z=0.0)
    assert np.allclose(out, np.array(2.5))


def test_temperature_dependent_opacity():
    rm = _make_model("temperature_dependent", {"base": 1.0, "alpha": 0.5, "beta": 2.0})
    Te = 3.0
    expected = 1.0 + 0.5 * Te ** 2.0
    out = rm._compute_opacity(Te=Te, ne=0.0, Z=0.0)
    assert np.allclose(out, expected)


def test_density_dependent_opacity_ne():
    rm = _make_model("density_dependent", {"base": 0.1, "alpha": 0.2, "ne_exponent": 1.5})
    ne = 4.0
    expected = 0.1 + 0.2 * ne ** 1.5
    out = rm._compute_opacity(Te=0.0, ne=ne, Z=0.0)
    assert np.allclose(out, expected)


def test_density_dependent_opacity_Z():
    rm = _make_model("density_dependent", {"base": 0.0, "alpha": 0.3, "Z_exponent": 2.0, "use_Z": True})
    Z = 5.0
    expected = 0.0 + 0.3 * Z ** 2.0
    out = rm._compute_opacity(Te=0.0, ne=0.0, Z=Z)
    assert np.allclose(out, expected)


def test_missing_constant_param():
    rm = _make_model("constant", {})
    with pytest.raises(ValueError):
        rm._compute_opacity(Te=0.0, ne=0.0, Z=0.0)


def test_missing_temperature_param():
    rm = _make_model("temperature_dependent", {"base": 1.0, "alpha": 0.5})
    with pytest.raises(ValueError):
        rm._compute_opacity(Te=1.0, ne=0.0, Z=0.0)


def test_missing_density_param():
    rm = _make_model("density_dependent", {"base": 0.1, "alpha": 0.2})
    with pytest.raises(ValueError):
        rm._compute_opacity(Te=0.0, ne=1.0, Z=0.0)


def test_missing_density_Z_exponent():
    rm = _make_model(
        "density_dependent", {"base": 0.1, "alpha": 0.2, "use_Z": True}
    )
    with pytest.raises(ValueError):
        rm._compute_opacity(Te=0.0, ne=0.0, Z=1.0)


def test_missing_params_object():
    rm = _make_model("constant", None)
    with pytest.raises(ValueError):
        rm._compute_opacity(Te=0.0, ne=0.0, Z=0.0)


def test_unknown_model():
    rm = _make_model("invalid_model", {})
    with pytest.raises(ValueError):
        rm._compute_opacity(Te=0.0, ne=0.0, Z=0.0)
