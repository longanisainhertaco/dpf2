import importlib.util
from pathlib import Path
import csv
import sys

# Load kinetics module directly
root = Path(__file__).resolve().parents[2]
kin_path = root / "src" / "dpf2" / "chemistry" / "kinetics.py"
spec = importlib.util.spec_from_file_location("kinetics", kin_path)
kinetics_mod = importlib.util.module_from_spec(spec)
sys.modules["kinetics"] = kinetics_mod
spec.loader.exec_module(kinetics_mod)  # type: ignore[attr-defined]

MultiSpeciesTransport = kinetics_mod.MultiSpeciesTransport

def load_dataset():
    path = Path(__file__).resolve().parent / "data" / "transport_ablation.csv"
    diff = {}
    abl = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            diff[row["species"]] = float(row["D"])
            abl[row["species"]] = float(row["ablation"])
    return diff, abl

def test_diffusion_step():
    diff, _ = load_dataset()
    transport = MultiSpeciesTransport(diffusion=diff, dx=1.0)
    n = {"A": [1.0, 0.0], "B": [0.0, 0.0]}
    res = transport.step(n, dt=1.0)
    assert abs(res["A"][0] - 0.9) < 1e-12
    assert abs(res["A"][1] - 0.1) < 1e-12

def test_wall_ablation():
    diff, abl = load_dataset()
    transport = MultiSpeciesTransport(diffusion=diff, dx=1.0)
    n = {"A": [0.0, 0.0], "B": [0.0, 0.0]}
    res = transport.step(n, dt=1.0, wall_ablation=abl)
    assert abs(res["A"][0] - 0.5) < 1e-12
    assert abs(res["B"][0] - 0.0) < 1e-12
