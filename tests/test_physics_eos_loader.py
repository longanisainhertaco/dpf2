import json

from dpf2.physics.eos import load_tabulated_eos, load_standard_eos


def test_standard_loader():
    eos = load_standard_eos("argon")
    val = eos.pressure_at(1.5, 15.0)
    assert abs(val - 1.75) < 1e-12


def test_opacity_loading(tmp_path):
    data = {
        "rho": [1.0, 2.0],
        "T": [10.0, 20.0],
        "p": [[1.0, 1.5], [2.0, 2.5]],
        "e": [[100.0, 150.0], [200.0, 250.0]],
        "opacity": [[0.1, 0.2], [0.3, 0.4]],
    }
    path = tmp_path / "table.json"
    with path.open("w") as f:
        json.dump(data, f)
    eos = load_tabulated_eos(path)
    assert abs(eos.opacity_at(1.0, 10.0) - 0.1) < 1e-12
