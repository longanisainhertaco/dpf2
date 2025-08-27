import sys
import types


class _NP:
    """Minimal numpy-like helper for AMR tests."""

    int8 = int

    @staticmethod
    def any(arr):
        data = getattr(arr, "data", arr)
        return any(any(any(v for v in row) for row in plane) for plane in data)


class Density:
    def __init__(self, shape):
        nx, ny, nz = shape
        self.data = [[[0 for _ in range(nz)] for _ in range(ny)] for _ in range(nx)]

    def __gt__(self, thresh):
        mask = [[[1 if v > thresh else 0 for v in row2] for row2 in row1] for row1 in self.data]
        return Mask(mask)


class Mask:
    def __init__(self, data):
        self.data = data

    def astype(self, _):
        return self.data


class DummyState:
    def __init__(self, shape):
        self.density = Density(shape)


def _install_pywarpx_stub():
    warpx_api = types.SimpleNamespace(regrid=lambda: None, write_plotfile=lambda *_: None)
    amr_state = {"tagged": None}

    def tag_cells(mask):
        amr_state["tagged"] = mask

    def tagging_stats():
        data = amr_state["tagged"]
        total = sum(v for plane in data for row in plane for v in row) if data else 0
        return {"tagged_cells": total}

    amr_api = types.SimpleNamespace(tag_cells=tag_cells, tagging_stats=tagging_stats)
    mod = types.ModuleType("pywarpx")
    mod.warpx = warpx_api
    mod.amr = amr_api
    picmi_mod = types.ModuleType("pywarpx.picmi")
    mod.picmi = picmi_mod
    sys.modules["pywarpx"] = mod
    sys.modules["pywarpx.picmi"] = picmi_mod
    return mod, amr_state


def test_refinement_triggers(monkeypatch):
    pyw, amr_state = _install_pywarpx_stub()
    from dpf2.simulation import adaptive_mesh_refinement as amr_mod

    # Replace numpy dependency with minimal helper
    amr_mod.np = _NP

    sim = amr_mod.AdvancedWarpXSimulation.__new__(amr_mod.AdvancedWarpXSimulation)
    sim.config = {
        "enable_amr": True,
        "refinement_threshold": 0.1,
        "ncell": [2, 2, 2],
        "amr_levels": 1,
    }

    state = DummyState((2, 2, 2))
    state.density.data[0][0][0] = 1.0

    called = {}

    def fake_regrid():
        called["regrid"] = True

    pyw.warpx.regrid = fake_regrid

    sim.refine_grid(state)

    assert called.get("regrid")
    assert amr_state["tagged"] is not None
    assert amr_state["tagged"][0][0][0] == 1
