"""Smoke test for WarpX wrapper optional dependencies."""

import pytest


def test_warpx_wrapper_import():
    """Import the WarpX wrapper when dependencies are present."""
    pytest.importorskip("adios2")
    pytest.importorskip("amrex")
    pytest.importorskip("picmi")
    pytest.importorskip("h5py")

    from dpf2.simulation.warpx_wrapper import WarpXWrapper  # noqa: F401
