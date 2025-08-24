import sys
from pathlib import Path

import pytest

from dpf2.simulation.circuit import CircuitModel  # type: ignore


class DummyCollision:
    def spitzer_resistivity(self, *args, **kwargs):
        return 0.0


class DummyFieldManager:
    def get_J(self):
        """Return a zero current density for testing."""
        return 0.0


def test_negative_parameters_raise():
    with pytest.raises(ValueError, match="non-negative"):
        CircuitModel(
            C=-1.0,
            L0=1.0,
            R0=1.0,
            anode_radius=0.01,
            cathode_radius=0.02,
            collision_model=DummyCollision(),
            field_manager=DummyFieldManager(),
        )


def test_anode_radius_must_be_smaller():
    with pytest.raises(ValueError, match="Anode radius must be smaller"):
        CircuitModel(
            C=1.0,
            L0=1.0,
            R0=1.0,
            anode_radius=0.02,
            cathode_radius=0.01,
            collision_model=DummyCollision(),
            field_manager=DummyFieldManager(),
        )


def test_collision_model_requires_spitzer():
    class BadCollision:
        """Collision model lacking the required spitzer_resistivity."""

    with pytest.raises(ValueError, match="spitzer_resistivity"):
        CircuitModel(
            C=1.0,
            L0=1.0,
            R0=1.0,
            anode_radius=0.01,
            cathode_radius=0.02,
            collision_model=BadCollision(),
            field_manager=DummyFieldManager(),
        )
