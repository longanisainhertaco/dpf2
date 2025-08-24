import pytest

from dpf2.simulation.module_registry import ModuleRegistry


class NotPhysics:
    """Simple class that does not inherit from PhysicsModule."""


def test_register_non_physicsmodule_raises_type_error():
    registry = ModuleRegistry()
    msg = f"{NotPhysics.__name__} must be a subclass of PhysicsModule"
    with pytest.raises(TypeError, match=msg):
        registry.register(NotPhysics)
