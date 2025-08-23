import pytest

from Simulation.module_registry import ModuleRegistry


class NotPhysics:
    pass


def test_register_non_physicsmodule_raises_type_error():
    registry = ModuleRegistry()
    with pytest.raises(TypeError, match="subclass of PhysicsModule"):
        registry.register(NotPhysics)
