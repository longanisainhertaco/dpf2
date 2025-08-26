import importlib
import sys
import textwrap

from dpf2.simulation.module_registry import ModuleRegistry

def test_discover_plugins_subpackage(tmp_path, monkeypatch):
    pkg = tmp_path / "myplugins"
    subpkg = pkg / "subpkg"
    subpkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (subpkg / "__init__.py").write_text("")
    (subpkg / "plugin.py").write_text(
        textwrap.dedent(
            """
            from dpf2.simulation.models import PhysicsModule
            class ExamplePlugin(PhysicsModule):
                def apply(self, state, dt):
                    pass
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    registry = ModuleRegistry()
    registry.discover_plugins("myplugins")
    from myplugins.subpkg.plugin import ExamplePlugin
    assert ExamplePlugin in registry.modules
