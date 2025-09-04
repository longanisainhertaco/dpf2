import importlib
import sys
import logging

import pytest


def _import_gui(monkeypatch, missing):
    """Reload ``dpf2.gui`` with ``missing`` module unavailable."""

    class Blocker:
        def find_spec(self, fullname, path, target=None):  # pragma: no cover - hook
            if fullname == missing:
                raise ModuleNotFoundError(fullname)
            return None

    blocker = Blocker()
    sys.meta_path.insert(0, blocker)
    sys.modules.pop("dpf2.gui", None)
    sys.modules.pop(missing, None)
    try:
        return importlib.import_module("dpf2.gui")
    finally:  # pragma: no cover - ensure cleanup
        sys.meta_path.remove(blocker)


def _assert_warning(monkeypatch, caplog, missing, expected):
    with caplog.at_level(logging.WARNING):
        _import_gui(monkeypatch, missing)
    assert expected in caplog.text


def test_dashboard_missing_logs_warning(monkeypatch, caplog):
    _assert_warning(
        monkeypatch,
        caplog,
        "dpf2.gui.dashboard",
        "pip install flask",
    )


def test_interactive_missing_logs_warning(monkeypatch, caplog):
    _assert_warning(
        monkeypatch,
        caplog,
        "dpf2.gui.interactive",
        "pip install dash",
    )


def test_qt_missing_logs_warning(monkeypatch, caplog):
    _assert_warning(
        monkeypatch,
        caplog,
        "dpf2.gui.qt_sweep",
        "pip install PyQt5",
    )
