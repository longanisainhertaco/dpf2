"""Lightweight web dashboard for running simulations."""

# ``create_app`` depends on optional ``flask``. Import lazily so downstream
# modules like :mod:`lab_mode_api` can be used without the dependency.
try:  # pragma: no cover - exercised in integration tests
    from .app import create_app
    __all__ = ["create_app"]
except Exception:  # pragma: no cover - flask not installed
    __all__: list[str] = []
