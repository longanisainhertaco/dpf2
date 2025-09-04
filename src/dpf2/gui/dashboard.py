"""Utilities for launching the built-in web dashboard."""

from ..web.app import create_app


def launch(host: str = "127.0.0.1", port: int = 5000, **kwargs) -> None:
    """Start the Flask-based dashboard.

    Parameters
    ----------
    host, port:
        Network location where the application should listen.
    kwargs:
        Additional keyword arguments forwarded to :func:`Flask.run`.
    """

    app = create_app()
    app.run(host=host, port=port, **kwargs)


__all__ = ["launch"]
