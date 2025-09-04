"""Dash-based interactive GUI for parametric studies.

This module provides a lightweight web interface using Plotly Dash that
exposes geometry presets, parameter sliders and run management utilities.  It
builds upon :class:`dpf2.gui.project_manager.ProjectManager` and the parametric
sweep helpers in :mod:`dpf2.optimization.param_sweep`.

The interface is intentionally simple and is designed for exploratory studies
rather than production work.  Dependencies are optional; if :mod:`dash` is not
installed, an informative :class:`RuntimeError` is raised when attempting to
launch the application.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from .project_manager import ProjectManager
from ..core.config import DPFConfig
from ..device_profiles import DeviceProfiles
from ..optimization.param_sweep import run_parametric_sweep, compute_sweep_metrics

try:  # pragma: no cover - optional dependency
    import dash
    from dash import Dash, dcc, html, Input, Output, State
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - allow import without dash
    Dash = None  # type: ignore[misc]


def _ensure_dash() -> None:
    """Raise if :mod:`dash` is unavailable."""

    if Dash is None:  # pragma: no cover - exercised only when dash missing
        raise RuntimeError("dash is required for the interactive GUI")


def launch(host: str = "127.0.0.1", port: int = 8050) -> None:
    """Launch the Dash-based GUI.

    Parameters
    ----------
    host, port:
        Network location where the server should listen.
    """

    _ensure_dash()

    pm = ProjectManager()
    app = Dash(__name__)

    presets = DeviceProfiles.with_defaults().devices
    preset_options = [
        {"label": name, "value": name} for name in presets.keys()
    ]

    app.layout = html.Div(
        [
            html.H1("DPF2 GUI"),
            dcc.Dropdown(id="preset", options=preset_options, placeholder="Geometry preset"),
            dcc.Slider(5_000, 30_000, 1_000, value=10_000, id="voltage", tooltip={"placement": "bottom"}),
            dcc.Slider(0.1, 5.0, 0.1, value=1.0, id="pressure", tooltip={"placement": "bottom"}),
            html.Button("Sweep Voltage", id="sweep_voltage"),
            html.Button("Sweep Pressure", id="sweep_pressure"),
            html.Button("Overlay Runs", id="overlay_runs"),
            html.Button("Export Metrics", id="export"),
            dcc.Graph(id="metrics_plot"),
        ]
    )

    def _make_config(preset: str | None, pressure: float, voltage: float) -> DPFConfig:
        cfg = DPFConfig()
        cfg.initial_pressure = pressure
        cfg.charging_voltage = voltage
        if preset and preset in presets:
            dev = presets[preset]
            cfg.anode_radius = dev.anode_radius_cm * 0.01
            cfg.cathode_radius = dev.cathode_radius_cm * 0.01
            cfg.electrode_length = dev.anode_length_cm * 0.01
        return cfg

    @app.callback(
        Output("metrics_plot", "figure"),
        Input("sweep_voltage", "n_clicks"),
        Input("sweep_pressure", "n_clicks"),
        Input("overlay_runs", "n_clicks"),
        Input("export", "n_clicks"),
        State("preset", "value"),
        State("pressure", "value"),
        State("voltage", "value"),
        prevent_initial_call=True,
    )
    def _run_actions(v_clicks: int, p_clicks: int, o_clicks: int, e_clicks: int,
                     preset: str | None, pressure: float, voltage: float):
        ctx = dash.callback_context
        if not ctx.triggered:
            return go.Figure()
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if button_id == "export":
            pm.export_metrics(Path("metrics.csv"))
            return go.Figure()

        if button_id == "overlay_runs":
            fig = go.Figure()
            for label, metrics in pm.metrics.items():
                vals = sorted(metrics.keys())
                fig.add_trace(go.Scatter(x=vals, y=[metrics[v]["yield"] for v in vals],
                                         mode="lines+markers", name=f"{label} yield"))
                fig.add_trace(go.Scatter(x=vals, y=[metrics[v]["pinch_time"] for v in vals],
                                         mode="lines+markers", name=f"{label} pinch"))
                fig.add_trace(go.Scatter(x=vals, y=[metrics[v]["efficiency"] for v in vals],
                                         mode="lines+markers", name=f"{label} eff"))
            return fig

        cfg = _make_config(preset, pressure, voltage)

        if button_id == "sweep_voltage":
            values: List[float] = [voltage * f for f in [0.8, 1.0, 1.2]]
            results = run_parametric_sweep(cfg, "charging_voltage", values)
            metrics = compute_sweep_metrics(cfg, results)
            pm.metrics[f"voltage_{len(pm.metrics)}"] = metrics
            vals = sorted(metrics.keys())
        else:
            values = [pressure * f for f in [0.5, 1.0, 1.5]]
            results = run_parametric_sweep(cfg, "initial_pressure", values)
            metrics = compute_sweep_metrics(cfg, results)
            pm.metrics[f"pressure_{len(pm.metrics)}"] = metrics
            vals = sorted(metrics.keys())

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=vals, y=[metrics[v]["yield"] for v in vals],
                                 mode="lines+markers", name="yield"))
        fig.add_trace(go.Scatter(x=vals, y=[metrics[v]["pinch_time"] for v in vals],
                                 mode="lines+markers", name="pinch time"))
        fig.add_trace(go.Scatter(x=vals, y=[metrics[v]["efficiency"] for v in vals],
                                 mode="lines+markers", name="efficiency"))
        return fig

    app.run_server(host=host, port=port)


__all__ = ["launch"]
