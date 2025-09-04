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

try:  # pragma: no cover - optional dependency
    import dash
    from dash import Dash, dcc, html, Input, Output, State
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
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
            dcc.Dropdown(
                id="preset",
                options=preset_options,
                placeholder="Device preset",
            ),
            html.Div(
                [
                    dcc.Input(
                        id="anode_radius",
                        type="number",
                        placeholder="Anode radius (cm)",
                    ),
                    dcc.Input(
                        id="cathode_radius",
                        type="number",
                        placeholder="Cathode radius (cm)",
                    ),
                    dcc.Input(
                        id="electrode_length",
                        type="number",
                        placeholder="Electrode length (cm)",
                    ),
                ]
            ),
            dcc.Slider(
                5_000,
                30_000,
                1_000,
                value=10_000,
                id="voltage",
                tooltip={"placement": "bottom"},
            ),
            dcc.Slider(
                0.1,
                5.0,
                0.1,
                value=1.0,
                id="pressure",
                tooltip={"placement": "bottom"},
            ),
            html.Button("Sweep Voltage", id="sweep_voltage"),
            html.Button("Sweep Pressure", id="sweep_pressure"),
            html.Button("Overlay Runs", id="overlay_runs"),
            html.Button("Pareto Search", id="pareto"),
            html.Button("Export Metrics", id="export"),
            html.Button("Export Overlay", id="export_overlay"),
            dcc.Graph(id="metrics_plot"),
        ]
    )

    def _make_config(
        preset: str | None,
        pressure: float,
        voltage: float,
        anode: float | None,
        cathode: float | None,
        length: float | None,
    ) -> DPFConfig:
        cfg = DPFConfig()
        cfg.initial_pressure = pressure
        cfg.charging_voltage = voltage
        if preset and preset in presets:
            dev = presets[preset]
            if anode is None:
                anode = dev.anode_radius_cm
            if cathode is None:
                cathode = dev.cathode_radius_cm
            if length is None:
                length = dev.anode_length_cm
        cfg.anode_radius = (anode or 0.0) * 0.01
        cfg.cathode_radius = (cathode or 0.0) * 0.01
        cfg.electrode_length = (length or 0.0) * 0.01
        return cfg

    @app.callback(
        Output("metrics_plot", "figure"),
        Input("sweep_voltage", "n_clicks"),
        Input("sweep_pressure", "n_clicks"),
        Input("overlay_runs", "n_clicks"),
        Input("pareto", "n_clicks"),
        Input("export", "n_clicks"),
        Input("export_overlay", "n_clicks"),
        State("preset", "value"),
        State("pressure", "value"),
        State("voltage", "value"),
        State("anode_radius", "value"),
        State("cathode_radius", "value"),
        State("electrode_length", "value"),
        prevent_initial_call=True,
    )
    def _run_actions(
        v_clicks: int,
        p_clicks: int,
        o_clicks: int,
        pa_clicks: int,
        e_clicks: int,
        eo_clicks: int,
        preset: str | None,
        pressure: float,
        voltage: float,
        anode: float | None,
        cathode: float | None,
        length: float | None,
    ):
        ctx = dash.callback_context
        if not ctx.triggered:
            return go.Figure()
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if button_id == "export":
            pm.export_metrics(Path("metrics.csv"))
            return go.Figure()

        if button_id == "export_overlay":
            pm.overlay_metrics(Path("overlay.png"))
            return go.Figure()

        if button_id == "overlay_runs":
            fig = make_subplots(
                rows=1,
                cols=3,
                subplot_titles=("Yield", "Pinch Time", "Efficiency"),
            )
            params = {pm.params.get(lbl, "") for lbl in pm.metrics}
            x_label = params.pop() if len(params) == 1 else "parameter"
            for label, metrics in pm.metrics.items():
                vals = sorted(metrics.keys())
                fig.add_trace(
                    go.Scatter(
                        x=vals,
                        y=[metrics[v]["yield"] for v in vals],
                        mode="lines+markers",
                        name=f"{label} yield",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=vals,
                        y=[metrics[v].get("pinch_time", 0.0) for v in vals],
                        mode="lines+markers",
                        name=f"{label} pinch",
                    ),
                    row=1,
                    col=2,
                )
                fig.add_trace(
                    go.Scatter(
                        x=vals,
                        y=[metrics[v]["efficiency"] for v in vals],
                        mode="lines+markers",
                        name=f"{label} eff",
                    ),
                    row=1,
                    col=3,
                )
            fig.update_xaxes(title_text=x_label, row=1, col=1)
            fig.update_xaxes(title_text=x_label, row=1, col=2)
            fig.update_xaxes(title_text=x_label, row=1, col=3)
            fig.update_yaxes(title_text="Yield", row=1, col=1)
            fig.update_yaxes(title_text="Pinch Time", row=1, col=2)
            fig.update_yaxes(title_text="Efficiency", row=1, col=3)
            return fig

        if button_id == "pareto":
            cfg = _make_config(preset, pressure, voltage, anode, cathode, length)
            bounds = {
                "charging_voltage": (voltage * 0.8, voltage * 1.2),
                "initial_pressure": (pressure * 0.5, pressure * 1.5),
            }
            pareto = pm.pareto_search(cfg, bounds, n_samples=20)
            fig = go.Figure(
                go.Scatter(
                    x=[p["spot_size"] for p in pareto],
                    y=[p["yield"] for p in pareto],
                    mode="markers",
                )
            )
            fig.update_xaxes(title_text="Spot Size")
            fig.update_yaxes(title_text="Yield")
            return fig

        cfg = _make_config(preset, pressure, voltage, anode, cathode, length)

        if button_id == "sweep_voltage":
            values: List[float] = [voltage * f for f in [0.8, 1.0, 1.2]]
            metrics = pm.run_sweep(
                f"voltage_{len(pm.metrics)}", cfg, "charging_voltage", values
            )
        else:
            values = [pressure * f for f in [0.5, 1.0, 1.5]]
            metrics = pm.run_sweep(
                f"pressure_{len(pm.metrics)}", cfg, "initial_pressure", values
            )

        s_vals = [metrics[v].get("S", 0.0) for v in sorted(metrics)]
        y_vals = [metrics[v]["yield"] for v in sorted(metrics)]
        fig = go.Figure(
            go.Scatter(x=s_vals, y=y_vals, mode="lines+markers")
        )
        fig.update_xaxes(title_text="S")
        fig.update_yaxes(title_text="Yield")
        return fig

    app.run_server(host=host, port=port)


__all__ = ["launch"]
