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
import json
import warnings
import base64
import io
import numpy as np

from .project_manager import ProjectManager
from ..core.config import DPFConfig
from ..device_profiles import DeviceProfiles
from ..optimization import OptimizationWarning
from ..visualization.sheath import jxb_field

try:  # pragma: no cover - optional dependency
    import dash
    from dash import Dash, dcc, html, Input, Output, State, no_update
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
except Exception:  # pragma: no cover - allow import without dash
    Dash = None  # type: ignore[misc]


HELP_TEXT_PATH = Path(__file__).with_name("help_text.json")
try:  # pragma: no cover - helper file may not exist
    HELP_TEXT = json.loads(HELP_TEXT_PATH.read_text())
except Exception:  # pragma: no cover - fallback when file missing
    HELP_TEXT = {}


def _ensure_dash() -> None:
    """Raise if :mod:`dash` is unavailable."""

    if Dash is None:  # pragma: no cover - exercised only when dash missing
        raise RuntimeError("dash is required for the interactive GUI")


def launch(host: str = "127.0.0.1", port: int = 8050, *, simplified: bool = False) -> None:
    """Launch the Dash-based GUI.

    Parameters
    ----------
    host, port:
        Network location where the server should listen.
    """

    _ensure_dash()

    pm = ProjectManager()
    app = Dash(__name__)

    # Surface optimisation warnings to the user interface
    warnings.simplefilter("always", OptimizationWarning)

    presets = DeviceProfiles.with_defaults().devices
    preset_options = [{"label": name, "value": name} for name in presets.keys()]

    def _info_span(key: str) -> html.Span:
        """Return a small "what's this?" tooltip span from JSON help text."""

        text = HELP_TEXT.get(key, "")
        return html.Span(
            "What's this?",
            title=text,
            style={
                "textDecoration": "underline",
                "cursor": "help",
                "marginLeft": "4px",
                "fontSize": "smaller",
            },
        )

    app.layout = html.Div(
        [
            html.H1("DPF2 GUI"),
            dcc.Tabs(
                id="phase_tabs",
                value="breakdown",
                children=[
                    dcc.Tab(
                        label="Breakdown",
                        value="breakdown",
                        children=[
                            html.P(
                                [
                                    "Initiate plasma by ionizing the fill gas.",
                                    _info_span("breakdown"),
                                ]
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Rundown",
                        value="rundown",
                        children=[
                            html.P(
                                [
                                    "Current drives the plasma sheath toward the axis.",
                                    _info_span("rundown"),
                                ]
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Pinch",
                        value="pinch",
                        children=[
                            html.P(
                                [
                                    "Final compression yields peak conditions and neutrons.",
                                    _info_span("pinch"),
                                ]
                            )
                        ],
                    ),
                ],
            ),
            dcc.Dropdown(
                id="preset",
                options=preset_options,
                placeholder="Device preset",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label(
                                [
                                    "Anode radius (cm)",
                                    _info_span("anode_radius"),
                                ]
                            ),
                            dcc.Input(id="anode_radius", type="number"),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label(
                                [
                                    "Cathode radius (cm)",
                                    _info_span("cathode_radius"),
                                ]
                            ),
                            dcc.Input(id="cathode_radius", type="number"),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label(
                                [
                                    "Electrode length (cm)",
                                    _info_span("electrode_length"),
                                ]
                            ),
                            dcc.Input(id="electrode_length", type="number"),
                        ]
                    ),
                ],
                style={"display": "flex", "gap": "1em"},
            ),
            html.Div(
                [
                    html.Label(
                        [
                            "Charging Voltage",
                            _info_span("charging_voltage"),
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
                ]
            ),
            html.Div(
                [
                    html.Label(
                        [
                            "Fill Pressure",
                            _info_span("initial_pressure"),
                        ]
                    ),
                    dcc.Slider(
                        0.1,
                        5.0,
                        0.1,
                        value=1.0,
                        id="pressure",
                        tooltip={"placement": "bottom"},
                    ),
                ]
            ),
            html.Button("Sweep Voltage", id="sweep_voltage"),
            html.Button("Sweep Pressure", id="sweep_pressure"),
            html.Button(
                "Overlay Runs", id="overlay_runs", style={} if not simplified else {"display": "none"}
            ),
            html.Button(
                "Pareto Search", id="pareto", style={} if not simplified else {"display": "none"}
            ),
            html.Button(
                "Export Metrics", id="export", style={} if not simplified else {"display": "none"}
            ),
            html.Button(
                "Export Overlay",
                id="export_overlay",
                style={} if not simplified else {"display": "none"},
            ),
            html.Button("Save Scene", id="save_scene"),
            dcc.Upload(id="load_scene", children=html.Button("Load Scene"), multiple=False),
            dcc.Download(id="download_scene"),
            dcc.Graph(id="metrics_plot"),
            html.H2("Sheath Overlay"),
            dcc.Graph(id="sheath_overlay"),
            html.Hr(),
            html.H2("Geometry"),
            dcc.Upload(
                id="geom_upload",
                children=html.Button("Upload Geometry"),
                multiple=False,
            ),
            html.Div(
                [
                    dcc.Input(id="geom_label", placeholder="label", type="text"),
                    dcc.Input(id="geom_dx", type="number", placeholder="dx"),
                    dcc.Input(id="geom_dy", type="number", placeholder="dy"),
                    dcc.Input(id="geom_dz", type="number", placeholder="dz"),
                    html.Button("Translate", id="geom_translate"),
                ],
                style={"display": "flex", "gap": "0.5em"},
            ),
            dcc.Graph(id="geometry_view"),
            html.H2("Circuit"),
            html.Div(
                [
                    dcc.Input(id="node_a", type="text", placeholder="node A"),
                    dcc.Input(id="node_b", type="text", placeholder="node B"),
                    dcc.Input(id="component", type="text", placeholder="component"),
                    html.Button("Add Component", id="add_component"),
                ],
                style={"display": "flex", "gap": "0.5em"},
            ),
            dcc.Graph(id="circuit_view"),
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

    def _overlay_figure() -> go.Figure:
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

    @app.callback(
        Output("metrics_plot", "figure"),
        Output("download_scene", "data"),
        Input("sweep_voltage", "n_clicks"),
        Input("sweep_pressure", "n_clicks"),
        Input("overlay_runs", "n_clicks"),
        Input("pareto", "n_clicks"),
        Input("export", "n_clicks"),
        Input("export_overlay", "n_clicks"),
        Input("save_scene", "n_clicks"),
        Input("load_scene", "contents"),
        State("load_scene", "filename"),
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
        s_clicks: int,
        load_contents: str | None,
        load_name: str | None,
        preset: str | None,
        pressure: float,
        voltage: float,
        anode: float | None,
        cathode: float | None,
        length: float | None,
    ):
        ctx = dash.callback_context
        if not ctx.triggered:
            return go.Figure(), no_update
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if button_id == "export":
            pm.export_metrics(Path("metrics.csv"))
            return go.Figure(), no_update

        if button_id == "export_overlay":
            pm.overlay_metrics(Path("overlay.png"))
            return go.Figure(), no_update

        if button_id == "save_scene":
            data = pm.export_scene(Path("scene.json")).read_text()
            return go.Figure(), dict(content=data, filename="scene.json")

        if button_id == "load_scene":
            if load_contents and load_name:
                decoded = base64.b64decode(load_contents.split(",", 1)[1]).decode()
                tmp = Path("uploads") / load_name
                tmp.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_text(decoded)
                pm.import_scene(tmp)
                fig = _overlay_figure()
                return fig, no_update
            return go.Figure(), no_update

        if button_id == "overlay_runs":
            fig = _overlay_figure()
            return fig, no_update

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
            return fig, no_update

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
        fig = go.Figure(go.Scatter(x=s_vals, y=y_vals, mode="lines+markers"))
        fig.update_xaxes(title_text="S")
        fig.update_yaxes(title_text="Yield")
        return fig, no_update

    @app.callback(
        Output("geometry_view", "figure"),
        Input("geom_upload", "contents"),
        Input("geom_translate", "n_clicks"),
        State("geom_upload", "filename"),
        State("geom_label", "value"),
        State("geom_dx", "value"),
        State("geom_dy", "value"),
        State("geom_dz", "value"),
        prevent_initial_call=True,
    )
    def _update_geometry(contents, n_clicks, filename, label, dx, dy, dz):
        ctx = dash.callback_context
        if not ctx.triggered:
            return go.Figure()
        trigger = ctx.triggered[0]["prop_id"].split(".")[0]
        lbl = label or "geom"
        if trigger == "geom_upload" and contents and filename:
            data = contents.split(",", 1)[1]
            decoded = base64.b64decode(data)
            tmp = Path("uploads") / filename
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(decoded)
            pm.import_geometry(lbl, tmp)
        elif trigger == "geom_translate" and lbl in pm.geometries:
            pm.transform_geometry(lbl, (dx or 0.0, dy or 0.0, dz or 0.0))
        if lbl in pm.geometries:
            return pm.geometry_figure(lbl)
        return go.Figure()

    @app.callback(
        Output("sheath_overlay", "figure"),
        Input("voltage", "value"),
        Input("pressure", "value"),
    )
    def _update_sheath_overlay(voltage, pressure):
        field = jxb_field(voltage, pressure, 0.0)
        fig = ff.create_quiver(
            field.x.ravel(),
            field.y.ravel(),
            field.u.ravel(),
            field.v.ravel(),
            scale=0.2,
        )
        v_norm = voltage / 30_000.0
        radius = 0.2 + v_norm * 0.6
        theta = np.linspace(0, 2 * np.pi, 100)
        fig.add_trace(
            go.Scatter(x=radius * np.cos(theta), y=radius * np.sin(theta), mode="lines")
        )
        fig.update_layout(
            xaxis=dict(scaleanchor="y", range=[-1, 1]),
            yaxis=dict(range=[-1, 1]),
        )
        return fig

    @app.callback(
        Output("circuit_view", "figure"),
        Input("add_component", "n_clicks"),
        State("node_a", "value"),
        State("node_b", "value"),
        State("component", "value"),
        prevent_initial_call=True,
    )
    def _update_circuit(n_clicks, a, b, comp):
        pm.add_component(comp or "comp", a or "A", b or "B")
        return pm.circuit_figure()

    app.run_server(host=host, port=port)


__all__ = ["launch"]
