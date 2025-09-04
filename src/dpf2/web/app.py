"""Minimal Flask application for running DPF simulations."""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Iterable
import uuid

from flask import Flask, redirect, render_template_string, request, url_for, jsonify

from ..core.config import DPFConfig
from ..core.simulation import DPFSimulation
from ..device_profiles import DeviceProfiles
from ..exceptions import ConfigurationError, SimulationRuntimeError
from ..optimization.param_sweep import (
    plot_sweep_results,
    run_parametric_sweep,
    compute_sweep_metrics,
    plot_metric_overlay,
)

# HTML templates rendered with simple ``render_template_string`` calls.  The
# templates expose a subset of configuration parameters and allow users to run
# single simulations, perform parametric sweeps and export configuration files
# for later CLI/HPC execution.
INDEX_HTML = """
<!doctype html>
<title>DPF2 Dashboard</title>
<h1>Run Simulation</h1>
<form method=post>
  Charging voltage: <input name=charging_voltage value="{{ cfg.charging_voltage }}"><br>
  Anode radius (m): <input name=anode_radius value="{{ cfg.anode_radius }}"><br>
  Cathode radius (m): <input name=cathode_radius value="{{ cfg.cathode_radius }}"><br>
  Electrode length (m): <input name=electrode_length value="{{ cfg.electrode_length }}"><br>
  Geometry preset:
  <select name=preset>
    <option value=""></option>
    {% for name in presets %}<option value="{{ name }}">{{ name }}</option>{% endfor %}
  </select><br>
  Sweep parameter: <input name=sweep_param><br>
  Sweep values (comma separated): <input name=sweep_values><br>
  Output directory: <input name=output value="{{ output }}"><br>
  <button name=action value=run>Run</button>
  <button name=action value=sweep>Run Sweep</button>
  <button name=action value=sweep_metrics>Sweep Metrics</button>
  <button name=action value=export>Export Config</button>
</form>
<p><a href="{{ url_for('diagnostics', output=output) }}">View diagnostics</a></p>
"""

DIAG_HTML = """
<!doctype html>
<title>Diagnostics</title>
<h1>Diagnostics</h1>
{% if plot %}<img src="{{ plot }}" alt="sweep plot"><br>{% endif %}
{% if metrics_plot %}<img src="{{ metrics_plot }}" alt="metrics plot"><br>{% endif %}
<ul>
{% for f in files %}<li>{{ f }}</li>{% endfor %}
</ul>
<p><a href="{{ url_for('index', output=request.args.get('output', 'output')) }}">Back</a></p>
"""


def _update_config_from_form(cfg: DPFConfig, form: dict, presets: dict[str, object]) -> DPFConfig:
    """Populate a :class:`DPFConfig` instance from form fields."""

    for field in ["charging_voltage", "anode_radius", "cathode_radius", "electrode_length"]:
        if form.get(field):
            setattr(cfg, field, float(form[field]))

    preset = form.get("preset")
    if preset and preset in presets:
        dev = presets[preset]
        cfg.anode_radius = dev.anode_radius_cm * 0.01
        cfg.cathode_radius = dev.cathode_radius_cm * 0.01
        cfg.electrode_length = dev.anode_length_cm * 0.01

    return cfg


def _parse_sweep_values(vals: str) -> Iterable[float]:
    for v in vals.split(","):
        v = v.strip()
        if not v:
            continue
        yield float(v)


def create_app() -> Flask:
    app = Flask(__name__)
    projects: dict[str, dict] = {}
    sweep_results: dict[str, dict] = {}
    upload_dir = Path("uploads")

    @app.route("/", methods=["GET", "POST"])
    def index():
        cfg = DPFConfig()
        presets = DeviceProfiles.with_defaults().devices
        output = request.form.get("output", "output")

        if request.method == "POST":
            cfg = _update_config_from_form(cfg, request.form, presets)
            action = request.form.get("action", "run")
            try:
                if action == "export":
                    Path(output).mkdir(parents=True, exist_ok=True)
                    cfg.to_file(Path(output) / "config.json")
                    msg = f"Exported configuration to {output}/config.json"
                    return render_template_string(INDEX_HTML + f"<p>{msg}</p>", cfg=cfg, presets=presets.keys(), output=output)
                if action == "sweep":
                    param = request.form.get("sweep_param")
                    values = request.form.get("sweep_values", "")
                    if param and values:
                        vals = list(_parse_sweep_values(values))
                        results = run_parametric_sweep(cfg, param, vals, output_dir=output)
                        plot_path = Path(output) / "sweep_plot.png"
                        plot_sweep_results(param, results, plot_path)
                elif action == "sweep_metrics":
                    param = request.form.get("sweep_param")
                    values = request.form.get("sweep_values", "")
                    if param and values:
                        vals = list(_parse_sweep_values(values))
                        results = run_parametric_sweep(cfg, param, vals, output_dir=output)
                        plot_sweep_results(param, results, Path(output) / "sweep_plot.png")
                        metrics = compute_sweep_metrics(cfg, results)
                        plot_metric_overlay(param, metrics, Path(output) / "sweep_metrics.png")
                else:
                    sim = DPFSimulation(cfg)
                    sim.run(output_dir=output)
                return redirect(url_for("diagnostics", output=output))
            except (ConfigurationError, SimulationRuntimeError, Exception) as exc:  # pragma: no cover - UI path
                return render_template_string(INDEX_HTML + f"<p>Error: {exc}</p>", cfg=cfg, presets=presets.keys(), output=output)

        return render_template_string(INDEX_HTML, cfg=cfg, presets=presets.keys(), output=output)

    @app.route("/diagnostics")
    def diagnostics():
        output = request.args.get("output", "output")
        try:
            files = sorted(os.listdir(output))
        except FileNotFoundError:
            files = []
        plot = None
        plot_path = Path(output) / "sweep_plot.png"
        if plot_path.exists():
            plot = "data:image/png;base64," + base64.b64encode(plot_path.read_bytes()).decode("ascii")
        metrics_plot = None
        metrics_path = Path(output) / "sweep_metrics.png"
        if metrics_path.exists():
            metrics_plot = "data:image/png;base64," + base64.b64encode(metrics_path.read_bytes()).decode("ascii")
        return render_template_string(DIAG_HTML, files=files, plot=plot, metrics_plot=metrics_plot)

    @app.route("/projects", methods=["GET", "POST"])
    def projects_ep():
        if request.method == "POST":
            proj_id = request.form.get("id") or f"proj-{uuid.uuid4().hex[:8]}"
            preset = request.form.get("preset")
            file = request.files.get("cad")
            cfg = {"id": proj_id, "preset": preset}
            if file:
                upload_dir.mkdir(parents=True, exist_ok=True)
                path = upload_dir / f"{proj_id}_{file.filename}"
                file.save(path)
                cfg["cad"] = str(path)
            projects[proj_id] = cfg
            return jsonify(cfg)
        return jsonify(list(projects.values()))

    @app.route("/sweep/<proj_id>", methods=["GET", "POST"])
    def sweep_ep(proj_id: str):
        if request.method == "POST":
            data = request.get_json() or {}
            sweep_results.setdefault(proj_id, {}).update(data)
            return {"status": "ok"}
        return sweep_results.get(proj_id, {})

    return app


if __name__ == "__main__":  # pragma: no cover
    create_app().run(debug=True)
