"""Minimal Flask application for running DPF simulations."""
from __future__ import annotations

import json
import os
from pathlib import Path

from flask import Flask, redirect, render_template_string, request, url_for

from ..core.config import DPFConfig
from ..core.simulation import DPFSimulation
from ..exceptions import ConfigurationError, SimulationRuntimeError

INDEX_HTML = """
<!doctype html>
<title>DPF2 Dashboard</title>
<h1>Run Simulation</h1>
<form method=post>
  Config file path: <input name=config><br>
  Output directory: <input name=output value="output"><br>
  <input type=submit value="Run">
</form>
<p><a href="{{ url_for('diagnostics') }}">View diagnostics</a></p>
"""

DIAG_HTML = """
<!doctype html>
<title>Diagnostics</title>
<h1>Diagnostics</h1>
<ul>
{% for f in files %}<li>{{ f }}</li>{% endfor %}
</ul>
<p><a href="{{ url_for('index') }}">Back</a></p>
"""


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "POST":
            cfg_path = request.form.get("config")
            output = request.form.get("output", "output")
            try:
                cfg = DPFConfig.from_file(cfg_path) if cfg_path else DPFConfig()
                sim = DPFSimulation(cfg)
                sim.run(output_dir=output)
                return redirect(url_for("diagnostics", output=output))
            except (ConfigurationError, SimulationRuntimeError, Exception) as exc:  # pragma: no cover - UI path
                return render_template_string(INDEX_HTML + f"<p>Error: {exc}</p>")
        return render_template_string(INDEX_HTML)

    @app.route("/diagnostics")
    def diagnostics():
        output = request.args.get("output", "output")
        try:
            files = sorted(os.listdir(output))
        except FileNotFoundError:
            files = []
        return render_template_string(DIAG_HTML, files=files)

    return app


if __name__ == "__main__":  # pragma: no cover
    create_app().run(debug=True)
