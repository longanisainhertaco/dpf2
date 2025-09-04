"""Command line interface for DPF2."""
import json
import logging
import dataclasses
from pathlib import Path
from dataclasses import asdict
from typing import Any
import os
import subprocess
import tempfile
import textwrap

import click

from dpf2.core.config import DPFConfig
from dpf2.core.simulation import DPFSimulation
from dpf2.core.bases import CouplingState
from dpf2.diagnostics.synthetic_signals import (
    current_waveform,
    voltage_waveform,
    coupled_current_waveform,
    coupled_voltage_waveform,
    rogowski_signal,
    bdot_signal,
)
from dpf2.synthetic_diagnostics import SyntheticDiagnostics
from dpf2.exceptions import ConfigurationError, SimulationRuntimeError
from dpf2.optimization.param_sweep import (
    run_parametric_sweep,
    plot_sweep_results,
    compute_sweep_metrics,
    plot_metric_overlay,
)
from dpf2.physics.axial_rundown import shock_parameter, plot_shock_parameter
from .errors import format_error


def _prompt_with_range(prompt: str, default: float, minimum: float, maximum: float, tip: str) -> float:
    """Prompt the user for a floating point value within a range.

    Displays ``tip`` whenever the entered value falls outside ``minimum`` and
    ``maximum`` and reprompts until a valid value is provided.
    """

    while True:
        value = click.prompt(prompt, type=float, default=default)
        if minimum <= value <= maximum:
            return value
        click.echo(f"{prompt} must be between {minimum} and {maximum}. {tip}")


def _validate_range(name: str, value: float, minimum: float, maximum: float, tip: str) -> float:
    """Validate that ``value`` lies within the given range.

    Raises ``click.BadParameter`` with a contextual tip on failure.
    """

    if not (minimum <= value <= maximum):
        raise click.BadParameter(
            f"{name} must be between {minimum} and {maximum}. {tip}"
        )
    return value


def _to_float(val: Any) -> float:
    """Best-effort conversion to float supporting stubbed types."""
    try:
        return float(val)
    except TypeError:
        return float(getattr(val, "data", val))


def _launch_notebook() -> None:
    """Launch Jupyter with DPF2 helpers preloaded."""
    startup = textwrap.dedent(
        """
        import matplotlib.pyplot as plt
        from dpf2.core.config import DPFConfig
        from dpf2.core.simulation import DPFSimulation
        from dpf2.synthetic_diagnostics import SyntheticDiagnostics
        print('DPF2 notebook ready: DPFConfig, DPFSimulation, SyntheticDiagnostics loaded')
        """
    )
    with tempfile.NamedTemporaryFile("w", delete=False) as fh:
        fh.write(startup)
    env = os.environ.copy()
    env["PYTHONSTARTUP"] = fh.name
    notebook = Path(__file__).resolve().parents[2] / "examples" / "notebooks" / "quickstart.ipynb"
    try:
        subprocess.run(["jupyter", "notebook", str(notebook)], env=env, check=True)
    except FileNotFoundError:
        raise click.ClickException(
            format_error("NOTEBOOK", "Jupyter is not installed", "Install the 'notebook' package.")
        )
    except subprocess.CalledProcessError as e:
        raise click.ClickException(
            format_error("NOTEBOOK", f"Jupyter exited with code {e.returncode}")
        )

logger = logging.getLogger(__name__)


def build_config_wizard() -> DPFConfig:
    """Interactively build a :class:`DPFConfig` with contextual hints."""

    click.echo("DPF2 configuration wizard\n")
    defaults = DPFConfig()
    click.echo("Geometry presets: mather, filippov or custom.")
    preset = click.prompt(
        "Select geometry preset",
        type=click.Choice(["mather", "filippov", "custom"]),
        default="mather",
        show_choices=True,
    )
    if preset == "filippov":
        defaults = dataclasses.replace(
            defaults, cathode_radius=0.03, anode_radius=0.05, electrode_length=0.08
        )
    elif preset == "mather":
        defaults = dataclasses.replace(
            defaults, cathode_radius=0.02, anode_radius=0.04, electrode_length=0.1
        )

    # --- Device size -------------------------------------------------
    click.echo("Device geometry:")
    click.echo("  Electrode dimensions set the physical scale of the device.")
    cathode_radius = click.prompt(
        "Cathode radius [m]",
        type=click.FloatRange(1e-3, 1.0),
        default=defaults.cathode_radius,
        show_default=True,
    )
    anode_radius = click.prompt(
        "Anode radius [m] (must exceed cathode radius)",
        type=click.FloatRange(1e-3, 1.0),
        default=defaults.anode_radius,
        show_default=True,
    )
    electrode_length = click.prompt(
        "Electrode length [m]",
        type=click.FloatRange(1e-3, 2.0),
        default=defaults.electrode_length,
        show_default=True,
    )

    # --- Fill gas ----------------------------------------------------
    click.echo("\nPlasma fill parameters:")
    gas_type = click.prompt(
        "Fill gas", type=str, default=defaults.gas_type, show_default=True
    )
    initial_pressure = click.prompt(
        "Initial pressure [Pa]",
        type=click.FloatRange(1.0, None),
        default=defaults.initial_pressure,
        show_default=True,
    )

    # --- Capacitor bank ---------------------------------------------
    click.echo("\nExternal circuit:")
    click.echo("Capacitor bank and wiring values influence current rise.")
    capacitance = click.prompt(
        "Capacitance [F]",
        type=click.FloatRange(1e-9, None),
        default=defaults.capacitance,
        show_default=True,
    )
    inductance = click.prompt(
        "Inductance [H]",
        type=click.FloatRange(1e-9, None),
        default=defaults.inductance,
        show_default=True,
    )
    resistance = click.prompt(
        "Resistance [Ohm]",
        type=click.FloatRange(0.0, None),
        default=defaults.resistance,
        show_default=True,
    )
    charging_voltage = click.prompt(
        "Charging voltage [V]",
        type=click.FloatRange(1.0, None),
        default=defaults.charging_voltage,
        show_default=True,
    )

    # --- Advanced options -------------------------------------------
    advanced_cfg: dict[str, float | int] = {}
    if click.confirm("Configure advanced mesh and timing options?", default=False):
        click.echo("\nMesh and solver controls:")
        advanced_cfg["nr_cells"] = click.prompt(
            "Radial cells",
            type=click.IntRange(1, 10000),
            default=defaults.nr_cells,
            show_default=True,
        )
        advanced_cfg["nz_cells"] = click.prompt(
            "Axial cells",
            type=click.IntRange(1, 10000),
            default=defaults.nz_cells,
            show_default=True,
        )
        advanced_cfg["cfl_number"] = click.prompt(
            "CFL number",
            type=click.FloatRange(0.0, 1.0),
            default=defaults.cfl_number,
            show_default=True,
        )
        advanced_cfg["end_time"] = click.prompt(
            "Simulation end time [s]",
            type=click.FloatRange(0.0, None),
            default=defaults.end_time,
            show_default=True,
        )

    cfg_dict = asdict(defaults)
    cfg_dict.update(
        {
            "cathode_radius": cathode_radius,
            "anode_radius": anode_radius,
            "electrode_length": electrode_length,
            "gas_type": gas_type,
            "initial_pressure": initial_pressure,
            "capacitance": capacitance,
            "inductance": inductance,
            "resistance": resistance,
            "charging_voltage": charging_voltage,
        }
    )
    cfg_dict.update(advanced_cfg)
    return DPFConfig(**cfg_dict)


@click.group(invoke_without_command=True)
@click.option(
    "--notebook",
    is_flag=True,
    help="Launch Jupyter notebook with plotting widgets preloaded",
)
@click.pass_context
def main(ctx: click.Context, notebook: bool) -> None:
    """Entry point for the DPF2 command line interface."""
    if notebook:
        _launch_notebook()
        return
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command()
@click.option("-c", "--config", type=click.Path(exists=False), help="Path to config file")
@click.option("-o", "--output", type=click.Path(), default="output", help="Output directory")
@click.option("--voltage", type=float, help="Charging voltage [V]")
@click.option(
    "--segment-length",
    "segment_length",
    type=float,
    help="Electrode segment length [m]",
)
@click.option("--verbose", is_flag=True, help="Report solver progress and energy diagnostics")
@click.option(
    "--live-plot",
    is_flag=True,
    help="Stream current/voltage plots during simulation",
)
@click.option(
    "--synthetic",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Run synthetic diagnostics using configuration file",
)
@click.option(
    "--diagnostics",
    is_flag=True,
    help="Save key waveforms and diagnostics at completion",
)
@click.option("--wizard", is_flag=True, help="Interactive mode to build configuration")
def simulate(
    config: str | None,
    output: str,
    voltage: float | None,
    segment_length: float | None,
    verbose: bool,
    live_plot: bool,
    synthetic: str | None,
    diagnostics: bool,
    wizard: bool,
) -> None:

    """Run a DPF simulation."""
    try:
        if verbose:
            logging.basicConfig(level=logging.INFO)

        if wizard:
            cfg = build_config_wizard()
        else:
            cfg = DPFConfig.from_file(config) if config else DPFConfig()

            default_voltage = getattr(cfg, "charging_voltage", 15000.0)
            default_length = getattr(cfg, "electrode_length", 0.10)

            # Prompt and validate voltage
            if voltage is None:
                if click.get_text_stream("stdin").isatty():
                    voltage = _prompt_with_range(
                        "Charging voltage [V]",
                        default_voltage,
                        1000.0,
                        100000.0,
                        "Tip: values are in volts; try 15000 for 15 kV.",
                    )
                else:
                    voltage = default_voltage
            else:
                voltage = _validate_range(
                    "voltage", voltage, 1000.0, 100000.0, "Check the units (volts)."
                )

            # Prompt and validate segment length
            if segment_length is None:
                if click.get_text_stream("stdin").isatty():
                    segment_length = _prompt_with_range(
                        "Segment length [m]",
                        default_length,
                        0.01,
                        1.0,
                        "Tip: specify meters; e.g. 0.1 for 10 cm.",
                    )
                else:
                    segment_length = default_length
            else:
                segment_length = _validate_range(
                    "segment length",
                    segment_length,
                    0.01,
                    1.0,
                    "Ensure the length is given in metres.",
                )

            if hasattr(cfg, "charging_voltage"):
                cfg.charging_voltage = voltage
            if hasattr(cfg, "electrode_length"):
                cfg.electrode_length = segment_length

        sim = DPFSimulation(cfg)

        live_times: list[float] = []
        live_currents: list[float] = []
        live_voltages: list[float] = []
        plot_backend: tuple | None = None
        if live_plot:
            if not click.get_text_stream("stdout").isatty():
                raise click.ClickException(
                    format_error(
                        "PLOT",
                        "Live plotting requires an interactive terminal",
                        "Run in a real terminal or omit --live-plot.",
                    )
                )
            try:
                import matplotlib.pyplot as mplt

                if hasattr(mplt, "ion") and hasattr(mplt, "subplots"):
                    mplt.ion()
                    fig, ax = mplt.subplots()
                    (line_i,) = ax.plot([], [], label="current")
                    (line_v,) = ax.plot([], [], label="voltage")
                    ax.set_xlabel("time [s]")
                    ax.legend()
                    fig.tight_layout()
                    plot_backend = ("matplotlib", mplt, fig, ax, line_i, line_v)
            except Exception:
                try:
                    import plotext as ptx

                    plot_backend = ("plotext", ptx)
                    ptx.clt()
                except Exception:
                    raise click.ClickException(
                        format_error(
                            "PLOT",
                            "Live plotting requires matplotlib or plotext",
                            "Install matplotlib or plotext to enable --live-plot.",
                        )
                    )

        progress_cb = None
        pbar = None
        if verbose:
            from tqdm import tqdm

            dt0 = cfg.cfl_number * min(sim.mesh.dr, sim.mesh.dz)
            total_steps = int(cfg.end_time / dt0) + 1 if dt0 > 0 else None
            pbar = tqdm(total=total_steps, desc="Simulating", unit="step")

            def _update(step: int, time: float) -> None:
                pbar.update(1)
                pbar.set_postfix(time=f"{time:.3e}s")
                if plot_backend is not None:
                    live_times.append(sim.time)
                    live_currents.append(sim.current)
                    live_voltages.append(sim.voltage)
                    if plot_backend[0] == "matplotlib":
                        _, plt, fig, ax, line_i, line_v = plot_backend
                        line_i.set_data(live_times, live_currents)
                        line_v.set_data(live_times, live_voltages)
                        ax.relim()
                        ax.autoscale_view()
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                    else:
                        _, plt = plot_backend
                        plt.clt()
                        plt.plot(live_times, live_currents, label="current")
                        plt.plot(live_times, live_voltages, label="voltage")
                        plt.xlabel("time [s]")
                        plt.legend()
                        plt.show()

            progress_cb = _update
        elif plot_backend is not None:
            def _update(step: int, time: float) -> None:
                live_times.append(sim.time)
                live_currents.append(sim.current)
                live_voltages.append(sim.voltage)
                if plot_backend[0] == "matplotlib":
                    _, plt, fig, ax, line_i, line_v = plot_backend
                    line_i.set_data(live_times, live_currents)
                    line_v.set_data(live_times, live_voltages)
                    ax.relim()
                    ax.autoscale_view()
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                else:
                    _, plt = plot_backend
                    plt.clt()
                    plt.plot(live_times, live_currents, label="current")
                    plt.plot(live_times, live_voltages, label="voltage")
                    plt.xlabel("time [s]")
                    plt.legend()
                    plt.show()

            progress_cb = _update

        run_kwargs = {"output_dir": output, "verbose": verbose}
        if progress_cb is not None:
            run_kwargs["progress_cb"] = progress_cb
        times, currents, voltages = sim.run(**run_kwargs)

        # Compute and plot axial rundown similarity parameter S
        try:
            S = shock_parameter(currents, cfg.anode_radius, cfg.initial_pressure)
            shock_path = plot_shock_parameter(times, S, Path(output) / "shock_trend.png")
            if verbose:
                click.echo(f"Shock parameter plot written to {shock_path}")
        except Exception:
            if verbose:
                click.echo("Shock parameter plot failed")

        if pbar is not None:
            pbar.close()
        if plot_backend and plot_backend[0] == "matplotlib":
            plt = plot_backend[1]
            try:
                plt.ioff()
            except Exception:
                pass

        if synthetic:
            cfg_data = json.loads(Path(synthetic).read_text())
            diag_cfg = SyntheticDiagnostics.model_validate(cfg_data)
            history = [
                CouplingState(current=i, voltage=v) for i, v in zip(currents, voltages)
            ]
            dt = times[1] - times[0] if len(times) > 1 else 0.0
            outputs: dict[str, list[float]] = {}
            if diag_cfg.synthetic_current_waveform_enabled:
                outputs["current"] = current_waveform(history)
            if diag_cfg.synthetic_voltage_waveform_enabled:
                outputs["voltage"] = voltage_waveform(history)
            if diag_cfg.synthetic_coupled_current_waveform_enabled:
                outputs["coupled_current"] = coupled_current_waveform(history)
            if diag_cfg.synthetic_coupled_voltage_waveform_enabled:
                outputs["coupled_voltage"] = coupled_voltage_waveform(history)
            if diag_cfg.synthetic_rogowski_signal_enabled:
                outputs["rogowski"] = rogowski_signal(history, dt)
            if diag_cfg.synthetic_bdot_signal_enabled:
                outputs["bdot"] = bdot_signal(history, 0.01, dt)
            out_file = Path(output) / "synthetic_signals.json"
            out_file.write_text(json.dumps(outputs))
            click.echo(json.dumps(outputs))

        if diagnostics:
            diag = {
                "time": times,
                "current": currents,
                "voltage": voltages,
                "summary": {
                    "peak_current": max(currents) if currents else 0.0,
                    "peak_voltage": max(voltages) if voltages else 0.0,
                    "final_time": times[-1] if times else 0.0,
                },
            }
            diag_file = Path(output) / "diagnostics.json"
            diag_file.write_text(json.dumps(diag))
            click.echo(f"Diagnostics written to {diag_file}")

        try:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.plot(times, currents, label="current")
            if voltages:
                plt.plot(times, voltages, label="voltage")
            plt.xlabel("time [s]")
            plt.legend()
            plt.tight_layout()
            quicklook = Path(output) / "quicklook.png"
            plt.savefig(quicklook)
            try:  # pragma: no cover - interactive plot optional
                plt.show()
            except Exception:
                pass
            click.echo(f"Plot written to {quicklook}")
        except Exception as e:  # pragma: no cover - plotting optional
            if verbose:
                click.echo(f"Plotting failed: {e}")
    except ConfigurationError as e:
        logger.error("Configuration error: %s", e)
        hint = None
        if getattr(e, "hints", None):
            hint = "; ".join(f"{f}: {h}" for f, h in e.hints.items())
        elif getattr(e, "fields", None):
            hint = f"Check fields: {', '.join(e.fields)}"
        raise click.ClickException(format_error("CONFIG", str(e), hint))
    except SimulationRuntimeError as e:
        logger.error("Simulation error: %s", e)
        raise click.ClickException(format_error("SIMULATION", str(e)))
    except Exception as e:
        logger.exception("Unexpected error running simulation")
        raise click.ClickException(format_error("UNEXPECTED", str(e)))


@main.command()
@click.option("--config", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--dataset", type=str, required=True)
@click.option("--outdir", type=click.Path(file_okay=False), default="validation")
def validate(config: str, dataset: str, outdir: str) -> None:
    """Run a validation simulation and compare with experimental data."""
    from .validate import run_validation
    try:
        from .validate import run_validation

        ok = run_validation(Path(config), dataset, outdir=Path(outdir))
        if not ok:
            raise click.ClickException(format_error("VALIDATION", "Validation failed"))
    except Exception as e:  # pragma: no cover - defensive
        raise click.ClickException(format_error("VALIDATION", str(e)))


@main.command("validate-config")
@click.option(
    "--config", type=click.Path(exists=True, dir_okay=False), required=True
)
def validate_config(config: str) -> None:
    """Validate a configuration file."""
    try:
        DPFConfig.from_file(config)
        click.echo("Configuration is valid")
    except ConfigurationError as e:
        hint = None
        if getattr(e, "hints", None):
            hint = "; ".join(f"{f}: {h}" for f, h in e.hints.items())
        elif getattr(e, "fields", None):
            hint = f"Check fields: {', '.join(e.fields)}"
        raise click.ClickException(format_error("CONFIG", str(e), hint))


@main.command()
@click.option("--input", type=click.Path(file_okay=False), required=True)
@click.option("--output", type=click.Path(), default="plot.png")
def plot(input: str, output: str) -> None:
    """Plot current and voltage from simulation outputs."""
    import h5py
    import matplotlib.pyplot as plt

    files = sorted(Path(input).glob("data_*.h5"))
    if not files:
        raise click.ClickException(format_error("PLOT", f"No HDF5 files found in {input}"))

    times, currents, voltages = [], [], []
    for fname in files:
        with h5py.File(fname, "r") as fh:
            times.append(_to_float(fh["time"][()]))
            if "current" in fh:
                currents.append(_to_float(fh["current"][()]))
            if "voltage" in fh:
                voltages.append(_to_float(fh["voltage"][()]))

    if not currents:
        raise click.ClickException(format_error("PLOT", "No current data available"))

    plt.figure()
    plt.plot(times, currents, label="current")
    if voltages:
        plt.plot(times, voltages, label="voltage")
    plt.xlabel("time [s]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    click.echo(f"Plot written to {output}")


@main.command("plot-run")
@click.option(
    "--run-dir", type=click.Path(file_okay=False, exists=True), required=True
)
@click.option("--output", type=click.Path(), default="plot.png")
def plot_run(run_dir: str, output: str) -> None:
    """Quickly plot current and voltage from an existing run directory."""
    import h5py
    import matplotlib.pyplot as plt

    files = sorted(Path(run_dir).glob("data_*.h5"))
    if not files:
        raise click.ClickException(format_error("PLOT", f"No HDF5 files found in {run_dir}"))

    times, currents, voltages = [], [], []
    for fname in files:
        with h5py.File(fname, "r") as fh:
            times.append(_to_float(fh["time"][()]))
            if "current" in fh:
                currents.append(_to_float(fh["current"][()]))
            if "voltage" in fh:
                voltages.append(_to_float(fh["voltage"][()]))

    if not currents:
        raise click.ClickException(format_error("PLOT", "No current data available"))

    plt.figure()
    plt.plot(times, currents, label="current")
    if voltages:
        plt.plot(times, voltages, label="voltage")
    plt.xlabel("time [s]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    click.echo(f"Plot written to {output}")


@main.command("param-sweep")
@click.option("--config", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--parameter", type=str, required=True)
@click.option("--values", type=float, multiple=True, required=True, help="Values to sweep")
@click.option("--output", type=click.Path(file_okay=False), default="sweep_output")
def param_sweep_cmd(
    config: str, parameter: str, values: tuple[float, ...], output: str
) -> None:
    """Run a parameter sweep and plot current, yield and efficiency overlays."""

    try:
        cfg = DPFConfig.from_file(config)
        results = run_parametric_sweep(cfg, parameter, values, output_dir=output)
        plot_sweep_results(parameter, results, Path(output) / "sweep_plot.png")
        metrics = compute_sweep_metrics(cfg, results)
        plot_metric_overlay(parameter, metrics, Path(output) / "sweep_metrics.png")
    except Exception as e:
        raise click.ClickException(format_error("SWEEP", str(e)))

    click.echo(f"Sweep complete. Results written to {output}")


@main.command()
@click.option(
    "--history",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="JSON file containing a list of CouplingState dictionaries",
)
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False),
    required=False,
    help="Optional synthetic diagnostics configuration JSON",
)
@click.option("--current", is_flag=True, help="Output current waveform")
@click.option("--voltage", is_flag=True, help="Output voltage waveform")
@click.option("--rogowski", is_flag=True, help="Output Rogowski signal")
@click.option("--bdot", is_flag=True, help="Output B-dot signal")
@click.option("--dt", type=float, default=1e-9, help="Time step for derivatives [s]")
@click.option(
    "--radius", type=float, default=0.01, help="Probe radius for B-dot signal [m]"
)
def diagnostics(
    history: str,
    config: str | None,
    current: bool,
    voltage: bool,
    rogowski: bool,
    bdot: bool,
    dt: float,
    radius: float,
) -> None:
    """Generate synthetic diagnostics from a coupling history."""
    try:
        data = json.loads(Path(history).read_text())
        states = [CouplingState(**d) for d in data]

        outputs: dict[str, list[float]] = {}

        if config:
            cfg_data = json.loads(Path(config).read_text())
            cfg = SyntheticDiagnostics.model_validate(cfg_data)
            if cfg.synthetic_current_waveform_enabled:
                outputs["current"] = current_waveform(states)
            if cfg.synthetic_voltage_waveform_enabled:
                outputs["voltage"] = voltage_waveform(states)
            if cfg.synthetic_rogowski_signal_enabled:
                outputs["rogowski"] = rogowski_signal(states, dt)
            if cfg.synthetic_bdot_signal_enabled:
                outputs["bdot"] = bdot_signal(states, radius, dt)

        if current:
            outputs["current"] = current_waveform(states)
        if voltage:
            outputs["voltage"] = voltage_waveform(states)
        if rogowski:
            outputs["rogowski"] = rogowski_signal(states, dt)
        if bdot:
            outputs["bdot"] = bdot_signal(states, radius, dt)

        click.echo(json.dumps(outputs))
    except Exception as e:
        raise click.ClickException(format_error("DIAGNOSTICS", str(e)))


@main.command()
def schema() -> None:
    """Print the configuration schema."""
    fields = {
        f.name: {
            "type": getattr(f.type, "__name__", str(f.type)),
            "default": f.default,
        }
        for f in dataclasses.fields(DPFConfig)
    }
    click.echo(json.dumps(fields, indent=2))


@main.command()
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default="config_generated.json",
    help="File to write generated configuration",
)
def wizard(output: str) -> None:
    """Interactive wizard for building a configuration."""

    cfg = build_config_wizard()
    with open(output, "w") as fh:
        json.dump(asdict(cfg), fh, indent=2)

    click.echo(f"Configuration saved to {output}")


if __name__ == "__main__":
    main()
