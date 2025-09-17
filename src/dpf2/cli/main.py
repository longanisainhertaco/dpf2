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
import random

import numpy as np

import click
import numpy as np
import statistics

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
    angular_neutron_spectrum,
)
from dpf2.diagnostics.neutron_yield import simulate_tof_detectors, save_tof_hdf5
from dpf2.synthetic_diagnostics import SyntheticDiagnostics
from dpf2.exceptions import ConfigurationError, SimulationRuntimeError
from dpf2.diagnostics.thresholds import (
    compute_debye_length,
    check_thresholds,
    plasma_inductance_circuit,
)
from dpf2.optimization.param_sweep import (
    run_parametric_sweep,
    compute_sweep_metrics,
    plot_metric_overlay,
    plot_yield_vs_S,
)
from dpf2.gui.project_manager import ProjectManager
from dpf2.gui import interactive
from dpf2.indexing import build_code_index, write_markdown_index

from dpf2.device_profiles import DeviceProfiles

from dpf2.scaling_laws import sweep_yield_scaling
from dpf2.uq.sampling import latin_hypercube, sobol_sample
from dpf2.uq.analysis import sobol_indices, uncertainty_band

from .errors import format_error
from .lab import write_manifest


def _prompt_with_range(
    prompt: str, default: float, minimum: float, maximum: float, tip: str
) -> float:
    """Prompt the user for a floating point value within a range.

    Displays ``tip`` whenever the entered value falls outside ``minimum`` and
    ``maximum`` and reprompts until a valid value is provided.
    """

    while True:
        value = click.prompt(prompt, type=float, default=default)
        if minimum <= value <= maximum:
            return value
        click.echo(f"{prompt} must be between {minimum} and {maximum}. {tip}")


def _validate_range(
    name: str, value: float, minimum: float, maximum: float, tip: str
) -> float:
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
    notebook = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "notebooks"
        / "quickstart.ipynb"
    )
    try:
        subprocess.run(["jupyter", "notebook", str(notebook)], env=env, check=True)
    except FileNotFoundError:
        raise click.ClickException(
            format_error(
                "NOTEBOOK",
                "Jupyter is not installed",
                "Install the 'notebook' package.",
            )
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
    click.echo(
        "Geometry presets: mather, filippov, tapered, hollow, re-entrant or custom."
    )
    preset = click.prompt(
        "Select geometry preset",
        type=click.Choice(
            ["mather", "filippov", "tapered", "hollow", "re-entrant", "custom"]
        ),
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
    elif preset == "tapered":
        defaults = dataclasses.replace(
            defaults, cathode_radius=0.02, anode_radius=0.04, electrode_length=0.1
        )
    elif preset == "hollow":
        defaults = dataclasses.replace(
            defaults, cathode_radius=0.025, anode_radius=0.05, electrode_length=0.12
        )
    elif preset == "re-entrant":
        defaults = dataclasses.replace(
            defaults, cathode_radius=0.02, anode_radius=0.045, electrode_length=0.1
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

    geometry_params: dict[str, Any] = {"type": preset, "length": electrode_length}
    if preset == "tapered":
        r_top = click.prompt(
            "Top radius [m]",
            type=click.FloatRange(1e-3, 1.0),
            default=anode_radius,
            show_default=True,
        )
        geometry_params.update({"r_base": cathode_radius, "r_top": r_top})
    elif preset == "hollow":
        r_inner = click.prompt(
            "Inner bore radius [m]",
            type=click.FloatRange(0.0, cathode_radius),
            default=cathode_radius / 2,
            show_default=True,
        )
        geometry_params.update({"r_outer": cathode_radius, "r_inner": r_inner})
    else:
        geometry_params.update({"r_outer": anode_radius, "r_inner": cathode_radius})

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
    cfg_dict["geometry"] = geometry_params
    return DPFConfig(**cfg_dict)


@click.group(invoke_without_command=True)
@click.option(
    "--notebook",
    is_flag=True,
    help="Launch Jupyter notebook with plotting widgets preloaded",
)
@click.option(
    "--lab-mode",
    is_flag=True,
    help="Record a reproducibility manifest alongside outputs",
)
@click.option(
    "--student",
    is_flag=True,
    help="Launch the simplified student GUI",
)
@click.pass_context
def main(ctx: click.Context, notebook: bool, lab_mode: bool, student: bool) -> None:
    """Entry point for the DPF2 command line interface."""
    ctx.ensure_object(dict)
    ctx.obj["lab_mode"] = lab_mode
    ctx.obj["student"] = student
    if notebook:
        _launch_notebook()
        return
    if student:
        interactive.launch(simplified=True)
        return
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command()
@click.option(
    "-c", "--config", type=click.Path(exists=False), help="Path to config file"
)
@click.option(
    "-o", "--output", type=click.Path(), default="output", help="Output directory"
)
@click.option("--voltage", type=float, help="Charging voltage [V]")
@click.option(
    "--segment-length",
    "segment_length",
    type=float,
    help="Electrode segment length [m]",
)
@click.option(
    "--verbose", is_flag=True, help="Report solver progress and energy diagnostics"
)
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
@click.option(
    "--device",
    type=click.Choice(sorted(DeviceProfiles.with_defaults().devices.keys())),
    help="Preset device geometry",
)
@click.option("--wizard", is_flag=True, help="Interactive mode to build configuration")
@click.option(
    "--shots",
    type=int,
    default=1,
    show_default=True,
    help="Number of jittered shots to run when lab-mode is enabled",
)
@click.pass_context
def simulate(
    ctx: click.Context,
    config: str | None,
    output: str,
    voltage: float | None,
    segment_length: float | None,
    verbose: bool,
    live_plot: bool,
    synthetic: str | None,
    diagnostics: bool,
    device: str | None,
    wizard: bool,
    shots: int,
) -> None:
    """Run a DPF simulation."""
    try:
        if verbose:
            logging.basicConfig(level=logging.INFO)

        if wizard:
            cfg = build_config_wizard()
        else:
            cfg = DPFConfig.from_file(config) if config else DPFConfig()

        if device:
            presets = DeviceProfiles.with_defaults().devices
            if device not in presets:
                raise click.ClickException(f"Unknown device preset: {device}")
            dev = presets[device]
            cfg.anode_radius = dev.anode_radius_cm * 0.01
            cfg.cathode_radius = dev.cathode_radius_cm * 0.01
            cfg.electrode_length = dev.anode_length_cm * 0.01
            bank = dev.capacitor_bank
            cfg.capacitance = bank.get("C", cfg.capacitance)
            cfg.inductance = bank.get("L", cfg.inductance)
            cfg.resistance = bank.get("R", cfg.resistance)
            cfg.gas_type = dev.working_gas
            C = bank.get("C")
            if C:
                cfg.charging_voltage = (2.0 * dev.energy_kJ * 1000.0 / C) ** 0.5
            elif dev.breakdown_voltage_kV:
                cfg.charging_voltage = dev.breakdown_voltage_kV * 1000.0

        default_voltage = getattr(cfg, "charging_voltage", 15000.0)
        default_length = getattr(cfg, "electrode_length", 0.10)

        if not wizard:
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
                    "voltage", voltage, 1000.0, 100000.0, "Check the units (volts).",
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

        warnings_list: list[str] = []

        seeds = {"python": random.getstate()[1][0]}
        try:
            seeds["numpy"] = int(np.random.get_state()[1][0])
        except Exception:
            try:
                rng = np.random.default_rng()
                seeds["numpy"] = int(rng.bit_generator.state["state"]["state"])
            except Exception:
                seeds["numpy"] = 0

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

        run_kwargs = {"output_dir": output, "verbose": verbose, "seeds": seeds}
        if progress_cb is not None:
            run_kwargs["progress_cb"] = progress_cb
        times, currents, voltages = sim.run(**run_kwargs)

        # Compute and plot axial rundown similarity parameter S
        try:
            S = shock_parameter(currents, cfg.anode_radius, cfg.initial_pressure)
            shock_path = plot_shock_parameter(
                times, S, Path(output) / "shock_trend.png"
            )
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

        if ctx.obj.get("lab_mode") and shots > 1:
            # In lab mode with multiple shots requested, run an ensemble with
            # simple jitter applied to key inputs for each realization.
            for idx in range(shots):
                shot_cfg = dataclasses.replace(
                    cfg,
                    charging_voltage=random.gauss(
                        cfg.charging_voltage, cfg.charging_voltage * 0.02
                    ),
                    initial_pressure=random.gauss(
                        cfg.initial_pressure, cfg.initial_pressure * 0.02
                    ),
                )
                shot_sim = DPFSimulation(shot_cfg)
                shot_dir = Path(output) / f"shot_{idx:03d}"
                shot_seeds = {"python": random.getstate()[1][0]}
                try:
                    shot_seeds["numpy"] = int(np.random.get_state()[1][0])
                except Exception:
                    try:
                        rng = np.random.default_rng()
                        shot_seeds["numpy"] = int(
                            rng.bit_generator.state["state"]["state"]
                        )
                    except Exception:
                        shot_seeds["numpy"] = 0
                shot_sim.run(output_dir=str(shot_dir), seeds=shot_seeds)
                ppc = getattr(
                    getattr(shot_cfg, "warpx_settings", None),
                    "max_particles_per_cell",
                    None,
                )
                cfg_paths = [p for p in [config, synthetic] if p]
                write_manifest(
                    shot_dir,
                    config_paths=cfg_paths,
                    config=asdict(shot_cfg),
                    ppc=ppc,
                    seeds=shot_seeds,
                    warnings=warnings_list,
                )
            return

        if ctx.obj.get("lab_mode"):

            ppc = getattr(
                getattr(cfg, "warpx_settings", None), "max_particles_per_cell", None
            )

            cfg_paths = [p for p in [config, synthetic] if p]
            write_manifest(
                output,
                config_paths=cfg_paths,
                config=asdict(cfg),
                ppc=ppc,
                seeds=seeds,
                warnings=warnings_list,
            )

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
@click.pass_context
def validate(ctx: click.Context, config: str, dataset: str, outdir: str) -> None:
    """Run a validation simulation and compare with experimental data."""
    from .validate import run_validation

    try:
        from .validate import run_validation

        ok = run_validation(
            Path(config),
            dataset,
            outdir=Path(outdir),
            lab_mode=ctx.obj.get("lab_mode", False),
        )
        if not ok:
            raise click.ClickException(format_error("VALIDATION", "Validation failed"))
    except Exception as e:  # pragma: no cover - defensive
        raise click.ClickException(format_error("VALIDATION", str(e)))


@main.command("validate-config")
@click.option("--config", type=click.Path(exists=True, dir_okay=False), required=True)
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
        raise click.ClickException(
            format_error("PLOT", f"No HDF5 files found in {input}")
        )

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
@click.option("--run-dir", type=click.Path(file_okay=False, exists=True), required=True)
@click.option("--output", type=click.Path(), default="plot.png")
def plot_run(run_dir: str, output: str) -> None:
    """Quickly plot current and voltage from an existing run directory."""
    import h5py
    import matplotlib.pyplot as plt

    files = sorted(Path(run_dir).glob("data_*.h5"))
    if not files:
        raise click.ClickException(
            format_error("PLOT", f"No HDF5 files found in {run_dir}")
        )

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
@click.option(
    "--values", type=float, multiple=True, required=True, help="Values to sweep"
)
@click.option("--output", type=click.Path(file_okay=False), default="sweep_output")
@click.option("--kpi", is_flag=True, help="Generate KPI plots without GUI")
@click.pass_context
def param_sweep_cmd(
    ctx: click.Context,
    config: str,
    parameter: str,
    values: tuple[float, ...],
    output: str,
    kpi: bool,
) -> None:
    """Run a parameter sweep and optionally generate KPI plots."""

    try:
        cfg = DPFConfig.from_file(config)
        if kpi:
            pm = ProjectManager(project=Path(output).name)
            label = f"{parameter}_sweep"
            pm.run_sweep(
                label,
                cfg,
                parameter,
                values,
                output_dir=output,
            )
            pm.overlay_metrics()
        else:
            results = run_parametric_sweep(
                cfg,
                parameter,
                values,
                output_dir=output,
                lab_mode=ctx.obj.get("lab_mode", False),
                config_path=config,
            )
            metrics = compute_sweep_metrics(cfg, results, parameter)
            plot_metric_overlay(parameter, metrics, Path(output) / "sweep_metrics.png")
            plot_yield_vs_S(metrics, Path(output) / "yield_vs_S.png")
    except Exception as e:
        raise click.ClickException(format_error("SWEEP", str(e)))

    click.echo(f"Sweep complete. Results written to {output}")


@main.command("uq-sweep")
@click.option("--config", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option(
    "--parameters",
    type=str,
    required=True,
    help="JSON mapping of parameter bounds, e.g. '{\"capacitance\":[1e-6,5e-6]}'",
)
@click.option(
    "--method", type=click.Choice(["lhs", "sobol"]), default="lhs", show_default=True
)
@click.option("--samples", type=int, default=4, show_default=True)
@click.option("--output", type=click.Path(dir_okay=False), default="uq_results.json")
def uq_sweep_cmd(
    config: str, parameters: str, method: str, samples: int, output: str
) -> None:
    """Run a multi-parameter sweep using UQ sampling schemes."""

    try:
        cfg = DPFConfig.from_file(config)
        bounds = json.loads(parameters)
        sampler = latin_hypercube if method == "lhs" else sobol_sample
        sample = sampler(bounds, samples)
        results: list[dict[str, Any]] = []
        names = list(bounds)
        peak_currents: list[float] = []
        for row in sample:
            params = {n: float(v) for n, v in zip(names, row)}
            cfg_i = dataclasses.replace(cfg, **params)
            sim = DPFSimulation(cfg_i)
            _, currents, _ = sim.run()
            peak = max(currents)
            peak_currents.append(peak)
            results.append({"params": params, "peak_current": peak})
        sobol = sobol_indices(sample, peak_currents, names)
        band = uncertainty_band(peak_currents)
        Path(output).write_text(
            json.dumps(
                {"results": results, "sobol_indices": sobol, "uncertainty_band": band},
                indent=2,
            )
        )
    except Exception as e:
        raise click.ClickException(format_error("UQ", str(e)))

    click.echo(f"UQ sweep complete. Results written to {output}")


@main.command("latin-hypercube")
@click.option(
    "--parameters",
    type=str,
    required=True,
    help="JSON mapping of parameter bounds, e.g. '{\"capacitance\":[1e-6,5e-6]}'",
)
@click.option("--samples", type=int, default=4, show_default=True)
@click.option("--seed", type=int, default=None)
@click.option("--output", type=click.Path(dir_okay=False), default="lhs_samples.json")
def latin_hypercube_cmd(
    parameters: str, samples: int, seed: int | None, output: str
) -> None:
    """Generate Latin hypercube samples for batch sweeps."""

    try:
        bounds = json.loads(parameters)
        sample = latin_hypercube(bounds, samples, seed=seed)
        names = list(bounds)
        combos = [{n: float(v) for n, v in zip(names, row)} for row in sample]
        Path(output).write_text(json.dumps(combos, indent=2))
    except Exception as e:  # pragma: no cover - runtime formatting
        raise click.ClickException(format_error("UQ", str(e)))

    click.echo(f"Latin hypercube samples written to {output}")


@main.command("sobol-sample")
@click.option(
    "--parameters",
    type=str,
    required=True,
    help="JSON mapping of parameter bounds, e.g. '{\"capacitance\":[1e-6,5e-6]}'",
)
@click.option("--samples", type=int, default=4, show_default=True)
@click.option("--seed", type=int, default=None)
@click.option("--output", type=click.Path(dir_okay=False), default="sobol_samples.json")
def sobol_sample_cmd(
    parameters: str, samples: int, seed: int | None, output: str
) -> None:
    """Generate Sobol sequence samples for batch sweeps."""

    try:
        bounds = json.loads(parameters)
        sample = sobol_sample(bounds, samples, seed=seed)
        names = list(bounds)
        combos = [{n: float(v) for n, v in zip(names, row)} for row in sample]
        Path(output).write_text(json.dumps(combos, indent=2))
    except Exception as e:  # pragma: no cover - runtime formatting
        raise click.ClickException(format_error("UQ", str(e)))

    click.echo(f"Sobol samples written to {output}")


@main.command("uq-stats")
@click.option("--input", type=click.Path(exists=True, dir_okay=False), required=True)
def uq_stats_cmd(input: str) -> None:
    """Compute statistics from a UQ sweep results file."""

    try:
        data = json.loads(Path(input).read_text())
        rows = data if isinstance(data, list) else data.get("results", [])
        currents = [r["peak_current"] for r in rows]
        stats = {
            "mean_peak_current": statistics.mean(currents),
            "std_peak_current": statistics.pstdev(currents),
        }
        click.echo(json.dumps(stats))
    except Exception as e:
        raise click.ClickException(format_error("UQ", str(e)))


@main.command("scaling")
@click.option("--config", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--parameter", type=str, required=True)
@click.option("--values", type=float, multiple=True, required=True)
@click.option(
    "--output",
    type=click.Path(dir_okay=False),
    default="scaling_report.json",
    help="File to write scaling analysis JSON",
)
def scaling_cmd(
    config: str, parameter: str, values: tuple[float, ...], output: str
) -> None:
    """Run a sweep and report fitted scaling exponents."""

    try:
        cfg = DPFConfig.from_file(config)
        res = sweep_yield_scaling(cfg, parameter, values)
        Path(output).write_text(json.dumps(res, indent=2))
        click.echo(
            f"m_current={res['m_current']:.3f} m_parameter={res['m_parameter']:.3f}"
        )
    except Exception as e:
        raise click.ClickException(format_error("SCALING", str(e)))


@main.command("make-surrogate")
@click.option("--data", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option(
    "--outdir",
    type=click.Path(file_okay=False),
    default="ai/surrogates",
    show_default=True,
    help="Directory to write surrogate model",
)
def make_surrogate(data: str, outdir: str) -> None:
    """Train a yield-vs-pressure surrogate and export an ONNX model."""

    try:
        arr = np.loadtxt(data, delimiter=",", dtype=np.float32)
        x = arr[:, 0]
        y = arr[:, 1]
        a, b = np.polyfit(x, y, 1)
        domain = [float(x.min()), float(x.max())]
        preds = a * x + b
        err = float(np.sqrt(np.mean((preds - y) ** 2)))

        out_dir = Path(outdir)
        out_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = out_dir / "yield_model.onnx"
        try:  # pragma: no cover - optional dependency
            import onnx  # type: ignore
            from onnx import helper, TensorProto  # type: ignore

            input_t = helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, [None, 1]
            )
            output_t = helper.make_tensor_value_info(
                "output", TensorProto.FLOAT, [None, 1]
            )
            a_t = helper.make_tensor("A", TensorProto.FLOAT, [1, 1], [a])
            b_t = helper.make_tensor("B", TensorProto.FLOAT, [1], [b])
            node1 = helper.make_node("MatMul", ["input", "A"], ["tmp"])
            node2 = helper.make_node("Add", ["tmp", "B"], ["output"])
            graph = helper.make_graph(
                [node1, node2], "linreg", [input_t], [output_t], [a_t, b_t]
            )
            model = helper.make_model(graph, producer_name="dpf2")
            onnx.save(model, onnx_path)
        except Exception:  # pragma: no cover - fallback path
            onnx_path.write_text("placeholder")

        meta = {
            "coeffs": [float(a), float(b)],
            "training_domain": domain,
            "error": err,
            "onnx": "yield_model.onnx",
        }
        with (out_dir / "yield_model.json").open("w") as fh:
            json.dump(meta, fh, indent=2)
        click.echo(f"Surrogate written to {out_dir}")
    except Exception as e:
        raise click.ClickException(format_error("SURROGATE", str(e)))


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
@click.option(
    "--rogowski-cal",
    type=click.Path(exists=True, dir_okay=False),
    help="Optional Rogowski calibration file",
)
@click.option(
    "--bdot-cal",
    type=click.Path(exists=True, dir_okay=False),
    help="Optional B-dot calibration file",
)
@click.option(
    "--sxr-cal",
    type=click.Path(exists=True, dir_okay=False),
    help="Optional SXR calibration file",
)
@click.option(
    "--tof-cal",
    type=click.Path(exists=True, dir_okay=False),
    help="Optional neutron TOF calibration file",
)
@click.option(
    "--anisotropy-plot",
    is_flag=True,
    help="Output simple angular neutron spectrum",
)
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
    rogowski_cal: str | None,
    bdot_cal: str | None,
    sxr_cal: str | None,
    tof_cal: str | None,
    anisotropy_plot: bool,
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
                outputs["rogowski"] = rogowski_signal(
                    states, dt, calibration_file=rogowski_cal
                )
            if cfg.synthetic_bdot_signal_enabled:
                outputs["bdot"] = bdot_signal(
                    states, radius, dt, calibration_file=bdot_cal
                )

        if current:
            outputs["current"] = current_waveform(states)
        if voltage:
            outputs["voltage"] = voltage_waveform(states)
        if rogowski:
            outputs["rogowski"] = rogowski_signal(
                states, dt, calibration_file=rogowski_cal
            )
        if bdot:
            outputs["bdot"] = bdot_signal(
                states, radius, dt, calibration_file=bdot_cal
            )

        if anisotropy_plot:
            angles = [0.0, 90.0, 180.0]
            outputs["anisotropy"] = angular_neutron_spectrum(angles, 1.0, 0.0)

        click.echo(json.dumps(outputs))
    except Exception as e:
        raise click.ClickException(format_error("DIAGNOSTICS", str(e)))


@main.command()
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Configuration file to export",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default="shared_config.json",
    help="Destination path for exported configuration",
)
def share(config: str, output: str) -> None:
    """Export a configuration for sharing with classmates."""
    try:
        cfg = DPFConfig.from_file(config)
        with open(output, "w") as fh:
            json.dump(asdict(cfg), fh, indent=2)
        click.echo(f"Configuration written to {output}")
    except Exception as e:
        raise click.ClickException(format_error("SHARE", str(e)))


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


@main.command("export-neutron-summary")
@click.option(
    "--angles",
    type=str,
    default="0.0",
    help="Comma separated detector angles in degrees",
)
@click.option("--distance", type=float, default=1.0, help="Detector distance [m]")
@click.option(
    "--outfile",
    type=click.Path(),
    default="neutron_summary.h5",
    help="Destination HDF5 file",
)
def export_neutron_summary(angles: str, distance: float, outfile: str) -> None:
    """Export a simple neutron TOF summary for chosen geometry."""

    ang_list = [float(a) for a in angles.split(",") if a]

    class _FlatEDF:
        def energy_distribution(self, angle_deg: float):  # pragma: no cover - simple stub
            return [0.0, 1.0], [1.0, 1.0]

    cross_section = lambda e: 1.0
    time_bins = [0.0, 1e-7, 2e-7]
    dets = simulate_tof_detectors(_FlatEDF(), cross_section, ang_list, distance, time_bins)
    save_tof_hdf5(outfile, time_bins, dets)
    click.echo(f"HDF5 summary written to {outfile}")


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


@main.command()
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False),
    default="docs/code_index.md",
    show_default=True,
    help="Destination Markdown file for the code index.",
)
@click.option(
    "--package",
    type=str,
    default="dpf2",
    show_default=True,
    help="Python package to index.",
)
@click.option(
    "--source-root",
    type=click.Path(file_okay=False),
    default=None,
    help="Override path to the package root.",
)
def index(output: str, package: str, source_root: str | None) -> None:
    """Generate a Markdown index of the code base."""

    try:
        if source_root is None:
            root = Path(__file__).resolve().parents[2] / package.replace(".", "/")
        else:
            root = Path(source_root)

        if not root.exists():
            raise click.ClickException(
                format_error("INDEX", f"Package root {root} does not exist")
            )

        entries = build_code_index(package, root)
        write_markdown_index(entries, Path(output))
        click.echo(
            f"Indexed {len(entries)} modules from {root} into {output}"
        )
    except click.ClickException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise click.ClickException(format_error("INDEX", str(exc)))


from .benchmark import benchmark, match_benchmark
main.add_command(match_benchmark)
main.add_command(benchmark)
if __name__ == "__main__":
    main()
