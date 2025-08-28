"""Command line interface for DPF2."""
import json
import logging
import dataclasses
from pathlib import Path
from dataclasses import asdict

import click

from dpf2.core.config import DPFConfig
from dpf2.core.simulation import DPFSimulation
from dpf2.core.bases import CouplingState
from dpf2.diagnostics.synthetic_signals import (
    current_waveform,
    voltage_waveform,
    rogowski_signal,
    bdot_signal,
)
from dpf2.synthetic_diagnostics import SyntheticDiagnostics
from dpf2.exceptions import ConfigurationError, SimulationRuntimeError

logger = logging.getLogger(__name__)


@click.group()
def main() -> None:
    """Entry point for the DPF2 command line interface."""


@main.command()
@click.option("-c", "--config", type=click.Path(exists=False), help="Path to config file")
@click.option("-o", "--output", type=click.Path(), default="output", help="Output directory")
@click.option("--verbose", is_flag=True, help="Report solver progress and energy diagnostics")
def simulate(config: str | None, output: str, verbose: bool) -> None:
    """Run a DPF simulation."""
    try:
        if verbose:
            logging.basicConfig(level=logging.INFO)
        cfg = DPFConfig.from_file(config) if config else DPFConfig()
        sim = DPFSimulation(cfg)
        times, currents, voltages = sim.run(output_dir=output, verbose=verbose)

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
        raise click.ClickException(f"Configuration error: {e}")
    except SimulationRuntimeError as e:
        logger.error("Simulation error: %s", e)
        raise click.ClickException(f"Simulation error: {e}")
    except Exception as e:
        logger.exception("Unexpected error running simulation")
        raise click.ClickException(f"Unexpected error: {e}")


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
            raise click.ClickException("Validation failed")
    except Exception as e:  # pragma: no cover - defensive
        raise click.ClickException(str(e))


@main.command()
@click.option("--input", type=click.Path(file_okay=False), required=True)
@click.option("--output", type=click.Path(), default="plot.png")
def plot(input: str, output: str) -> None:
    """Plot current and voltage from simulation outputs."""
    import h5py
    import matplotlib.pyplot as plt

    files = sorted(Path(input).glob("data_*.h5"))
    if not files:
        raise click.ClickException(f"No HDF5 files found in {input}")

    times, currents, voltages = [], [], []
    for fname in files:
        with h5py.File(fname, "r") as fh:
            times.append(float(fh["time"][()]))
            if "current" in fh:
                currents.append(float(fh["current"][()]))
            if "voltage" in fh:
                voltages.append(float(fh["voltage"][()]))

    if not currents:
        raise click.ClickException("No current data available")

    plt.figure()
    plt.plot(times, currents, label="current")
    if voltages:
        plt.plot(times, voltages, label="voltage")
    plt.xlabel("time [s]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    click.echo(f"Plot written to {output}")


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

    click.echo("DPF2 configuration wizard\n")
    defaults = DPFConfig()

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
    gas_type = click.prompt("Fill gas", type=str, default=defaults.gas_type, show_default=True)
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
    cfg = DPFConfig(**cfg_dict)

    with open(output, "w") as fh:
        json.dump(asdict(cfg), fh, indent=2)

    click.echo(f"Configuration saved to {output}")


if __name__ == "__main__":
    main()
