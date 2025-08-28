"""Command line interface for DPF2."""
import json
import logging
import dataclasses
from pathlib import Path
from dataclasses import asdict

import click

from dpf2.core.config import DPFConfig
from dpf2.core.simulation import DPFSimulation
from dpf2.exceptions import ConfigurationError, SimulationRuntimeError
from .validate import run_validation

logger = logging.getLogger(__name__)


@click.group()
def main() -> None:
    """Entry point for the DPF2 command line interface."""


@main.command()
@click.option("-c", "--config", type=click.Path(exists=False), help="Path to config file")
@click.option("-o", "--output", type=click.Path(), default="output", help="Output directory")
def simulate(config: str | None, output: str) -> None:
    """Run a DPF simulation."""
    try:
        cfg = DPFConfig.from_file(config) if config else DPFConfig()
        sim = DPFSimulation(cfg)
        sim.run(output_dir=output)
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
    try:
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
    cathode_radius = click.prompt(
        "Cathode radius [m]", type=float, default=defaults.cathode_radius, show_default=True
    )
    anode_radius = click.prompt(
        "Anode radius [m]", type=float, default=defaults.anode_radius, show_default=True
    )
    electrode_length = click.prompt(
        "Electrode length [m]", type=float, default=defaults.electrode_length, show_default=True
    )

    # --- Fill gas ----------------------------------------------------
    click.echo("\nPlasma fill parameters:")
    gas_type = click.prompt("Fill gas", type=str, default=defaults.gas_type, show_default=True)
    initial_pressure = click.prompt(
        "Initial pressure [Pa]",
        type=float,
        default=defaults.initial_pressure,
        show_default=True,
    )

    # --- Capacitor bank ---------------------------------------------
    click.echo("\nExternal circuit:")
    click.echo("Capacitor bank and wiring values influence current rise.")
    capacitance = click.prompt(
        "Capacitance [F]",
        type=float,
        default=defaults.capacitance,
        show_default=True,
    )
    inductance = click.prompt(
        "Inductance [H]",
        type=float,
        default=defaults.inductance,
        show_default=True,
    )
    resistance = click.prompt(
        "Resistance [Ohm]",
        type=float,
        default=defaults.resistance,
        show_default=True,
    )
    charging_voltage = click.prompt(
        "Charging voltage [V]",
        type=float,
        default=defaults.charging_voltage,
        show_default=True,
    )

    # --- Advanced options -------------------------------------------
    advanced_cfg: dict[str, float | int] = {}
    if click.confirm("Configure advanced mesh and timing options?", default=False):
        click.echo("\nMesh and solver controls:")
        advanced_cfg["nr_cells"] = click.prompt(
            "Radial cells",
            type=int,
            default=defaults.nr_cells,
            show_default=True,
        )
        advanced_cfg["nz_cells"] = click.prompt(
            "Axial cells",
            type=int,
            default=defaults.nz_cells,
            show_default=True,
        )
        advanced_cfg["cfl_number"] = click.prompt(
            "CFL number",
            type=float,
            default=defaults.cfl_number,
            show_default=True,
        )
        advanced_cfg["end_time"] = click.prompt(
            "Simulation end time [s]",
            type=float,
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
