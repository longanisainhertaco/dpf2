"""Command line interface for DPF2."""
import json
import logging
from dataclasses import asdict

import click

from dpf2.core.config import DPFConfig
from dpf2.core.simulation import DPFSimulation
from dpf2.exceptions import ConfigurationError, SimulationRuntimeError

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
    cathode_radius = click.prompt(
        "Cathode radius [m]", type=float, default=defaults.cathode_radius
    )
    anode_radius = click.prompt(
        "Anode radius [m]", type=float, default=defaults.anode_radius
    )
    electrode_length = click.prompt(
        "Electrode length [m]", type=float, default=defaults.electrode_length
    )

    # --- Fill gas ----------------------------------------------------
    gas_type = click.prompt("Fill gas", type=str, default=defaults.gas_type)
    initial_pressure = click.prompt(
        "Initial pressure [Pa]", type=float, default=defaults.initial_pressure
    )

    # --- Capacitor bank ---------------------------------------------
    capacitance = click.prompt(
        "Capacitance [F]", type=float, default=defaults.capacitance
    )
    inductance = click.prompt(
        "Inductance [H]", type=float, default=defaults.inductance
    )
    resistance = click.prompt(
        "Resistance [Ohm]", type=float, default=defaults.resistance
    )
    charging_voltage = click.prompt(
        "Charging voltage [V]", type=float, default=defaults.charging_voltage
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
    cfg = DPFConfig(**cfg_dict)

    with open(output, "w") as fh:
        json.dump(asdict(cfg), fh, indent=2)

    click.echo(f"Configuration saved to {output}")


if __name__ == "__main__":
    main()
