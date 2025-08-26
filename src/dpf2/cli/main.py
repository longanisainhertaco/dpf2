"""Command line interface for DPF2."""
import logging

import click

from dpf2.core.config import DPFConfig
from dpf2.core.simulation import DPFSimulation
from dpf2.exceptions import ConfigurationError, SimulationRuntimeError

logger = logging.getLogger(__name__)


@click.group()
def main() -> None:
    """Entry point for the DPF2 command line interface."""


@main.command()
@click.option("--config", type=click.Path(exists=False), help="Path to config file")
@click.option("--output", type=click.Path(), default="output", help="Output directory")
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


if __name__ == "__main__":
    main()
