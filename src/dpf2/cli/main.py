"""Command line interface for DPF2."""
import click

from ..core.config import DPFConfig
from ..core.simulation import DPFSimulation


@click.group()
def main() -> None:
    """Entry point for the DPF2 command line interface."""


@main.command()
@click.option("--config", type=click.Path(exists=False), help="Path to config file")
@click.option("--output", type=click.Path(), default="output", help="Output directory")
def simulate(config: str | None, output: str) -> None:
    """Run a DPF simulation."""
    cfg = DPFConfig() if config is None else DPFConfig()
    sim = DPFSimulation(cfg)
    sim.run(output_dir=output)


if __name__ == "__main__":
    main()
