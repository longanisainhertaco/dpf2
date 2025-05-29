"""Command line interface for the DPF simulator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dpf_config import DPFConfig

from .simulation_engine import SimulationEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dpf2", description="Dense Plasma Focus simulator")
    sub = parser.add_subparsers(dest="command")

    sim = sub.add_parser("simulate", help="Run a simulation")
    sim.add_argument("config", type=Path, help="Path to JSON/YAML configuration")
    sim.add_argument("-o", "--output", type=Path, default=Path("results.json"), help="Output file")
    sim.add_argument("--method", choices=["analytical", "ode"], default="analytical", help="Circuit solver method")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "simulate":
        cfg = DPFConfig.from_file(args.config)
        engine = SimulationEngine(cfg)
        results = engine.run(method=args.method)
        args.output.write_text(json.dumps(results.to_dict(), indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
