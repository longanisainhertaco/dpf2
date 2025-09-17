"""Command line interface for the DPF simulator."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from ..dpf_config import DPFConfig

from ..simulation_engine import SimulationEngine
from .lab import write_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dpf2", description="Dense Plasma Focus simulator"
    )
    sub = parser.add_subparsers(dest="command")

    sim = sub.add_parser("simulate", help="Run a simulation")
    sim.add_argument("config", type=Path, help="Path to JSON/YAML configuration")
    sim.add_argument(
        "-o", "--output", type=Path, default=Path("results.json"), help="Output file"
    )
    sim.add_argument(
        "--method",
        choices=["analytical", "ode"],
        default="analytical",
        help="Circuit solver method",
    )
    sim.add_argument(
        "--pinch-model",
        choices=["analytic", "semi-analytic", "mhd"],
        default="analytic",
        help="Pinch dynamics model",
    )
    sim.add_argument(
        "--lab-mode",
        action="store_true",
        help="Record a reproducibility manifest alongside outputs",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "simulate":
        cfg = DPFConfig.from_file(args.config)
        engine = SimulationEngine(cfg)
        if args.lab_mode:
            seeds = {"python": random.getstate()[1][0]}
            try:
                seeds["numpy"] = int(np.random.get_state()[1][0])
            except Exception:
                try:
                    rng = np.random.default_rng()
                    seeds["numpy"] = int(rng.bit_generator.state["state"]["state"])
                except Exception:
                    seeds["numpy"] = 0
        results = engine.run(method=args.method, pinch_model=args.pinch_model)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results.to_dict(), indent=2))
        if args.lab_mode:
            ppc = getattr(
                getattr(cfg, "warpx_settings", None), "max_particles_per_cell", None
            )
            cfg_dict = (
                cfg.model_dump(mode="python")
                if hasattr(cfg, "model_dump")
                else cfg.__dict__
            )
            write_manifest(
                args.output.parent,
                config_paths=[str(args.config)],
                config=cfg_dict,
                ppc=ppc,
                seeds=seeds,
            )
    else:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
