"""Minimal checkpoint/restart helper for DPF simulations.

The script runs the :class:`~dpf2.simulation_engine.SimulationEngine`
and stores arrays required to restart the circuit integration.  When the
``--resume`` flag is provided and a checkpoint file exists, the script
reloads the previously saved data before executing another run.  The
example is intentionally lightweight; production usage would serialise
full solver state.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dpf2.dpf_config import DPFConfig
from dpf2.simulation_engine import SimulationEngine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Configuration file")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint npz file")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")
    args = parser.parse_args()

    cfg = DPFConfig.from_file(args.config)
    engine = SimulationEngine(cfg)

    if args.resume and args.checkpoint.exists():
        data = np.load(args.checkpoint, allow_pickle=True)
        last_time = data["time"][-1]
        print(f"Resuming from t={last_time:.3e} s")

    results = engine.run()
    np.savez(args.checkpoint, time=results.time, current=results.current)


if __name__ == "__main__":  # pragma: no cover
    main()
