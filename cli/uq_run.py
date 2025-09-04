"""Command line interface for generating UQ sample batches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dpf2.uq.samplers import latin_hypercube, sobol_sample


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Latin hypercube or Sobol samples for batch studies",
    )
    parser.add_argument(
        "--parameters",
        required=True,
        help="JSON mapping of parameter bounds, e.g. '{\"x\":[0,1]}'",
    )
    parser.add_argument("--samples", type=int, default=4, help="Number of samples")
    parser.add_argument(
        "--method",
        choices=["lhs", "sobol"],
        default="lhs",
        help="Sampling strategy",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("uq_samples.json"),
        help="Output JSON file",
    )
    args = parser.parse_args()

    bounds = json.loads(args.parameters)
    sampler = latin_hypercube if args.method == "lhs" else sobol_sample
    arr = sampler(bounds, args.samples, seed=args.seed)
    names = list(bounds)
    combos = [{n: float(v) for n, v in zip(names, row)} for row in arr]
    args.output.write_text(json.dumps(combos, indent=2))


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
