from __future__ import annotations

"""Utility CLI to compare alternative datasets and show impact bands."""

import argparse
from pathlib import Path

from ..io.manifest import capture_dataset_metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two datasets")
    parser.add_argument("dataset", type=Path, help="Primary dataset")
    parser.add_argument(
        "--swap",
        type=Path,
        required=True,
        help="Alternate dataset to compare against",
    )
    args = parser.parse_args(argv)

    info = capture_dataset_metadata(
        {
            "atomic": {
                "base": {"path": args.dataset, "doi": "n/a", "version": "n/a"},
                "swap": {"path": args.swap, "doi": "n/a", "version": "n/a"},
            }
        }
    )
    base_hash = info["atomic"]["base"]["hash"]
    swap_hash = info["atomic"]["swap"]["hash"]
    print(f"Base hash: {base_hash}")
    print(f"Swap hash: {swap_hash}")
    if base_hash != swap_hash:
        print("Datasets differ - impact band may be significant.")
    else:
        print("Datasets identical.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
