"""Entry point for dataset comparison helper.

This thin wrapper simply exposes :func:`dpf2.cli.validate_datasets.main` as a
stand-alone script so that users can invoke the dataset comparison utility via
``python cli/validate_datasets.py``.
"""

from dpf2.cli.validate_datasets import main


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
