"""Basic performance regression tests for scaling utilities."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_benchmark_scaling_script(tmp_path: Path) -> None:
    outdir = tmp_path / "results"
    subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_scaling.py",
            "--max-workers",
            "1",
            "--problem-size",
            "10",
            "--outdir",
            str(outdir),
        ],
        check=True,
    )
    data = json.loads((outdir / "scaling.json").read_text())
    assert {"strong", "weak", "roofline"} <= data.keys()
