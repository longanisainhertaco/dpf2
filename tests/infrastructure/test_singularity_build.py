"""Ensure Singularity definition files build successfully."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def _builder() -> str | None:
    return shutil.which("singularity") or shutil.which("apptainer")


def _build(def_file: Path, tmpdir: Path, builder: str) -> None:
    out = tmpdir / def_file.stem
    cmd = [builder, "build", "--sandbox", str(out), str(def_file)]
    subprocess.run(cmd, check=True)
    assert out.exists()


@pytest.mark.skipif(_builder() is None, reason="Singularity/Apptainer not available")
def test_definition_files_build(tmp_path: Path) -> None:
    builder = _builder()
    defs = list(Path("infrastructure/singularity").glob("*.def"))
    assert defs, "No Singularity definition files found"
    for d in defs:
        _build(d, tmp_path, builder)
