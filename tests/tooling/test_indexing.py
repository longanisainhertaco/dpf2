from __future__ import annotations

from pathlib import Path

from dpf2.indexing import build_code_index, render_markdown


def test_build_code_index_includes_core_config(tmp_path: Path) -> None:
    package_root = Path(__file__).resolve().parents[2] / "src" / "dpf2"
    entries = build_code_index("dpf2", package_root)

    modules = {entry.module for entry in entries}
    assert "dpf2.core.config" in modules

    markdown = render_markdown(entries[:5])
    assert markdown.startswith("# DPF2 Code Index")

    output = tmp_path / "index.md"
    output.write_text(markdown, encoding="utf-8")
    assert output.exists()
