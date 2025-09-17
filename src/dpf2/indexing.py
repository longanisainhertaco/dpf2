"""Utilities to build a lightweight index of the DPF2 source tree.

The goal of this module is to provide a dependency-free way to
summarise the package structure.  It scans the Python sources using
``ast`` so we do not need to import optional heavy dependencies during
index generation.  The resulting data structure can be rendered to
Markdown for documentation or quick navigation aids.
"""

from __future__ import annotations

import ast
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence


@dataclass(slots=True)
class SymbolEntry:
    """Representation of a discovered class or function."""

    name: str
    summary: str | None


@dataclass(slots=True)
class ModuleEntry:
    """Structured summary of a single Python module."""

    module: str
    path: Path
    relative_path: Path
    summary: str | None
    classes: list[SymbolEntry]
    functions: list[SymbolEntry]


def _iter_python_files(package: str, package_root: Path) -> Iterator[tuple[str, Path]]:
    """Yield dotted module names and file paths under ``package_root``."""

    base = package_root.resolve()
    for path in sorted(base.rglob("*.py")):
        relative = path.relative_to(base)
        parts = list(relative.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        module_parts = [package] + parts
        dotted = ".".join(part for part in module_parts if part)
        yield dotted, path


def _summarise(doc: str | None, *, width: int = 100) -> str | None:
    """Return a compact single-line summary from ``doc`` if present."""

    if not doc:
        return None
    first_line = doc.strip().splitlines()[0].strip()
    if not first_line:
        return None
    return textwrap.shorten(first_line, width=width, placeholder="…")


def build_code_index(package: str, package_root: Path) -> list[ModuleEntry]:
    """Scan ``package_root`` and build an index for ``package``."""

    resolved_root = package_root.resolve()
    parents = list(resolved_root.parents)
    repo_root = parents[1] if len(parents) > 1 else parents[0]

    entries: list[ModuleEntry] = []
    for dotted, path in _iter_python_files(package, resolved_root):
        try:
            source = path.read_text(encoding="utf-8")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                module_ast = ast.parse(source)
        except (SyntaxError, OSError, UnicodeDecodeError):
            # Skip files that cannot be parsed; continue with remaining ones.
            continue

        module_summary = _summarise(ast.get_docstring(module_ast))
        classes: list[SymbolEntry] = []
        functions: list[SymbolEntry] = []

        for node in module_ast.body:
            if isinstance(node, ast.ClassDef):
                classes.append(
                    SymbolEntry(node.name, _summarise(ast.get_docstring(node)))
                )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(
                    SymbolEntry(node.name, _summarise(ast.get_docstring(node)))
                )

        try:
            display_path = path.relative_to(repo_root)
        except ValueError:
            display_path = path

        entries.append(
            ModuleEntry(
                module=(
                    f"{package}.{dotted.split('.', 1)[-1]}"
                    if dotted != package
                    else package
                ),
                path=path,
                relative_path=display_path,
                summary=module_summary,
                classes=classes,
                functions=functions,
            )
        )

    entries.sort(key=lambda entry: entry.module)
    return entries


def render_markdown(
    entries: Sequence[ModuleEntry], heading: str = "DPF2 Code Index"
) -> str:
    """Render ``entries`` as a Markdown document."""

    lines: list[str] = [f"# {heading}", "", "Generated with `dpf2.indexing`.", ""]
    lines.append(f"Indexed modules: {len(entries)}")
    lines.append("")

    for entry in entries:
        lines.append(f"## {entry.module}")
        lines.append(f"*Source:* `{entry.relative_path}`")
        lines.append("")
        if entry.summary:
            lines.append(entry.summary)
        else:
            lines.append("_No module docstring available._")
        lines.append("")

        if entry.classes:
            lines.append("### Classes")
            for symbol in entry.classes:
                summary = symbol.summary or "_No docstring._"
                lines.append(f"- ``{symbol.name}``: {summary}")
            lines.append("")

        if entry.functions:
            lines.append("### Functions")
            for symbol in entry.functions:
                summary = symbol.summary or "_No docstring._"
                lines.append(f"- ``{symbol.name}``: {summary}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_markdown_index(entries: Sequence[ModuleEntry], destination: Path) -> None:
    """Write ``entries`` to ``destination`` in Markdown format."""

    destination = destination.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_markdown(entries), encoding="utf-8")


def generate_markdown_index(
    package: str,
    package_root: Path,
    destination: Path,
    *,
    heading: str = "DPF2 Code Index",
) -> None:
    """Convenience wrapper that builds and writes an index in one step."""

    entries = build_code_index(package, package_root)
    write_markdown_index(entries, destination)
