"""Demonstration of loading geometry files with per-surface material tags."""

from pathlib import Path
from dpf2.geometry.loaders import load_cad_geometry


HERE = Path(__file__).resolve().parent


def show(path: Path) -> None:
    data = load_cad_geometry(path)
    mats = data.get("materials", [])
    print(f"{path.name}: {mats}")


def main() -> None:
    show(HERE / "tagged.stl")
    show(HERE / "tagged.vtk")


if __name__ == "__main__":
    main()
