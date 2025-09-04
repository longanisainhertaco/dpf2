from __future__ import annotations

from pathlib import Path
import numpy as np

from .plots import plot_current_voltage, plot_vector_field_overlay


def main(output_dir: str = "sandbox_output") -> None:
    """Generate simple current/voltage and vector-field plots for students."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, 1.0, 100)
    current = np.sin(2 * np.pi * t)
    voltage = np.cos(2 * np.pi * t)
    plot_current_voltage(t, current, voltage, out / "current_voltage.png")
    x = np.linspace(-1.0, 1.0, 20)
    y = np.linspace(-1.0, 1.0, 20)
    X, Y = np.meshgrid(x, y)
    U = -Y
    V = X
    plot_vector_field_overlay(X, Y, U, V, out / "vector_field.png")
    print(f"Sandbox plots written to {out}")


if __name__ == "__main__":
    main()
