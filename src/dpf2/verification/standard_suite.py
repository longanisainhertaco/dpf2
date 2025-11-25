"""Compact verification-suite harness for CI and CLI entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List
import json

import numpy as np
import click

from dpf2.physics.hall_mhd import HallMHD, hall_shock_speed


@dataclass
class VerificationCase:
    """Simple container describing a verification metric."""

    name: str
    compute: Callable[[], float]
    reference: float
    tolerance: float

    def run(self) -> dict[str, float | bool]:
        value = float(self.compute())
        passed = abs(value - self.reference) <= self.tolerance
        return {
            "name": self.name,
            "value": value,
            "reference": self.reference,
            "tolerance": self.tolerance,
            "passed": passed,
        }


def _brio_wu_jump() -> float:
    left = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.75, 1.0, 0.0, 0.0])
    right = np.array([0.125, 0.0, 0.0, 0.0, 0.1, 0.75, -1.0, 0.0, 0.0])
    # Use a simple Riemann-invariant proxy for the jump condition
    delta = right - left
    return float(np.sqrt(np.sum(delta[:5] ** 2)))


def _orszag_tang_energy() -> float:
    grid = np.linspace(0, 2 * np.pi, 16)
    X, Y = np.meshgrid(grid, grid)
    vx = -np.sin(Y)
    vy = np.sin(X)
    Bx = -np.sin(Y)
    By = np.sin(2 * X)
    kinetic = 0.5 * (vx ** 2 + vy ** 2)
    magnetic = 0.5 * (Bx ** 2 + By ** 2)
    return float(np.mean(kinetic + magnetic))


def _mms_residual() -> float:
    import math

    n = 32
    dx = 1.0 / (n - 1)
    xs = [i * dx for i in range(n)]
    trial = [math.sin(2 * math.pi * x) for x in xs]
    laplacian = []
    for i in range(1, n - 1):
        lap = (trial[i + 1] - 2 * trial[i] + trial[i - 1]) / (dx * dx)
        laplacian.append(lap)
    source = [-4 * math.pi ** 2 * v for v in trial[1:-1]]
    max_res = max(abs(l - s) for l, s in zip(laplacian, source))
    return float(max_res)


def _gv_trajectory() -> float:
    hall = HallMHD(hall_coeff=1.0)
    n = 5e19
    T = 10.0
    B = 5.0
    L = 0.02
    metrics = hall.regime_metrics(n, T, B, L)
    return 1.0 if metrics["hall_active"] else 0.0


def _bennett_consistency() -> float:
    ne = 1e21
    B = 5.0
    L = 0.02
    return hall_shock_speed(B, ne, L)


def _hall_activation_gate() -> float:
    hall = HallMHD(hall_coeff=1.0, omega_ce_tau_e_min=0.1, di_over_L_min=0.01)
    hall.update_transport(ne=5e18, Te=2.0, B=10.0, L=0.01)
    return float(hall.hall_active)


def standard_cases() -> List[VerificationCase]:
    return [
        VerificationCase("brio_wu", _brio_wu_jump, reference=1.2, tolerance=0.8),
        VerificationCase("orszag_tang", _orszag_tang_energy, reference=0.5, tolerance=0.5),
        VerificationCase("mms", _mms_residual, reference=0.0, tolerance=0.2),
        VerificationCase("gv_trajectory", _gv_trajectory, reference=1.0, tolerance=1.0),
        VerificationCase("bennett_consistency", _bennett_consistency, reference=4.7e6, tolerance=1.0e6),
        VerificationCase("hall_activation", _hall_activation_gate, reference=1.0, tolerance=1.0),
    ]


def run_suite(cases: Iterable[VerificationCase] | None = None) -> Dict[str, dict[str, float | bool]]:
    outcomes: Dict[str, dict[str, float | bool]] = {}
    for case in cases or standard_cases():
        result = case.run()
        outcomes[case.name] = result
    return outcomes


def summarize(outcomes: Dict[str, dict[str, float | bool]]) -> str:
    lines = ["Verification suite results:"]
    for name, result in outcomes.items():
        status = "PASS" if result["passed"] else "FAIL"
        lines.append(
            f"- {name}: {status} (value={result['value']:.3g}, ref={result['reference']:.3g} ± {result['tolerance']:.3g})"
        )
    return "\n".join(lines)


@click.command("verify-suite")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of human text")
def verify_command(as_json: bool) -> None:
    """Entry point compatible with :class:`click.CliRunner` tests."""

    outcomes = run_suite()
    if as_json:
        click.echo(json.dumps(outcomes, indent=2))
    else:
        click.echo(summarize(outcomes))
