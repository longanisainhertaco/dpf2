"""Example workflows for neutron benchmark comparisons."""

from diagnostics.neutron.benchmarks import (
    load_pf1000_reference,
    load_mjolnir_reference,
    evaluate_pass_fail,
)


def run_pf1000_benchmark(pass_band: float = 0.1) -> bool:
    """Return ``True`` if sample data matches PF-1000 reference."""
    reference = load_pf1000_reference()
    # Offset current trace within pass band
    simulated = reference["current"] + 0.05
    return evaluate_pass_fail(simulated, reference["current"], pass_band)


def run_mjolnir_benchmark(pass_band: float = 0.1) -> bool:
    """Return ``True`` if sample data matches MJOLNIR reference."""
    reference = load_mjolnir_reference()
    # Offset current trace beyond pass band to demonstrate failure
    simulated = reference["current"] + 0.5
    return evaluate_pass_fail(simulated, reference["current"], pass_band)


if __name__ == "__main__":
    print("PF-1000 benchmark passed:", run_pf1000_benchmark())
    print("MJOLNIR benchmark passed:", run_mjolnir_benchmark())
