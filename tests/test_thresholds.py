import warnings

from dpf2.diagnostics.thresholds import (
    compute_debye_length,
    plasma_inductance_circuit,
    check_thresholds,
)


def test_debye_length_positive():
    ld = compute_debye_length(10.0, 1e18)
    assert ld > 0


def test_plasma_inductance_circuit():
    L = plasma_inductance_circuit(10.0, 2.0, 1.0, 3.0)
    assert abs(L - (10.0 - 2.0 * 1.0) / 3.0) < 1e-12


def test_check_thresholds_warns():
    with warnings.catch_warnings(record=True) as w:
        msgs = check_thresholds(
            dt=2.0,
            debye_length=0.1,
            cell_size=0.2,
            particles_per_cell=5,
            max_dt=1.0,
            min_debye_cells=1.0,
            min_particles_per_cell=10,
        )
        assert len(msgs) == 3
        assert len(w) == 3
