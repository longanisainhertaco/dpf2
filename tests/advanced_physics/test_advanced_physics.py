"""Tests for advanced physics modules."""

from __future__ import annotations

import numpy as np
import pytest

from dpf2.advanced_physics.hall_mhd import (
    HallMHDSolver3D,
    Grid3D,
    MHDState3D,
    CTUpdate,
    compute_curl,
    compute_divergence,
    project_div_free,
    whistler_frequency,
    phase_velocity,
    dispersion_relation,
)
from dpf2.advanced_physics.kinetic import (
    HybridPICSolver,
    Particle,
    ParticleSpecies,
    dd_fusion_cross_section,
    dt_fusion_cross_section,
    beam_target_yield,
    BeamTargetModel,
)
from dpf2.advanced_physics.radiation import (
    MultigroupRadiationSolver,
    EnergyGroup,
    RadiationState,
)
from dpf2.advanced_physics.atomic import (
    NLTEIonization,
    IonizationState,
    AtomicData,
)


class TestGrid3D:
    """Tests for Grid3D class."""

    def test_grid_creation(self):
        grid = Grid3D(nx=10, ny=10, nz=10, dx=0.1, dy=0.1, dz=0.1)
        assert grid.shape == (10, 10, 10)
        assert grid.cell_volume == pytest.approx(0.001)

    def test_cell_centers(self):
        grid = Grid3D(nx=4, ny=4, nz=4, dx=1.0, dy=1.0, dz=1.0)
        assert grid.x[0] == pytest.approx(0.5)
        assert grid.x[-1] == pytest.approx(3.5)


class TestMHDState3D:
    """Tests for MHDState3D class."""

    def test_velocity_computation(self):
        rho = np.ones((4, 4, 4)) * 2.0
        mom = np.zeros((4, 4, 4, 3))
        mom[..., 0] = 10.0  # vx = 5 m/s
        energy = np.ones((4, 4, 4)) * 1e5
        B = np.zeros((4, 4, 4, 3))

        state = MHDState3D(rho=rho, mom=mom, energy=energy, B=B)
        v = state.velocity()
        assert v[0, 0, 0, 0] == pytest.approx(5.0)

    def test_pressure_positive(self):
        rho = np.ones((4, 4, 4))
        mom = np.zeros((4, 4, 4, 3))
        energy = np.ones((4, 4, 4)) * 1e5
        B = np.zeros((4, 4, 4, 3))

        state = MHDState3D(rho=rho, mom=mom, energy=energy, B=B)
        p = state.pressure()
        assert np.all(p > 0)


class TestConstrainedTransport:
    """Tests for constrained transport functions."""

    def test_compute_curl_zero_field(self):
        F = np.zeros((8, 8, 8, 3))
        curl_F = compute_curl(F, 0.1, 0.1, 0.1)
        assert np.allclose(curl_F, 0.0)

    def test_divergence_of_curl_is_zero(self):
        F = np.random.randn(8, 8, 8, 3)
        curl_F = compute_curl(F, 0.1, 0.1, 0.1)
        div_curl_F = compute_divergence(curl_F, 0.1, 0.1, 0.1)
        # Should be approximately zero (numerical precision)
        assert np.max(np.abs(div_curl_F)) < 0.1

    def test_project_div_free_reduces_divergence(self):
        B = np.random.randn(8, 8, 8, 3)
        div_before = compute_divergence(B, 0.1, 0.1, 0.1)

        B_clean = project_div_free(B)
        div_after = compute_divergence(B_clean, 0.1, 0.1, 0.1)

        # Projection should reduce divergence
        assert np.max(np.abs(div_after)) <= np.max(np.abs(div_before)) + 1e-10


class TestHallMHDSolver3D:
    """Tests for HallMHDSolver3D class."""

    def test_solver_initialization(self):
        grid = Grid3D(nx=8, ny=8, nz=8, dx=0.1, dy=0.1, dz=0.1)
        solver = HallMHDSolver3D(grid=grid, eta=1e-4, eta_H=1e-5)
        assert solver.eta == 1e-4
        assert solver.eta_H == 1e-5

    def test_current_density_computation(self):
        grid = Grid3D(nx=8, ny=8, nz=8, dx=0.1, dy=0.1, dz=0.1)
        solver = HallMHDSolver3D(grid=grid)

        B = np.zeros((8, 8, 8, 3))
        B[..., 2] = 1.0  # Uniform Bz
        J = solver.compute_current_density(B)
        # Uniform field has no current
        assert np.allclose(J, 0.0, atol=1e-10)

    def test_hall_field_perpendicular_to_B(self):
        grid = Grid3D(nx=4, ny=4, nz=4)
        solver = HallMHDSolver3D(grid=grid, eta_H=1.0)

        J = np.zeros((4, 4, 4, 3))
        J[..., 0] = 1.0  # Jx
        B = np.zeros((4, 4, 4, 3))
        B[..., 2] = 1.0  # Bz

        E_H = solver.compute_hall_field(J, B)
        # E_H should be in y-direction (J x B)
        assert np.abs(E_H[0, 0, 0, 1]) > 0

    def test_step_conserves_divergence_free(self):
        grid = Grid3D(nx=8, ny=8, nz=8, dx=0.1, dy=0.1, dz=0.1)
        solver = HallMHDSolver3D(grid=grid, eta=1e-4)

        # Create initial state
        mu_0 = 4e-7 * np.pi  # Permeability of free space
        rho = np.ones(grid.shape)
        mom = np.zeros(grid.shape + (3,))
        B = np.zeros(grid.shape + (3,))
        B[..., 2] = 1.0
        # Proper energy with thermal + magnetic
        thermal_e = 1e5 / (5.0/3.0 - 1)
        mag_e = 1.0**2 / (2 * mu_0)
        energy = np.ones(grid.shape) * (thermal_e + mag_e)

        state = MHDState3D(rho=rho, mom=mom, energy=energy, B=B)
        div_before = solver.divergence_error(state)

        dt = solver.cfl_timestep(state)
        new_state = solver.step(state, dt)

        div_after = solver.divergence_error(new_state)
        assert div_after < 1e-10


class TestWhistlerDispersion:
    """Tests for whistler dispersion relations."""

    def test_whistler_frequency_positive(self):
        omega = whistler_frequency(1e6, 1e20, 1.0)
        assert omega > 0

    def test_whistler_frequency_scales_with_k_squared(self):
        n_e = 1e20
        B = 1.0
        omega1 = whistler_frequency(1e6, n_e, B)
        omega2 = whistler_frequency(2e6, n_e, B)
        # omega ~ k^2
        assert omega2 / omega1 == pytest.approx(4.0, rel=0.1)

    def test_phase_velocity_positive(self):
        v_ph = phase_velocity(1e6, 1e20, 1.0)
        assert v_ph > 0


class TestHybridPICSolver:
    """Tests for HybridPICSolver class."""

    def test_solver_initialization(self):
        solver = HybridPICSolver(nx=8, ny=8, nz=8, dx=0.01, dy=0.01, dz=0.01)
        assert solver.shape == (8, 8, 8)
        assert len(solver.particles) == 0

    def test_add_particles(self):
        solver = HybridPICSolver(nx=8, ny=8, nz=8)
        positions = np.random.rand(10, 3) * 0.5
        velocities = np.zeros((10, 3))

        solver.add_particles(positions, velocities)
        assert len(solver.particles) == 10

    def test_deposit_density(self):
        solver = HybridPICSolver(nx=8, ny=8, nz=8, dx=1.0, dy=1.0, dz=1.0)
        positions = np.array([[0.5, 0.5, 0.5]])
        velocities = np.zeros((1, 3))
        solver.add_particles(positions, velocities)

        solver.deposit_density()
        assert np.sum(solver.n_i) > 0


class TestParticleSpecies:
    """Tests for ParticleSpecies class."""

    def test_deuterium_creation(self):
        D = ParticleSpecies.deuterium()
        assert D.name == "D"
        assert D.charge > 0

    def test_tritium_creation(self):
        T = ParticleSpecies.tritium()
        assert T.name == "T"
        assert T.mass > ParticleSpecies.deuterium().mass


class TestFusionCrossSections:
    """Tests for fusion cross section functions."""

    def test_dd_cross_section_positive(self):
        sigma = dd_fusion_cross_section(100.0)
        assert sigma > 0

    def test_dt_cross_section_higher_than_dd(self):
        # At same energy, D-T should have higher cross section
        sigma_dd = dd_fusion_cross_section(50.0)
        sigma_dt = dt_fusion_cross_section(50.0)
        assert sigma_dt > sigma_dd

    def test_cross_section_array_input(self):
        E = np.array([10.0, 50.0, 100.0])
        sigma = dd_fusion_cross_section(E)
        assert sigma.shape == (3,)
        assert np.all(sigma > 0)


class TestBeamTargetYield:
    """Tests for beam target yield calculation."""

    def test_yield_positive(self):
        yield_val = beam_target_yield(
            n_beam=1e18,
            n_target=1e24,
            E_beam_keV=100.0,
            volume=1e-6,
            path_length=0.01,
            reaction="DD"
        )
        assert yield_val > 0

    def test_yield_scales_with_density(self):
        yield1 = beam_target_yield(1e18, 1e24, 100.0, 1e-6, 0.01)
        yield2 = beam_target_yield(2e18, 1e24, 100.0, 1e-6, 0.01)
        assert yield2 / yield1 == pytest.approx(2.0, rel=0.1)


class TestMultigroupRadiationSolver:
    """Tests for MultigroupRadiationSolver class."""

    def test_solver_initialization(self):
        solver = MultigroupRadiationSolver(nx=8, ny=8, nz=8)
        assert solver.n_groups == 10  # Default
        assert solver.state is not None

    def test_opacity_positive(self):
        solver = MultigroupRadiationSolver(nx=4, ny=4, nz=4)
        rho = np.ones((4, 4, 4))
        T = np.ones((4, 4, 4)) * 1e6

        kappa = solver.compute_opacity(rho, T, 0)
        assert np.all(kappa > 0)

    def test_emission_positive(self):
        solver = MultigroupRadiationSolver(nx=4, ny=4, nz=4)
        rho = np.ones((4, 4, 4))
        T = np.ones((4, 4, 4)) * 1e6

        emission = solver.compute_emission(rho, T, 0)
        assert np.all(emission >= 0)


class TestEnergyGroup:
    """Tests for EnergyGroup class."""

    def test_center_energy(self):
        group = EnergyGroup(E_low=1.0, E_high=10.0)
        assert group.E_center == pytest.approx(np.sqrt(10.0))

    def test_contains(self):
        group = EnergyGroup(E_low=1.0, E_high=10.0)
        assert group.contains(5.0)
        assert not group.contains(0.5)
        assert not group.contains(15.0)


class TestNLTEIonization:
    """Tests for NLTEIonization class."""

    def test_model_initialization(self):
        data = AtomicData.hydrogen()
        model = NLTEIonization(atomic_data=data)
        assert model.atomic_data.Z == 1

    def test_ionization_rates_positive(self):
        data = AtomicData.hydrogen()
        model = NLTEIonization(atomic_data=data)

        rates = model.compute_ionization_rates(1e20, 1e5)
        assert np.all(rates >= 0)

    def test_recombination_rates_positive(self):
        data = AtomicData.hydrogen()
        model = NLTEIonization(atomic_data=data)

        rates = model.compute_recombination_rates(1e20, 1e5)
        assert np.all(rates >= 0)

    def test_steady_state_sums_to_one(self):
        data = AtomicData.hydrogen()
        model = NLTEIonization(atomic_data=data)

        state = model.solve_rate_equations(1e20, 1e5, method="steady")
        assert np.sum(state.populations) == pytest.approx(1.0)


class TestAtomicData:
    """Tests for AtomicData class."""

    def test_hydrogen_data(self):
        H = AtomicData.hydrogen()
        assert H.Z == 1
        assert H.symbol == "H"
        assert len(H.ionization_energies) == 1

    def test_argon_data(self):
        Ar = AtomicData.argon()
        assert Ar.Z == 18
        assert len(Ar.ionization_energies) == 18


class TestIonizationState:
    """Tests for IonizationState class."""

    def test_neutral_state(self):
        state = IonizationState.neutral(Z=2)
        assert state.populations[0] == 1.0
        assert state.Z_mean == 0.0

    def test_fully_ionized_state(self):
        state = IonizationState.fully_ionized(Z=2)
        assert state.populations[-1] == 1.0
        assert state.Z_mean == 2.0
