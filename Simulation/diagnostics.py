import numpy as np
import h5py
import json
import logging
import time
from scipy.constants import c, m_n, m_e, mu_0, e
from scipy.interpolate import interp1d
from pyevtk.hl import imageToVTK
from utils import FieldManager, SimulationState # Import FieldManager and SimulationState

logger = logging.getLogger(__name__)

# --- Diagnostic Base Class ---
class Diagnostic:
    def __init__(self, name, field_manager: FieldManager):
        self.name = name
        self.field_manager = field_manager

    def record(self, t, circuit, fluid, pic=None, radiation=None, state: SimulationState = None):
        raise NotImplementedError

    def to_hdf5(self, hdf5_group):
        raise NotImplementedError

# --- Interferometry ---
class Interferometry(Diagnostic):
    def __init__(self, name, p0, p1, field_manager: FieldManager):
        super().__init__(name, field_manager)
        self.p0 = np.array(p0)
        self.p1 = np.array(p1)
        self.data = []

    def record(self, t, circuit, fluid, pic=None, radiation=None, state: SimulationState = None):
        rho = state.density
        dx = state.dx
        domain_lo = state.domain_lo
        L = np.linalg.norm(self.p1 - self.p0)
        Np = int(np.ceil(L / dx))
        pts = np.linspace(self.p0, self.p1, Np)
        dens = []
        for pt in pts:
            xi, yi, zi = (pt[0] - domain_lo[0]) / dx, (pt[1] - domain_lo[1]) / dx, (pt[2] - domain_lo[2]) / dx
            i, j, k = int(np.floor(xi)), int(np.floor(yi)), int(np.floor(zi))
            dens.append(rho[i, j, k])
        line_integral = np.trapz(dens, dx=dx)
        # Calculate phase shift (simplified)
        phase_shift = line_integral * 2.25e-18  # Example constant
        self.data.append({'time': t, 'phase_shift': phase_shift})

    def to_hdf5(self, hdf5_group):
        grp = hdf5_group.create_group(self.name)
        grp.create_dataset('time', data=[d['time'] for d in self.data])
        grp.create_dataset('phase_shift', data=[d['phase_shift'] for d in self.data])

# --- X-ray Detector ---
class XrayDetector(Diagnostic):
    def __init__(self, name, position, field_manager: FieldManager, energy_bins=None, detector_response=None):
        super().__init__(name, field_manager)
        self.position = np.array(position)
        self.energy_bins = energy_bins or [0, np.inf]
        self.detector_response = detector_response or (lambda E: 1.0)  # Default: constant efficiency
        self.data = []

    def record(self, t, circuit, fluid, pic=None, radiation=None, state: SimulationState = None):
        if radiation:
            dx = state.dx
            domain_lo = state.domain_lo
            if hasattr(radiation, 'get_energy_resolved_emission'):
                energy_bins, P_rad_energy = radiation.get_energy_resolved_emission(state)
            else:
                P_rad_total = radiation.total_radiated_energy
                energy_bins = [0, np.inf]
                P_rad_energy = [P_rad_total]

            signal = 0.0
            for i, E_bin in enumerate(energy_bins[:-1]):
                detector_efficiency = self.detector_response(E_bin)
                # Integrate over the volume
                # distance from cell centers
                dxs = state._X - self.position[0]; dys = state._Y - self.position[1]; dzs = state._Z - self.position[2]
                dist2 = dxs * dxs + dys * dys + dzs * dzs
                signal += np.sum(P_rad_energy[i] * state.cell_volume / dist2) * detector_efficiency
            self.data.append({'time': t, 'signal': signal})

    def to_hdf5(self, hdf5_group):
        grp = hdf5_group.create_group(self.name)
        grp.create_dataset('time', data=[d['time'] for d in self.data])
        grp.create_dataset('signal', data=[d['signal'] for d in self.data])

# --- Neutron Detector ---
class NeutronDetector(Diagnostic):
    def __init__(self, name, position, time_bins, field_manager: FieldManager, reaction='DD'):
        super().__init__(name, field_manager)
        self.position = np.array(position)
        self.time_bins = time_bins
        self.reaction = reaction
        self.data = []

    def record(self, t, circuit, fluid, pic=None, radiation=None, state: SimulationState = None):
        if pic and hasattr(pic, 'get_neutron_events'):
            events = pic.get_neutron_events(reaction=self.reaction)
            tof = []
            for ev in events:
                ev_pos = np.array(ev['position'])
                E_n = ev['energy'] * 1.602e-13  # keV->J
                v = np.sqrt(2 * E_n / m_n)
                dist = np.linalg.norm(ev_pos - self.position)
                tof.append(ev['time'] + dist / v)
            hist, _ = np.histogram(tof, bins=self.time_bins)
            self.data.append({'time': t, 'histogram': hist})

    def to_hdf5(self, hdf5_group):
        grp = hdf5_group.create_group(self.name)
        grp.create_dataset('time', data=[d['time'] for d in self.data])
        grp.create_dataset('histogram', data=np.array([d['histogram'] for d in self.data]), compression='gzip')

# --- Mode Analysis ---
class ModeAnalysis(Diagnostic):
    def __init__(self, name, r, z_indices, modes, field_manager: FieldManager):
        super().__init__(name, field_manager)
        self.r = r
        self.z_indices = z_indices
        self.modes = modes
        self.data = []

    def record(self, t, circuit, fluid, pic=None, radiation=None, state: SimulationState = None):
        rho = state.density
        dx = state.dx
        mode_amp = {}
        for kz in self.z_indices:
            dens_slice = rho[:, :, kz]
            R = np.sqrt(state._X[:, :, kz] ** 2 + state._Y[:, :, kz] ** 2)
            Theta = np.arctan2(state._Y[:, :, kz], state._X[:, :, kz])
            mask = np.abs(R - self.r) < (dx * 0.5)
            thetas = Theta[mask]
            vals = dens_slice[mask]
            for m in self.modes:
                Fm = np.sum(vals * np.exp(-1j * m * thetas))
                mode_amp[f"m{m}_z{kz}"] = np.abs(Fm)
        self.data.append({'time': t, 'mode_amplitudes': mode_amp})

    def to_hdf5(self, hdf5_group):
        grp = hdf5_group.create_group(self.name)
        grp.create_dataset('time', data=[d['time'] for d in self.data])
        mode_data = {k: [d['mode_amplitudes'].get(k, 0) for d in self.data] for k in self.data[0]['mode_amplitudes']}
        for k, v in mode_data.items():
            grp.create_dataset(k, data=v)

# --- Thomson Scattering ---
class ThomsonScattering(Diagnostic):
    def __init__(self, name, laser_wavelength, scattering_angle, position, field_manager: FieldManager):
        super().__init__(name, field_manager)
        self.laser_wavelength = laser_wavelength
        self.scattering_angle = scattering_angle
        self.position = np.array(position)
        self.data = []

    def record(self, t, circuit, fluid, pic=None, radiation=None, state: SimulationState = None):
        # Placeholder for Thomson scattering calculation
        # This would involve calculating the scattered power spectrum
        # based on electron density and temperature at the specified position.
        # This is a complex calculation and is left as a placeholder here.
        # You would need to implement the full Thomson scattering theory here.
        self.data.append({'time': t, 'signal': 0.0})

    def to_hdf5(self, hdf5_group):
        grp = hdf5_group.create_group(self.name)
        grp.create_dataset('time', data=[d['time'] for d in self.data])
        grp.create_dataset('signal', data=[d['signal'] for d in self.data])

# --- Main Diagnostics Class ---
class Diagnostics:
    def __init__(self, hdf5_filename, config, domain_lo, grid_shape, dx, gamma, field_manager: FieldManager, full_interval=10, adaptive_interval_threshold=0.1):
        self.hdf5_filename = hdf5_filename
        self.config = config
        self.domain_lo = np.array(domain_lo)
        self.grid_shape = grid_shape
        self.dx = dx
        self.cell_volume = dx**3
        self.gamma = gamma
        self.field_manager = field_manager
        self.full_interval = full_interval
        self.adaptive_interval_threshold = adaptive_interval_threshold
        self.diagnostics = []
        self.summary = []
        self.snapshots = []
        self.checkpoints = []
        self.timing = []
        self._step = 0
        self._last_time = None
        self._last_current = None
        self._last_rho_max = 0.0

        # Prepare grid coordinates for synthetic operations
        nx, ny, nz = self.grid_shape
        x0, y0, z0 = self.domain_lo
        xs = x0 + (np.arange(nx)+0.5)*dx
        ys = y0 + (np.arange(ny)+0.5)*dx
        zs = z0 + (np.arange(nz)+0.5)*dx
        self._X, self._Y, self._Z = np.meshgrid(xs, ys, zs, indexing='ij')

    def add_diagnostic(self, diagnostic):
        self.diagnostics.append(diagnostic)

    def record(self, t, circuit, fluid, pic=None, radiation=None, checkpoint_id=None, timings=None):
        try:
            start = time.perf_counter()

            I = circuit.get_current()
            V = circuit.get_voltage()

            # compute dI/dt synthetic Rogowski
            if self._last_time is None:
                dIdt = 0.0
            else:
                dt = t - self._last_time
                dIdt = (I - self._last_current) / dt if dt>0 else 0.0

            self._last_time = t
            self._last_current = I

            state = fluid.get_state()
            rho = state.density
            vel = state.velocity
            pres = state.pressure
            B = self.field_manager.get_B()

            # Energies
            E_th = np.sum(pres/(self.gamma-1.0)*self.cell_volume)
            v2 = np.sum(vel**2,axis=-1)
            E_kin = np.sum(0.5*rho*v2*self.cell_volume)
            B2 = np.sum(B**2,axis=-1)
            E_mag = np.sum(B2/(2*mu_0)*self.cell_volume)
            E_rad = radiation.total_radiated_energy if radiation else 0.0

            # Divergence of B
            divB_max, divB_l2 = self.compute_divergences(B)

            # Timing
            end = time.perf_counter()
            elapsed = end - start
            tdict = timings or {}
            tdict['diagnostics'] = elapsed
            self.timing.append(tdict)

            # Build summary record
            rec = {
                'time': t,
                'current': I,
                'voltage': V,
                'dI_dt': dIdt,
                'E_thermal': E_th,
                'E_kinetic': E_kin,
                'E_magnetic': E_mag,
                'E_radiated': E_rad,
                'divB_max': divB_max,
                'divB_l2': divB_l2,
                'timing': tdict,
                'checkpoint': checkpoint_id
            }
            self.summary.append(rec)

            # Call each diagnostic
            for diagnostic in self.diagnostics:
                diagnostic.record(t, circuit, fluid, pic, radiation, state)

            # Adaptive snapshot frequency
            rho_max = np.max(rho)
            if self._step % self.full_interval == 0 or abs(rho_max - self._last_rho_max) / self._last_rho_max > self.adaptive_interval_threshold:
                snap = {
                    'time': t,
                    'density': rho.copy(),
                    'pressure': pres.copy(),
                    'velocity': vel.copy(),
                    'magnetic': B.copy(),
                }
                self.snapshots.append({'snapshot': snap, 'checkpoint': checkpoint_id})
                self._last_rho_max = rho_max

            self._step += 1
        except Exception as e:
            logger.error(f"Error recording diagnostics: {e}")

    def get_latest(self):
        """Return the latest summary record."""
        return self.summary[-1] if self.summary else None

    def to_hdf5(self):
        """Write diagnostics to HDF5."""
        try:
            with h5py.File(self.hdf5_filename, 'w') as f:
                # Provenance / config
                cfggrp = f.create_group('config')
                for k,v in self.config.items():
                    cfggrp.attrs[k] = json.dumps(v)

                # Time series
                ts = f.create_group('time_series')
                keys = list(self.summary[0].keys())
                for key in keys:
                    data = [rec[key] for rec in self.summary]
                    ts.create_dataset(key, data=data, compression='gzip')

                # Snapshots
                snaps = f.create_group('snapshots')
                for idx, item in enumerate(self.snapshots):
                    grp = snaps.create_group(f'step_{idx}')
                    grp.attrs['checkpoint'] = item['checkpoint']
                    grp.create_dataset('time', data=item['snapshot']['time'])
                    grp.create_dataset('density', data=item['snapshot']['density'], compression='gzip')
                    grp.create_dataset('pressure', data=item['snapshot']['pressure'], compression='gzip')
                    grp.create_dataset('velocity', data=item['snapshot']['velocity'], compression='gzip')
                    grp.create_dataset('magnetic', data=item['snapshot']['magnetic'], compression='gzip')

                # Diagnostic-specific data
                diag_grp = f.create_group('diagnostics')
                for diagnostic in self.diagnostics:
                    diagnostic.to_hdf5(diag_grp)
        except Exception as e:
            logger.error(f"Error writing to HDF5: {e}")

    def to_vtk(self, filename_base):
        """Write snapshots to VTK files."""
        try:
            for idx, item in enumerate(self.snapshots):
                snap = item['snapshot']
                imageToVTK(f"{filename_base}_{idx}", cellData=snap)
        except Exception as e:
            logger.error(f"Error writing to VTK: {e}")

    def to_json(self):
        """Return latest summary as JSON."""
        latest = self.get_latest()
        if not latest:
            return '{}'
        clean = {}
        for k,v in latest.items():
            try:
                json.dumps(v)
                clean[k] = v
            except (TypeError, ValueError):
                clean[k] = str(v)
        return json.dumps(clean)

    def compute_divergences(self, B=None):
        """Computes the divergence of the magnetic field."""
        try:
            if B is None:
                B = self.field_manager.get_B()
            divB = self.field_manager.compute_divergence(B)
            divB_max = np.max(np.abs(divB))
            divB_l2 = np.linalg.norm(divB.flatten())
            return divB_max, divB_l2
        except Exception as e:
            logger.error(f"Error computing divergence of B: {e}")
            return 0.0, 0.0
