import numpy as np
import json
import logging
import time
try:  # optional dependency
    import h5py
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "h5py is required; install dpf2[warpx]"
    ) from exc
from scipy.constants import c, m_n, m_e, mu_0, e, epsilon_0, k as k_B
from scipy.interpolate import interp1d
try:
    from pyevtk.hl import imageToVTK
except Exception as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "pyevtk is required for VTK diagnostics. Install via 'pip install "
        "dpf2[diagnostics]' or 'pip install pyevtk'."
    ) from exc
from dpf2.core.bases import DiagnosticsBase
from .utils import FieldManager, SimulationState

# Classical electron radius (m)
r_e = e ** 2 / (4 * np.pi * epsilon_0 * m_e * c ** 2)

logger = logging.getLogger(__name__)

# --- Diagnostic Base Class ---
class Diagnostic(DiagnosticsBase):
    """Base class for all simulation diagnostics.

    Sub-classes are expected to produce a *payload* mapping of
    ``{name: array_like}`` for each time step.  The :meth:`record` method
    handles timestamping and error logging while :meth:`to_hdf5` serialises
    the accumulated records using a minimal schema::

        <diagnostic name>/
            time [N]        -- record times
            <payload-key>   -- stacked payload arrays

    Diagnostic authors should either supply a ``payload`` directly or a
    ``callback`` returning one when calling ``super().record``.  Additional
    metadata may be written by overriding :meth:`to_hdf5` and invoking
    ``super().to_hdf5`` first.
    """

    def __init__(self, name, field_manager: FieldManager):
        self.name = name
        self.field_manager = field_manager
        # Store individual records as dictionaries ``{"time": t, ...}``
        self.data: list[dict] = []

    def record(
        self,
        t,
        circuit,
        fluid,
        pic=None,
        radiation=None,
        state: SimulationState = None,
        *,
        payload=None,
        callback=None,
    ):
        """Record a diagnostic data point.

        Parameters
        ----------
        t:
            Simulation time in seconds.
        payload, callback:
            Either a mapping of dataset names to values or a callback
            returning such a mapping.  When provided, the data is stored
            alongside the timestamp.
        """

        try:
            if callback is not None:
                payload = callback()
            if payload is None:
                return
            # Attach simulation time and wall-clock timestamp for provenance
            record = {
                "time": float(t),
                "wall_time": float(time.time()),
            }
            record.update(payload)
            self.data.append(record)
        except Exception:  # pragma: no cover - error path
            logger.exception("diagnostic %s failed to record", self.name)

    def to_hdf5(self, hdf5_group):
        """Serialise records to an ``hdf5_group``.

        Returns the created HDF5 group to allow subclasses to attach
        additional datasets or attributes.
        """

        grp = hdf5_group.require_group(self.name)
        # Minimal openPMD-like identifiers
        grp.attrs.setdefault("openPMD", "1.1.0")
        grp.attrs.setdefault("openPMDextension", 0)

        try:
            if self.data:
                times = [d["time"] for d in self.data]
                wtimes = [d.get("wall_time", float("nan")) for d in self.data]
                t_ds = grp.create_dataset("time", data=times)
                t_ds.attrs["unitSI"] = 1.0
                wt_ds = grp.create_dataset("wall_time", data=wtimes)
                wt_ds.attrs["unitSI"] = 1.0

                keys = [k for k in self.data[0] if k not in ("time", "wall_time")]
                for k in keys:
                    arr = np.array([d[k] for d in self.data])
                    ds = grp.create_dataset(k, data=arr)
                    ds.attrs["unitSI"] = 1.0
        except Exception:  # pragma: no cover - error path
            logger.exception("diagnostic %s failed to serialise", self.name)
        return grp

# --- Interferometry ---
class Interferometry(Diagnostic):
    def __init__(self, name, p0, p1, field_manager: FieldManager):
        super().__init__(name, field_manager)
        self.p0 = np.array(p0)
        self.p1 = np.array(p1)

    def record(self, t, circuit, fluid, pic=None, radiation=None, state: SimulationState = None):
        def _compute():
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
            phase_shift = line_integral * 2.25e-18  # Example constant
            return {"phase_shift": phase_shift}

        super().record(t, circuit, fluid, pic=pic, radiation=radiation, state=state, callback=_compute)

    def to_hdf5(self, hdf5_group):
        return super().to_hdf5(hdf5_group)

# --- X-ray Detector ---
class XrayDetector(Diagnostic):
    def __init__(self, name, position, field_manager: FieldManager, energy_bins=None, detector_response=None):
        super().__init__(name, field_manager)
        self.position = np.array(position)
        self.energy_bins = energy_bins or [0, np.inf]
        self.detector_response = detector_response or (lambda E: 1.0)  # Default: constant efficiency

    def record(self, t, circuit, fluid, pic=None, radiation=None, state: SimulationState = None):
        def _compute():
            if not radiation:
                return None
            if hasattr(radiation, 'get_energy_resolved_emission'):
                energy_bins, P_rad_energy = radiation.get_energy_resolved_emission(state)
            else:
                P_rad_total = radiation.total_radiated_energy
                energy_bins = [0, np.inf]
                P_rad_energy = [P_rad_total]

            signal = 0.0
            for i, E_bin in enumerate(energy_bins[:-1]):
                detector_efficiency = self.detector_response(E_bin)
                dxs = state._X - self.position[0]
                dys = state._Y - self.position[1]
                dzs = state._Z - self.position[2]
                dist2 = dxs * dxs + dys * dys + dzs * dzs
                signal += np.sum(P_rad_energy[i] * state.cell_volume / dist2) * detector_efficiency
            return {"signal": signal}

        super().record(t, circuit, fluid, pic=pic, radiation=radiation, state=state, callback=_compute)

    def to_hdf5(self, hdf5_group):
        grp = super().to_hdf5(hdf5_group)
        grp.create_dataset('energy_bins', data=self.energy_bins)
        return grp

# --- Neutron Detector ---
class NeutronDetector(Diagnostic):
    def __init__(self, name, position, time_bins, field_manager: FieldManager, reaction='DD'):
        super().__init__(name, field_manager)
        self.position = np.array(position)
        self.time_bins = time_bins
        self.reaction = reaction

    def record(self, t, circuit, fluid, pic=None, radiation=None, state: SimulationState = None):
        def _compute():
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
                return {"histogram": hist}
            return None

        super().record(t, circuit, fluid, pic=pic, radiation=radiation, state=state, callback=_compute)

    def to_hdf5(self, hdf5_group):
        grp = super().to_hdf5(hdf5_group)
        grp.create_dataset('time_bins', data=self.time_bins)
        grp.attrs['reaction'] = self.reaction
        return grp

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
        # Precompute geometry factor for differential cross-section
        self._geom_factor = (1 + np.cos(self.scattering_angle) ** 2) / 2
        # Spectral grid will be created on first record
        self._wavelength_grid = None

    def record(self, t, circuit, fluid, pic=None, radiation=None, state: SimulationState = None):
        """Calculate a simplified Thomson scattering spectrum.

        The spectrum is approximated as a Gaussian profile whose width is
        determined by the electron temperature (thermal broadening) and whose
        amplitude scales with the local electron density and scattering
        geometry.
        """

        if state is None:
            return

        # Extract local electron density and temperature
        ne_grid = getattr(state, "electron_density", None)
        if ne_grid is None:
            ne_grid = getattr(state, "ion_density", state.density)
        Te_grid = getattr(state, "electron_temperature", None)

        dx, dy, dz = state.dx, state.dy, state.dz
        x0, y0, z0 = state.domain_lo
        xi = int((self.position[0] - x0) / dx)
        yi = int((self.position[1] - y0) / dy)
        zi = int((self.position[2] - z0) / dz)
        ne = float(ne_grid[xi, yi, zi])
        Te = float(Te_grid[xi, yi, zi]) if Te_grid is not None else 0.0

        # Thermal broadening of wavelength
        delta_lambda = self.laser_wavelength * np.sqrt(2 * k_B * max(Te, 1e-6) / m_e) / c
        if self._wavelength_grid is None:
            self._wavelength_grid = np.linspace(
                self.laser_wavelength - 5 * delta_lambda,
                self.laser_wavelength + 5 * delta_lambda,
                100,
            )
        wl = self._wavelength_grid
        width = delta_lambda if delta_lambda > 0 else self.laser_wavelength * 1e-9
        spectrum = (
            ne
            * r_e ** 2
            * self._geom_factor
            * np.exp(-0.5 * ((wl - self.laser_wavelength) / width) ** 2)
        )

        self.data.append({"time": t, "wavelength": wl, "spectrum": spectrum})

    def to_hdf5(self, hdf5_group):
        grp = hdf5_group.create_group(self.name)
        times = [d["time"] for d in self.data]
        grp.create_dataset("time", data=times)
        if self.data:
            grp.create_dataset("wavelength", data=self.data[0]["wavelength"])
            spectra = np.array([d["spectrum"] for d in self.data])
            grp.create_dataset("spectrum", data=spectra)

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
            coupled_currents = getattr(circuit, "get_coupled_currents", lambda: None)()
            coupled_voltages = getattr(circuit, "get_coupled_voltages", lambda: None)()

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
            if coupled_currents is not None:
                rec['coupled_currents'] = coupled_currents
            if coupled_voltages is not None:
                rec['coupled_voltages'] = coupled_voltages
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
                f.attrs.update({"openPMD": "1.1.0", "openPMDextension": 0})
                # Provenance / config snapshot
                prov = f.create_group('provenance')
                prov.attrs['created'] = time.time()
                prov.attrs['software'] = 'dpf2'
                prov.create_dataset('config', data=np.string_(json.dumps(self.config)))

                # Time series
                ts = f.create_group('time_series')
                ts.attrs.update({"openPMD": "1.1.0", "openPMDextension": 0})
                keys = list(self.summary[0].keys())
                for key in keys:
                    data = [rec[key] for rec in self.summary]
                    ds = ts.create_dataset(key, data=data, compression='gzip')
                    ds.attrs['unitSI'] = 1.0

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
                diag_grp.attrs.update({"openPMD": "1.1.0", "openPMDextension": 0})
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
