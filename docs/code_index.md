# DPF2 Code Index

Generated with `dpf2.indexing`.

Indexed modules: 226

## dpf2.ablation
*Source:* `src/dpf2/ablation.py`

_No module docstring available._

### Functions
- ``insulator_sleeve_area``: Return inner surface area of cylindrical insulator sleeve.
- ``ablation_mass_energy_source``: Compute mass and energy source due to insulator ablation.

## dpf2.advanced_options
*Source:* `src/dpf2/advanced_options.py`

_No module docstring available._

### Classes
- ``AdvancedOptions``: Developer and experimental toggles for DPF simulations.

## dpf2.ai
*Source:* `src/dpf2/ai/__init__.py`

Surrogate model interfaces for AI integration.

## dpf2.ai.simple_surrogates
*Source:* `src/dpf2/ai/simple_surrogates.py`

Lightweight runtime helpers for linear surrogate models.

### Classes
- ``LinearSurrogate``: Simple linear regression surrogate ``y = a*x + b``.
- ``ONNXLinearSurrogate``: Wrapper around :class:`ONNXSurrogateModel` with domain checks.

### Functions
- ``load_yield_surrogate``: Return the surrogate model for neutron yield.
- ``load_pinch_time_surrogate``: Return the surrogate model for pinch time.

## dpf2.ai.surrogate
*Source:* `src/dpf2/ai/surrogate.py`

Abstractions for machine learning surrogate models.

### Classes
- ``SurrogateModel``: Base class for ML surrogate models.
- ``TorchSurrogateModel``: Surrogate model backed by a PyTorch ``ScriptModule``.
- ``ONNXSurrogateModel``: Surrogate model using ``onnxruntime`` for inference.

## dpf2.ai.training
*Source:* `src/dpf2/ai/training.py`

Utilities for training surrogate models.

### Functions
- ``load_numpy_dataset``: Create a torch ``DataLoader`` from numpy arrays.
- ``train_torch_model``: Train ``model`` using data from ``dataloader``.

## dpf2.amrex_settings
*Source:* `src/dpf2/amrex_settings.py`

_No module docstring available._

### Classes
- ``ElectrodeGeometry``: Electrode mesh and geometry configuration.
- ``AmrexSettings``: Configuration schema for AMReX solver settings.

## dpf2.axial_sheath
*Source:* `src/dpf2/axial_sheath.py`

_No module docstring available._

### Classes
- ``SheathResult``: _No docstring._
- ``AxialSheathModel``: Evolve axial sheath motion using :math:`J imes B` forcing.

## dpf2.benchmark_matching
*Source:* `src/dpf2/benchmark_matching.py`

_No module docstring available._

### Classes
- ``BenchmarkMatching``: Benchmarking and matching configuration.

## dpf2.boundary_conditions
*Source:* `src/dpf2/boundary_conditions.py`

_No module docstring available._

### Classes
- ``KineticSheath``: Evolve ion/electron fluxes at a boundary.
- ``BoundaryTypeEnum``: Defines boundary condition types per face and field.
- ``BoundaryConditions``: Validated schema for domain boundary conditions.

## dpf2.breakdown
*Source:* `src/dpf2/breakdown/__init__.py`

Breakdown models for DPF simulations.

## dpf2.breakdown.flashover
*Source:* `src/dpf2/breakdown/flashover.py`

_No module docstring available._

### Classes
- ``FlashoverParameters``: Parameters controlling the stochastic delay model.

### Functions
- ``conditioning_curve``: Return the conditioning multiplier for ``shot``.
- ``seea_stochastic_delay``: Sample a stochastic flashover delay.
- ``delay_series``: Generate a sequence of flashover delays over multiple shots.
- ``delay_statistics``: Return simple statistics for ``delays``.
- ``holdoff_voltage``: Sample a hold-off voltage for ``geometry``.
- ``holdoff_series``: Generate a sequence of hold-off voltages over multiple shots.
- ``jitter_statistics``: Return statistics for jitter values.

## dpf2.chemistry
*Source:* `src/dpf2/chemistry/__init__.py`

_No module docstring available._

### Classes
- ``ChemistryModel``: Return average ionisation state ``Zbar`` for density and temperature.
- ``SahaEquilibrium``: Toy Saha equilibrium model.
- ``FlychkTable``: Interpolate a pre-computed FLYCHK table (T vs Zbar).

### Functions
- ``create_chemistry``: _No docstring._

## dpf2.chemistry.kinetics
*Source:* `src/dpf2/chemistry/kinetics.py`

_No module docstring available._

### Classes
- ``RateTable``: Tabulated ionisation and recombination rate coefficients.
- ``RateEquations``: Simple multi-species collisional–radiative model.
- ``ImpurityModel``: Convenience wrapper for evolving impurity charge states.
- ``MultiSpeciesTransport``: Very small multi‑species diffusion and wall ablation model.

## dpf2.chemistry.metadata
*Source:* `src/dpf2/chemistry/metadata.py`

_No module docstring available._

### Classes
- ``DatasetMetadata``: Minimal provenance for a tabulated data set.

### Functions
- ``_require_metadata_fields``: Validate required provenance fields and build :class:`DatasetMetadata`.
- ``_load``: _No docstring._
- ``load_adas_metadata``: Load metadata for an ADAS table.
- ``load_lxcat_metadata``: Load metadata for an LXCat cross-section table.

## dpf2.circuit
*Source:* `src/dpf2/circuit/__init__.py`

Circuit subpackage providing models for distributed networks.

## dpf2.circuit.distributed
*Source:* `src/dpf2/circuit/distributed.py`

_No module docstring available._

### Classes
- ``TransmissionLineSegment``: Simple RLC transmission line segment with optional parasitics.
- ``BlumleinSection``: Transmission line segment with a triggerable switch representing a Blumlein block.
- ``MultiSectionLine``: Container representing a chain of transmission line segments.
- ``PlasmaInductance``: Dynamic inductive branch sourced from an external plasma solver.

### Functions
- ``_interp_profile``: Return interpolated profile value at time ``t``.
- ``assemble_matrices``: Assemble node based ``R``, ``L`` and ``C`` matrices for a network.

## dpf2.circuit.switches
*Source:* `src/dpf2/circuit/switches.py`

_No module docstring available._

### Classes
- ``TriggeredSwitch``: Ideal resistive switch with optional trigger times and parasitics.
- ``CrowbarStage``: Resistive crowbar stage engaging at ``trigger_time`` with optional jitter.

## dpf2.circuit_config
*Source:* `src/dpf2/circuit_config.py`

Circuit configuration schema for DPF simulations.

### Classes
- ``SegmentConfig``: Configuration for a single transmission line segment.
- ``SwitchConfig``: Configuration for a simple resistive switch.
- ``RLCSectionConfig``: Configuration for a lumped RLC driver section.
- ``CrowbarStageConfig``: Configuration for a resistive crowbar branch.
- ``CircuitConfig``: Validated external circuit configuration.

## dpf2.circuit_solver
*Source:* `src/dpf2/circuit_solver.py`

Simple RLC circuit solver for DPF simulations.

### Classes
- ``RLCCircuit``: Series RLC circuit parameters.
- ``CircuitSolver``: Compute current evolution for a series RLC circuit.

### Functions
- ``_gauss``: Return Gaussian sample using available RNG backends.
- ``_profile_to_interp``: Return interpolation functions for profile value and derivative.
- ``_run_distributed_network``: Integrate a chain of RLC segments using a state-space model.
- ``run_circuit_simulation``: Run RLC discharge with optional plasma and mutual inductance.

## dpf2.cli
*Source:* `src/dpf2/cli/__init__.py`

Command line interfaces for DPF2.

## dpf2.cli.benchmark
*Source:* `src/dpf2/cli/benchmark.py`

_No module docstring available._

### Functions
- ``_load_config``: Load a BenchmarkMatching configuration from ``path``.
- ``_load_waveform``: Return time and value arrays from a CSV file.
- ``benchmark``: Utilities for running frozen benchmarks.
- ``run``: Execute ``case`` and overlay results against references.
- ``match_benchmark``: Compare simulation outputs against benchmark traces.

## dpf2.cli.errors
*Source:* `src/dpf2/cli/errors.py`

Centralised error codes and remediation hints for the CLI.

### Classes
- ``CLIErrorInfo``: Definition of a CLI error with structured metadata.

### Functions
- ``format_error``: Format an error message with its code and remediation tip.

## dpf2.cli.lab
*Source:* `src/dpf2/cli/lab.py`

Utilities for lab-mode reproducibility manifests.

### Functions
- ``_code_hash``: Return current git commit hash if available.
- ``_environment``: Capture basic execution environment details.
- ``write_manifest``: Write a JSON manifest capturing reproducibility metadata.

## dpf2.cli.legacy_cli
*Source:* `src/dpf2/cli/legacy_cli.py`

Command line interface for the DPF simulator.

### Functions
- ``build_parser``: _No docstring._
- ``main``: _No docstring._

## dpf2.cli.main
*Source:* `src/dpf2/cli/main.py`

Command line interface for DPF2.

### Functions
- ``_prompt_with_range``: Prompt the user for a floating point value within a range.
- ``_validate_range``: Validate that ``value`` lies within the given range.
- ``_to_float``: Best-effort conversion to float supporting stubbed types.
- ``_launch_notebook``: Launch Jupyter with DPF2 helpers preloaded.
- ``build_config_wizard``: Interactively build a :class:`DPFConfig` with contextual hints.
- ``main``: Entry point for the DPF2 command line interface.
- ``simulate``: Run a DPF simulation.
- ``validate``: Run a validation simulation and compare with experimental data.
- ``validate_config``: Validate a configuration file.
- ``plot``: Plot current and voltage from simulation outputs.
- ``plot_run``: Quickly plot current and voltage from an existing run directory.
- ``param_sweep_cmd``: Run a parameter sweep and optionally generate KPI plots.
- ``uq_sweep_cmd``: Run a multi-parameter sweep using UQ sampling schemes.
- ``latin_hypercube_cmd``: Generate Latin hypercube samples for batch sweeps.
- ``sobol_sample_cmd``: Generate Sobol sequence samples for batch sweeps.
- ``uq_stats_cmd``: Compute statistics from a UQ sweep results file.
- ``scaling_cmd``: Run a sweep and report fitted scaling exponents.
- ``make_surrogate``: Train a yield-vs-pressure surrogate and export an ONNX model.
- ``diagnostics``: Generate synthetic diagnostics from a coupling history.
- ``share``: Export a configuration for sharing with classmates.
- ``schema``: Print the configuration schema.
- ``wizard``: Interactive wizard for building a configuration.
- ``index``: Generate a Markdown index of the code base.

## dpf2.cli.uq_run
*Source:* `src/dpf2/cli/uq_run.py`

Command line interface for running UQ calibration routines.

### Functions
- ``_load_waveform``: Load two-column ``time,current`` waveform data.
- ``main``: Entry point for the ``uq_run`` CLI.

## dpf2.cli.validate
*Source:* `src/dpf2/cli/validate.py`

Validation command line interface for DPF2.

### Functions
- ``_build_validation_suite``: Create a :class:`ValidationSuite` for bundled validation data.
- ``_load_experimental``: Load experimental observables from disk.
- ``_simulation_observables``: Extract observables from the simulation results.
- ``_plot_overlays``: Generate overlay plots of simulation vs. experiment.
- ``run_validation``: Execute a simulation and validate against experimental data.
- ``main``: _No docstring._

## dpf2.cli.validate_datasets
*Source:* `src/dpf2/cli/validate_datasets.py`

_No module docstring available._

### Functions
- ``main``: _No docstring._

## dpf2.core
*Source:* `src/dpf2/core/__init__.py`

Core components for simplified DPF simulations.

## dpf2.core.bases
*Source:* `src/dpf2/core/bases.py`

This file provides canonical interfaces for plasma, circuit and

### Classes
- ``CouplingState``: Coupling information exchanged between plasma and circuit solvers.
- ``PlasmaSolverBase``: Interface for plasma solvers coupled to an external circuit.
- ``CircuitSolverBase``: Interface for external circuit solvers.
- ``DiagnosticsBase``: Interface for simulation diagnostics.

## dpf2.core.circuit
*Source:* `src/dpf2/core/circuit.py`

_No module docstring available._

### Classes
- ``RLCCircuitSolver``: Series RLC circuit with optional plasma coupling.

## dpf2.core.config
*Source:* `src/dpf2/core/config.py`

Configuration schema for DPF simulations.

### Classes
- ``DPFConfig``: Simulation configuration parameters.

## dpf2.core.external_circuit
*Source:* `src/dpf2/core/external_circuit.py`

_No module docstring available._

### Classes
- ``ExternalCircuit``: Minimal external circuit model.

## dpf2.core.simulation
*Source:* `src/dpf2/core/simulation.py`

Core simulation driver.

### Classes
- ``DPFSimulation``: Main class orchestrating a DPF simulation.

## dpf2.core_schema
*Source:* `src/dpf2/core_schema.py`

Core configuration schemas for DPF simulations.

### Classes
- ``GeometryType``: Defines valid simulation geometries.
- ``ModeType``: Defines valid DPF solver modes.
- ``UnitsSystem``: Defines the base unit system used for internal normalization.
- ``ValidationPolicy``: Controls schema enforcement mode on load/override.
- ``EOSModel``: _No docstring._
- ``ResistivityModel``: _No docstring._
- ``IonizationModel``: _No docstring._
- ``IonizationFallback``: _No docstring._
- ``RadiationModel``: _No docstring._
- ``RadiationTransportModel``: _No docstring._
- ``LineEscapeMethod``: _No docstring._
- ``RadiationGeometryModel``: _No docstring._
- ``InstabilityModel``: _No docstring._
- ``CircuitFaultTypeEnum``: Enumerates possible circuit fault types.
- ``ConfigSectionBase``: Base class for all configuration sections.
- ``MaterialOpacity``: Per-material opacity definition across radiation groups.
- ``RadiationSettings``: Basic radiation configuration including group counts and opacities.
- ``DPFConfig``: Root configuration object.

### Functions
- ``to_camel_case``: _No docstring._

## dpf2.coupled_models
*Source:* `src/dpf2/coupled_models.py`

_No module docstring available._

### Classes
- ``EndToEndResult``: _No docstring._
- ``CoupledEndToEndModel``: Run pre-pulse, sheath, and pinch phases sequentially.
- ``NeutralPlasmaResult``: Result container for neutral/plasma coupled runs.
- ``NeutralPlasmaCoupler``: Couple a DSMC neutral solver to an arbitrary plasma solver.

## dpf2.device_profiles
*Source:* `src/dpf2/device_profiles.py`

Validated schema for Dense Plasma Focus device profiles.

### Classes
- ``InsulatorSleeve``: Geometry specification for the insulator sleeve.
- ``DeviceEntry``: Single device entry describing geometry and circuit parameters.
- ``DeviceProfiles``: Repository of known DPF machine configurations.

## dpf2.diagnostics
*Source:* `src/dpf2/diagnostics/__init__.py`

_No module docstring available._

### Classes
- ``SXRModel``: Simplified soft X-ray detector model.
- ``TOFModel``: Simplified neutron time-of-flight detector model.
- ``OutputField``: _No docstring._
- ``DetectorArrayGenerator``: Procedural generation for detector arrays.
- ``Diagnostics``: Diagnostics configuration schema.

### Functions
- ``apply_noise``: Apply a noise model to a signal sequence.
- ``apply_detector_response``: Apply detector response mapping to a signal sequence.

## dpf2.diagnostics.detector_models
*Source:* `src/dpf2/diagnostics/detector_models.py`

_No module docstring available._

### Functions
- ``_solid_angle``: Return the small-angle approximation for detector solid angle.
- ``cr39_response``: Estimate track density for a CR-39/RCF detector.
- ``time_gated_scintillator_response``: Integrate a TOF histogram within a gating window and apply geometry.

## dpf2.diagnostics.interferometry
*Source:* `src/dpf2/diagnostics/interferometry.py`

_No module docstring available._

### Functions
- ``interferometer_phase_shift``: Compute optical phase shift from line-integrated electron density.

## dpf2.diagnostics.iv_probes
*Source:* `src/dpf2/diagnostics/iv_probes.py`

_No module docstring available._

### Functions
- ``load_response``: Load an I-V probe response description from *path*.
- ``_rlc_kernel``: Generate an impulse response for a simple series RLC circuit.
- ``apply_response``: Apply the configured probe response effects to the input *signal*.

## dpf2.diagnostics.modes
*Source:* `src/dpf2/diagnostics/modes.py`

Fourier mode decomposition utilities.

### Functions
- ``azimuthal_mode_spectrum``: Return the azimuthal Fourier mode spectrum of ``field``.
- ``azimuthal_decomposition``: Return complex azimuthal Fourier coefficients of ``field``.
- ``growth_rate``: Estimate exponential growth rates between two spectra.
- ``lh_azimuthal_power``: Return azimuthal power near the lower-hybrid frequency.
- ``log_impedance_ratio``: Logarithmic plasma impedance relative to Spitzer prediction.

## dpf2.diagnostics.neutron
*Source:* `src/dpf2/diagnostics/neutron/__init__.py`

Neutron diagnostic utilities.

## dpf2.diagnostics.neutron.angular_distribution
*Source:* `src/dpf2/diagnostics/neutron/angular_distribution.py`

Angular distribution utilities for neutron diagnostics.

### Functions
- ``per_angle_yield``: Integrate spectra at each angle to obtain per-angle yields.
- ``forward_radial_backward_totals``: Aggregate per-angle yields into forward, radial, and backward totals.
- ``directional_yield``: Convenience wrapper returning directional totals directly.

## dpf2.diagnostics.neutron.tof_synthetic
*Source:* `src/dpf2/diagnostics/neutron/tof_synthetic.py`

_No module docstring available._

### Functions
- ``cross_correlate``: Return cross-correlation of sequences ``a`` and ``b``.
- ``cross_correlation_with_iv``: Cross-correlate ``counts`` with the ``I*V`` power history.
- ``synthetic_tof_from_iv``: Generate a synthetic neutron time-of-flight signal from I–V traces.

## dpf2.diagnostics.neutron_spectra
*Source:* `src/dpf2/diagnostics/neutron_spectra.py`

_No module docstring available._

### Classes
- ``Detector``: Simple representation of a neutron detector.
- ``DetectorLayout``: Container describing a collection of detectors on a ring.

### Functions
- ``synthetic_tof_spectrum``: Generate a simple neutron time-of-flight histogram.
- ``angular_spectrum``: Create a simple angular yield spectrum using a cosine model.
- ``anisotropy_metric``: Return a rudimentary anisotropy metric ``(max - min) / mean``.
- ``load_detector_layout``: Load a :class:`DetectorLayout` from a JSON or STL file.
- ``time_resolved_spectra``: Compute time-resolved spectra for each detector in ``layout``.
- ``directional_time_resolved_spectra``: Return forward, radial and backward time-resolved spectra.
- ``forward_radial_backward_counts``: Aggregate counts into forward, radial and backward groups.
- ``directional_counts_from_geometry``: Load a geometry file and return forward/radial/backward totals.
- ``anisotropy_ratios``: Return simple forward/backward and radial/backward ratios.
- ``cross_correlate_tof_with_circuit``: Cross-correlate a ToF spectrum with a circuit waveform.
- ``correlate_tof_peaks_with_circuit_iv``: Correlate ToF peaks with circuit power derived from ``I`` and ``V``.

## dpf2.diagnostics.neutron_yield
*Source:* `src/dpf2/diagnostics/neutron_yield.py`

_No module docstring available._

### Classes
- ``IonBeamEDF``: Interface providing ion energy distributions by angle.

### Functions
- ``compute_neutron_yield``: Compute total neutron yield from a reaction rate history.
- ``compute_beam_target_yield``: Integrate EDF×σ(E) and compute TOF histograms for each angle.
- ``compute_thermonuclear_yield``: Compute thermonuclear neutron yield from ion density and reactivity.
- ``yield_components_with_anisotropy``: Return beam-target and thermal yields with angular distribution.
- ``simulate_tof_detectors``: Generate synthetic neutron time-of-flight detector histograms.
- ``save_anisotropic_spectrum_hdf5``: Save anisotropic neutron spectrum in an HDF5 file with per-detector datasets.
- ``save_tof_hdf5``: Export synthetic time-of-flight detector data to an HDF5 file.
- ``tof_iv_cross_correlation``: Return zero-lag correlation between TOF signal and I/V traces.
- ``ez_beam_correlation``: Return zero-lag correlation between ``E_z`` and beam signals.
- ``angular_yield_map``: Wrapper around :func:`compute_directional_spectrum` for diagnostics.
- ``save_angular_yield_map_hdf5``: Export angular yield map to HDF5 using standard spectrum layout.

## dpf2.diagnostics.performance_metrics
*Source:* `src/dpf2/diagnostics/performance_metrics.py`

_No module docstring available._

### Functions
- ``compute_performance_metrics``: Compute basic performance KPIs for a DPF system.
- ``estimate_lifetime_sputtering``: Estimate electrode lifetime from a simple sputtering model.
- ``export_performance_metrics``: Save KPI data to CSV, HDF5 and generate basic visualisations.

## dpf2.diagnostics.pinhole_imaging
*Source:* `src/dpf2/diagnostics/pinhole_imaging.py`

_No module docstring available._

### Functions
- ``pinhole_image``: Generate a simple pinhole camera image from point sources.

## dpf2.diagnostics.plasma
*Source:* `src/dpf2/diagnostics/plasma.py`

_No module docstring available._

### Functions
- ``bennett_radius``: Return the Bennett pinch radius.
- ``plasma_beta``: Compute plasma beta for a cell.
- ``alfven_mach_number``: Return the Alfven Mach number for a cell.
- ``magnetic_reynolds_number``: Compute the magnetic Reynolds number.
- ``lundquist_number``: Compute the Lundquist number.
- ``save_density_temperature_map_hdf5``: Save 2D density and temperature maps to an HDF5 file.
- ``compute_eedf``: Compute an electron energy distribution function (EEDF).
- ``save_eedf_hdf5``: Save an EEDF to an HDF5 file.

## dpf2.diagnostics.quality_dashboard
*Source:* `src/dpf2/diagnostics/quality_dashboard.py`

_No module docstring available._

### Classes
- ``QualityDashboard``: Collect and persist basic quality metrics for simulation steps.

### Functions
- ``_main``: _No docstring._

## dpf2.diagnostics.regime_panel
*Source:* `src/dpf2/diagnostics/regime_panel.py`

Regime diagnostic panel tracking dimensionless parameters over time.

### Classes
- ``RegimePanel``: Compute and log dimensionless plasma regime parameters.

## dpf2.diagnostics.scope_trace
*Source:* `src/dpf2/diagnostics/scope_trace.py`

_No module docstring available._

### Functions
- ``compute_scope_trace``: Baseline subtract a scope trace.

## dpf2.diagnostics.streaming
*Source:* `src/dpf2/diagnostics/streaming.py`

Real-time streaming diagnostics for Dense Plasma Focus simulations.

### Classes
- ``NeutronYieldStreamer``: Stream approximate neutron production rate.
- ``XRayEmissionStreamer``: Stream simplified X-ray emission power.
- ``RealTimeComparator``: Hold experimental measurements and compare to simulation streams.

## dpf2.diagnostics.synthetic_signals
*Source:* `src/dpf2/diagnostics/synthetic_signals.py`

Lightweight synthetic diagnostic signal generators.

### Functions
- ``_apply``: _No docstring._
- ``current_waveform``: Return the circuit current for each time step.
- ``voltage_waveform``: Return the capacitor voltage for each time step.
- ``coupled_current_waveform``: Return current including simple back-reaction term.
- ``coupled_voltage_waveform``: Return voltage including mutual inductance contribution.
- ``_load_calibration_curve``: Load ``(time, response)`` arrays from a calibration file.
- ``_apply_instrument_response``: Convolve ``values`` with an impulse response defined by ``resp_t``/``resp_v``.
- ``_cfg_calibration``: Return calibration path for ``attr`` from :class:`DPFConfig`.
- ``rogowski_signal``: Compute a synthetic Rogowski coil signal ``dI/dt``.
- ``bdot_signal``: Generate a simple B-dot probe signal assuming axial field geometry.
- ``sxr_diode_signal``: Apply soft X-ray diode filter response to a signal history.
- ``neutron_tof_signal``: Generate a neutron time-of-flight detector signal.
- ``angular_neutron_spectrum``: Return a cosine-based angular neutron spectrum.

## dpf2.diagnostics.thresholds
*Source:* `src/dpf2/diagnostics/thresholds.py`

_No module docstring available._

### Classes
- ``ThresholdDashboard``: Record threshold metrics and feed them to a JSON dashboard.

### Functions
- ``compute_debye_length``: Return the electron Debye length in metres.
- ``plasma_inductance_circuit``: Compute effective inductance from circuit quantities.
- ``check_thresholds``: Check basic numerical stability thresholds.

## dpf2.diagnostics.xray
*Source:* `src/dpf2/diagnostics/xray.py`

_No module docstring available._

### Functions
- ``load_response``: Load an X-ray detector response description from *path*.
- ``apply_response``: Apply detector response effects to *signal* sampled at *times*.

## dpf2.diagnostics.xray_imaging
*Source:* `src/dpf2/diagnostics/xray_imaging.py`

_No module docstring available._

### Functions
- ``_be_filter``: Crude beryllium filter transmission.
- ``_al_filter``: Crude aluminium filter transmission.
- ``_ti_filter``: Crude titanium filter transmission.
- ``apply_filter_pack``: Apply transmission of a filter pack to photon energies.
- ``xray_image``: Form a simple pinhole X-ray image from photon positions and energies.
- ``pinhole_camera``: Synthesize a pinhole camera image including filter pack effects.

## dpf2.diagnostics.xray_spectra
*Source:* `src/dpf2/diagnostics/xray_spectra.py`

_No module docstring available._

### Functions
- ``compute_xray_spectrum``: Generate an X-ray spectrum from photon energies.

## dpf2.distributed_circuit
*Source:* `src/dpf2/distributed_circuit.py`

Compatibility layer for distributed circuit models.

## dpf2.dpf_config
*Source:* `src/dpf2/dpf_config.py`

_No module docstring available._

### Classes
- ``SimulationControl``: _No docstring._
- ``BreakdownModel``: _No docstring._
- ``PaschenModel``: _No docstring._
- ``InitialConditions``: _No docstring._
- ``PhysicsModels``: _No docstring._
- ``CircuitConfig``: _No docstring._
- ``ElectrodeGeometry``: _No docstring._
- ``AmrexSettings``: _No docstring._
- ``WarpXSettings``: _No docstring._
- ``Diagnostics``: _No docstring._
- ``BenchmarkMatching``: _No docstring._
- ``FaceType``: _No docstring._
- ``BoundaryConditions``: _No docstring._
- ``ParallelSettings``: _No docstring._
- ``DPFConfig``: _No docstring._

## dpf2.eos
*Source:* `src/dpf2/eos/__init__.py`

_No module docstring available._

### Classes
- ``EOSBase``: Common EOS interface.
- ``IdealGasEOS``: Ideal gas with constant heat capacity.
- ``TabulatedEOS``: Equation of state based on tabulated density/temperature data.
- ``RealGasEOS``: Multi‑species real gas EOS using tabulated thermochemistry.

### Functions
- ``_parse_mixture_fractions``: Normalise ``mixture_fractions`` input.
- ``create_eos``: Factory for EOS implementations.

## dpf2.eos.ideal_gas
*Source:* `src/dpf2/eos/ideal_gas.py`

Ideal gas equation of state backend.

### Classes
- ``IdealGasEOS``: Ideal‑gas equation of state with optional electron component.

## dpf2.exceptions
*Source:* `src/dpf2/exceptions.py`

Custom exception hierarchy for the DPF2 project.

### Classes
- ``DPFError``: Base class for all custom exceptions raised by DPF2.
- ``ConfigurationError``: Raised when a simulation or server configuration is invalid.
- ``SimulationRuntimeError``: Raised for runtime errors occurring during simulation execution.
- ``OutOfDomainError``: Raised when model inputs are outside the training domain.
- ``ServerError``: Base class for server-related errors.
- ``ExportError``: Raised when exporting results from the server fails.

## dpf2.experimental_variability
*Source:* `src/dpf2/experimental_variability.py`

_No module docstring available._

### Classes
- ``ExperimentalVariabilityModel``: Configuration of stochastic and environmental variability.
- ``MonteCarloVariability``: Utility for applying Monte-Carlo variations to a :class:`DPFConfig`.

## dpf2.fields.psatd_solver
*Source:* `src/dpf2/fields/psatd_solver.py`

_No module docstring available._

### Classes
- ``PSATDSolver``: Very small 1-D pseudo-spectral analytic time-domain solver.

## dpf2.fusion
*Source:* `src/dpf2/fusion.py`

_No module docstring available._

### Functions
- ``bosch_hale_dd``: Approximate D-D reactivity from Bosch-Hale parameterization.
- ``dd_fusion_rates``: Return separate thermonuclear and beam–target D–D fusion rates.
- ``dd_channel_fractions``: Return fractional contributions of thermonuclear and beam–target rates.
- ``dd_beam_target_angular_spectrum``: Return simple beam–target yield per angle.
- ``dd_directional_yields``: Return forward, radial and backward yield components.
- ``dd_yield_components``: Return D-D neutron yields with propagated uncertainties.

## dpf2.geometry
*Source:* `src/dpf2/geometry/__init__.py`

Geometry utilities for DPF simulations.

## dpf2.geometry.axisymmetric
*Source:* `src/dpf2/geometry/axisymmetric.py`

_No module docstring available._

### Classes
- ``AxisymmetricProfile``: Axisymmetric mesh profile defined by radial and axial coordinates.

## dpf2.geometry.inductance
*Source:* `src/dpf2/geometry/inductance.py`

_No module docstring available._

### Functions
- ``coaxial_inductance``: Return inductance of a straight coaxial plasma column.
- ``loop_mutual_inductance``: Approximate mutual inductance between two coaxial circular loops.

## dpf2.geometry.loaders
*Source:* `src/dpf2/geometry/loaders.py`

_No module docstring available._

### Functions
- ``_parse_step_like``: Parse a small subset of STEP/IGES style geometry.
- ``load_cad_geometry``: Load a minimal CAD style geometry description from ``path``.
- ``load_unstructured_mesh``: Load a very small subset of an unstructured mesh format.
- ``load_axisymmetric_mesh``: Load a minimal axisymmetric mesh description.

## dpf2.geometry.parameterized
*Source:* `src/dpf2/geometry/parameterized.py`

_No module docstring available._

### Classes
- ``TaperedGeometry``: Simple tapered column geometry.
- ``HollowGeometry``: Cylindrical geometry with an inner bore.
- ``ReentrantGeometry``: Re-entrant cavity geometry defined by straight segments.

## dpf2.geometry.triple_junction
*Source:* `src/dpf2/geometry/triple_junction.py`

_No module docstring available._

### Functions
- ``triple_junction_field``: Return the triple-junction field factor for ``geometry``.
- ``set_triple_junction_field_map``: Override or define a field map entry for ``geometry``.
- ``triple_junction_enhancement``: Return a geometry-dependent triple-junction field enhancement.

## dpf2.gpu_utils
*Source:* `src/dpf2/gpu_utils.py`

Utility helpers for optional GPU acceleration.

### Functions
- ``solve_linear``: Solve ``M x = b`` using the active array module ``xp``.

## dpf2.grid_resolution
*Source:* `src/dpf2/grid_resolution.py`

_No module docstring available._

### Classes
- ``GridResolution``: Domain and grid resolution configuration.

## dpf2.gui
*Source:* `src/dpf2/gui/__init__.py`

GUI utilities for DPF2.

## dpf2.gui.dashboard
*Source:* `src/dpf2/gui/dashboard.py`

Utilities for launching the built-in web dashboard and analysis helpers.

### Functions
- ``launch``: Start the Flask-based dashboard.
- ``run_sampling``: Run sampling experiments and visualize uncertainty bands.
- ``launch_sampling``: Convenience wrapper for :func:`run_sampling` used by dashboards.
- ``calibrate_from_file``: Calibrate model parameters against experimental data in ``data_file``.
- ``plot_posterior_distributions``: Plot marginal posterior distributions for calibrated parameters.
- ``plot_kpi_with_domain``: Plot KPI values with error bars and highlight training domain.
- ``flashover_delay_distribution``: Return stochastic flashover delays for GUI consumption.
- ``flashover_conditioning_curve``: Return conditioning multipliers suitable for plotting.

## dpf2.gui.interactive
*Source:* `src/dpf2/gui/interactive.py`

Dash-based interactive GUI for parametric studies.

### Functions
- ``_ensure_dash``: Raise if :mod:`dash` is unavailable.
- ``launch``: Launch the Dash-based GUI.

## dpf2.gui.project_manager
*Source:* `src/dpf2/gui/project_manager.py`

_No module docstring available._

### Classes
- ``ProjectManager``: Manage simulation sweeps and KPI comparisons.

## dpf2.gui.qt_sweep
*Source:* `src/dpf2/gui/qt_sweep.py`

_No module docstring available._

### Classes
- ``SweepPanel``: Simple widget displaying overlays of sweep metrics.
- ``_SweepWindow``: Main window for the sweep GUI.

### Functions
- ``_ensure_qt``: Raise if :mod:`PyQt5` is unavailable.
- ``plot_yield_vs_S_gv``: Plot yield versus shock parameter ``S`` and mark GV prediction.
- ``plot_yield_vs_param``: Plot yield against a swept parameter and highlight optimal ``S``.
- ``launch``: Launch the Qt-based sweep GUI.
- ``main``: Entry point for simple CLI-driven sweeps.

## dpf2.hall_mhd_solver
*Source:* `src/dpf2/hall_mhd_solver.py`

Hall-MHD solver operating on three-dimensional grids.

### Classes
- ``ChemistryModule``: Minimal interface for chemistry plugins.
- ``RadiationModule``: Minimal interface for radiation plugins.
- ``MHDState``: State container for the MHD variables.
- ``HallMHDSolver``: Stub for a 3-D Hall-MHD solver with CT and AMR hooks.

### Functions
- ``spitzer_resistivity``: Return classical Spitzer resistivity in ``Ω·m``.
- ``_dd``: Finite-volume forward difference with periodic boundaries.
- ``_divergence``: Compute a finite-volume divergence that preserves ``∇·(∇×A)=0``.
- ``_curl``: Compute a finite-volume curl consistent with the divergence kernel.
- ``_project_div_free``: Project a magnetic field onto its divergence-free component.
- ``_minmod``: Minmod limiter used for MUSCL reconstruction.
- ``_hll_flux``: Return HLL fluxes for direction ``i``.

## dpf2.hpc
*Source:* `src/dpf2/hpc/__init__.py`

HPC job management utilities.

## dpf2.hpc.manager
*Source:* `src/dpf2/hpc/manager.py`

Simple job manager for scheduling simulations on HPC resources.

### Classes
- ``JobManager``: Dispatch simulation jobs to different backends.

## dpf2.indexing
*Source:* `src/dpf2/indexing.py`

Utilities to build a lightweight index of the DPF2 source tree.

### Classes
- ``SymbolEntry``: Representation of a discovered class or function.
- ``ModuleEntry``: Structured summary of a single Python module.

### Functions
- ``_iter_python_files``: Yield dotted module names and file paths under ``package_root``.
- ``_summarise``: Return a compact single-line summary from ``doc`` if present.
- ``build_code_index``: Scan ``package_root`` and build an index for ``package``.
- ``render_markdown``: Render ``entries`` as a Markdown document.
- ``write_markdown_index``: Write ``entries`` to ``destination`` in Markdown format.
- ``generate_markdown_index``: Convenience wrapper that builds and writes an index in one step.

## dpf2.initial_conditions
*Source:* `src/dpf2/initial_conditions.py`

Initial conditions schema for DPF simulations.

### Classes
- ``ConfigSectionBase``: Common base class providing utility helpers.
- ``BreakdownModel``: _No docstring._
- ``PaschenModel``: _No docstring._
- ``InitialConditions``: _No docstring._

### Functions
- ``to_camel_case``: _No docstring._

## dpf2.io
*Source:* `src/dpf2/io/__init__.py`

_No module docstring available._

## dpf2.io.data_writer
*Source:* `src/dpf2/io/data_writer.py`

Utilities for writing simulation output.

### Classes
- ``DataWriter``: Write simulation data to disk with provenance metadata.

## dpf2.io.datasets
*Source:* `src/dpf2/io/datasets.py`

Utilities for working with reference data sets.

### Functions
- ``load_dataset_manifest``: Load the reference dataset manifest.

## dpf2.io.json_io
*Source:* `src/dpf2/io/json_io.py`

_No module docstring available._

### Functions
- ``export_config``: Write a :class:`~dpf2.dpf_config.DPFConfig` to ``path`` in JSON format.
- ``import_config``: Read ``path`` and return a :class:`~dpf2.dpf_config.DPFConfig`.

## dpf2.io.manifest
*Source:* `src/dpf2/io/manifest.py`

_No module docstring available._

### Functions
- ``_hash_file``: _No docstring._
- ``capture_dataset_metadata``: Compute hashes and attach DOI/version for referenced datasets.
- ``write_hdf5_dataset_manifest``: Embed dataset metadata in an HDF5 ``manifest`` group.

## dpf2.io.restart
*Source:* `src/dpf2/io/restart.py`

_No module docstring available._

### Classes
- ``RestartManager``: Handle writing and reading restart files with provenance metadata.

## dpf2.io.structured
*Source:* `src/dpf2/io/structured.py`

_No module docstring available._

### Classes
- ``StructuredOutputWriter``: Write structured diagnostic output such as JSON or YAML.

## dpf2.ionization
*Source:* `src/dpf2/ionization.py`

_No module docstring available._

### Functions
- ``_k_ion``: Electron impact ionization rate coefficient [m^3/s].
- ``_k_rec``: Radiative recombination rate coefficient [m^3/s].
- ``collisional_radiative_rhs``: Time derivative of electron density for a minimal CR model.
- ``equilibrium_electron_density``: Solve for the equilibrium electron density using bisection.
- ``ionization_energy_sink``: Energy loss rate due to ionization [J/m^3/s].

## dpf2.materials
*Source:* `src/dpf2/materials/__init__.py`

Material-related models and helpers.

## dpf2.materials.library
*Source:* `src/dpf2/materials/library.py`

_No module docstring available._

### Classes
- ``Material``: Static material properties.
- ``MaterialLibrary``: Simple registry of materials with basic properties.

## dpf2.materials.mdm
*Source:* `src/dpf2/materials/mdm.py`

_No module docstring available._

### Classes
- ``MaterialDamageModel``: Minimal material damage model.

## dpf2.materials.models
*Source:* `src/dpf2/materials/models.py`

_No module docstring available._

### Classes
- ``MaterialRef``: Reference to a material and optional coating information.

## dpf2.materials.sputtering
*Source:* `src/dpf2/materials/sputtering.py`

Sputtering models and impurity source terms.

### Classes
- ``Species``: Basic atomic description used by the sputtering helpers.

### Functions
- ``_threshold_energy``: Return the Sigmund threshold energy ``Eth``.
- ``sigmund_yield``: Return the Sigmund sputtering yield for normal incidence.
- ``yamamura_yield``: Return the Yamamura sputtering yield for an arbitrary angle.
- ``impurity_source_terms``: Return impurity source terms for a given incident flux.

## dpf2.materials.state
*Source:* `src/dpf2/materials/state.py`

_No module docstring available._

### Classes
- ``ComponentMaterialState``: Runtime state for a component's material.

## dpf2.materials.tables
*Source:* `src/dpf2/materials/tables.py`

Lookup tables for basic material electrical properties.

### Functions
- ``get_resistivity``: Return the resistivity for ``material``.
- ``get_skin_effect_coeff``: Return the skin effect coefficient for ``material``.

## dpf2.mesh
*Source:* `src/dpf2/mesh/__init__.py`

Mesh utilities and adaptive refinement wrappers.

## dpf2.mesh.amr
*Source:* `src/dpf2/mesh/amr/__init__.py`

_No module docstring available._

### Classes
- ``AMRMesh``: Light‑weight wrapper around a pyAMReX style AMR interface.

## dpf2.mesh.amr.criteria
*Source:* `src/dpf2/mesh/amr/criteria.py`

_No module docstring available._

### Functions
- ``plasma_gradient_refinement``: Return mask of cells where the gradient magnitude exceeds ``threshold``.
- ``debye_length_refinement``: Return mask where the Debye length ``lambda_D`` falls below ``threshold``.
- ``ion_inertial_length_refinement``: Return mask where the ion inertial length ``d_i`` falls below ``threshold``.
- ``pressure_gradient_refinement``: Convenience wrapper applying :func:`plasma_gradient_refinement` to pressure.
- ``current_density_refinement``: Return mask where the current density magnitude exceeds ``threshold``.
- ``current_gradient_refinement``: Refine based on gradients of the current magnitude.
- ``wavefront_refinement``: Tag cells where the change between two fields exceeds ``threshold``.

## dpf2.mesh.boundaries
*Source:* `src/dpf2/mesh/boundaries.py`

Boundary condition helpers for mesh-based field lists.

### Functions
- ``apply_bc``: Apply a basic boundary condition to ``field`` in-place.

## dpf2.mesh.mesh2d
*Source:* `src/dpf2/mesh/mesh2d.py`

2D cylindrical mesh utilities.

### Classes
- ``MeshCell``: Represents a single cell in the 2D mesh.
- ``Mesh2D``: Simple 2D cylindrical mesh.

## dpf2.mesh.mesh3d
*Source:* `src/dpf2/mesh/mesh3d.py`

3D Cartesian mesh utilities.

### Classes
- ``MeshCell3D``: Represents a single cell in the 3D mesh.
- ``Mesh3D``: Simple 3D Cartesian mesh.

## dpf2.mesh.readers
*Source:* `src/dpf2/mesh/readers.py`

_No module docstring available._

### Functions
- ``_ensure_meshio``: _No docstring._
- ``read_stl``: Read an STL surface mesh from ``path``.
- ``read_vtk``: Read an unstructured VTK mesh from ``path``.

## dpf2.metadata
*Source:* `src/dpf2/metadata.py`

Run metadata and provenance tracking for DPF simulations.

### Classes
- ``MLMetadata``: Metadata describing a surrogate or ML model.
- ``MLResult``: Results reported by the surrogate/ML run.
- ``Metadata``: Validated configuration for run metadata.

## dpf2.ml_model_config
*Source:* `src/dpf2/ml_model_config.py`

_No module docstring available._

### Classes
- ``MLModelConfig``: Validated configuration for machine learning models.

## dpf2.neutral
*Source:* `src/dpf2/neutral/__init__.py`

Neutral gas modeling tools.

## dpf2.neutral.dsmc
*Source:* `src/dpf2/neutral/dsmc.py`

Simplified Direct Simulation Monte Carlo (DSMC) neutral gas solver.

### Classes
- ``DSMC``: Very small DSMC solver with a tunable Knudsen number.

### Functions
- ``load_lxcat_table``: Return an ``(N, 2)`` array of energy [eV] and cross section [m^2].
- ``_validate_cross_sections``: Validate tabulated cross sections.

## dpf2.neutron_yield_model
*Source:* `src/dpf2/neutron_yield_model.py`

_No module docstring available._

### Classes
- ``IonBeamEDF``: Protocol providing ion energy distributions by angle.
- ``NeutronYieldModel``: Configuration for neutron yield modeling in DPF simulations.
- ``TabulatedIonEDF``: Simple in-memory implementation of :class:`IonBeamEDF`.

### Functions
- ``compute_directional_spectrum``: Compute energy spectra ``dN/dE`` for multiple detector angles.
- ``load_endf_b_viii``: Create a cross-section interpolator from an ENDF/B-VIII table.
- ``compute_angular_spectra``: Compute beam-target and thermonuclear spectra using ENDF/B-VIII data.
- ``partition_yield``: Integrate rate histories and return channel yields with uncertainties.
- ``directional_counts``: Integrate directional rates and return counts with uncertainties.

## dpf2.optimization
*Source:* `src/dpf2/optimization/__init__.py`

Optimization utilities for parameter inference and control.

### Classes
- ``OptimizationWarning``: Warning raised when queries fall outside the trained domain.

### Functions
- ``enable_optimization_warning_as_error``: Escalate :class:`OptimizationWarning` to an exception.

## dpf2.optimization.bayesian
*Source:* `src/dpf2/optimization/bayesian.py`

Simple Bayesian parameter inference utilities.

### Classes
- ``ParameterEstimate``: Gaussian estimate of a scalar parameter.
- ``BayesianParameterInference``: Perform lightweight Bayesian updates for simulation parameters.

## dpf2.optimization.multi_objective
*Source:* `src/dpf2/optimization/multi_objective.py`

Multi-objective optimization helpers.

### Classes
- ``ConvergenceRecord``: Simple container recording solver progress.

### Functions
- ``random_pareto_search``: Approximate the Pareto front for yield and spot size.
- ``nsga2``: Estimate the Pareto front using a lightweight NSGA-II implementation.

## dpf2.optimization.param_sweep
*Source:* `src/dpf2/optimization/param_sweep.py`

Utilities for evaluating surrogate models over parameter sweeps.

### Classes
- ``OutOfDomainError``: Raised when querying a surrogate outside its trained domain.

### Functions
- ``run_parametric_sweep``: Evaluate surrogate models for a set of parameter values.
- ``compute_sweep_metrics``: Compute simple metrics for surrogate sweep results.
- ``plot_metric_overlay``: Plot yield, pinch time and efficiency against a swept parameter.
- ``plot_yield_vs_S``: Plot yield as a function of the shock parameter ``S``.
- ``plot_yield_pressure_overlay``: Overlay yield vs. pressure curves for multiple sweeps.

## dpf2.parallel_settings
*Source:* `src/dpf2/parallel_settings.py`

_No module docstring available._

### Classes
- ``ParallelSettings``: Parallel execution and hardware configuration.

## dpf2.paschen
*Source:* `src/dpf2/paschen.py`

_No module docstring available._

### Functions
- ``paschen_breakdown_time``: Estimate the breakdown delay using a simple Paschen-like scaling.

## dpf2.physics
*Source:* `src/dpf2/physics/__init__.py`

_No module docstring available._

## dpf2.physics.anomalous_resistivity
*Source:* `src/dpf2/physics/anomalous_resistivity.py`

_No module docstring available._

### Classes
- ``SpectralResistivity``: Estimate anomalous resistivity from lower-hybrid drift spectra.

## dpf2.physics.axial_rundown
*Source:* `src/dpf2/physics/axial_rundown.py`

_No module docstring available._

### Functions
- ``shock_parameter``: Return the dimensionless shock parameter ``S``.
- ``plot_shock_parameter``: Plot ``S`` versus ``time`` and write to ``path``.

## dpf2.physics.em_wave
*Source:* `src/dpf2/physics/em_wave.py`

_No module docstring available._

### Classes
- ``FDTDSolver``: Very small 1-D FDTD Maxwell solver.

## dpf2.physics.energy
*Source:* `src/dpf2/physics/energy.py`

_No module docstring available._

### Classes
- ``EnergyTracker``: Accumulate energy components over time.

## dpf2.physics.eos
*Source:* `src/dpf2/physics/eos.py`

_No module docstring available._

### Classes
- ``TabulatedEOS``: Tabulated EOS with optional opacity data.

### Functions
- ``load_tabulated_eos``: Load EOS/opacity data from ``path``.
- ``load_standard_eos``: Load one of the small built-in EOS tables distributed with the package.

## dpf2.physics.flashover
*Source:* `src/dpf2/physics/flashover.py`

_No module docstring available._

### Classes
- ``FlashoverModel``: Track flashover delays and hold-off with conditioning history.

## dpf2.physics.gv_front
*Source:* `src/dpf2/physics/gv_front.py`

_No module docstring available._

### Classes
- ``GVFront``: Model the r--z current sheath front.

## dpf2.physics.hall_mhd
*Source:* `src/dpf2/physics/hall_mhd.py`

_No module docstring available._

### Classes
- ``HallMHD``: Resistive MHD with Hall term and circuit coupling.

### Functions
- ``nrl_braginskii``: Return simple Braginskii transport coefficients using NRL scalings.
- ``electron_collision_time``: Return the electron-ion collision time ``τ_e`` in seconds.
- ``hall_parameters``: Return ``ω_ce τ_e`` and ``d_i/L`` for gating decisions.
- ``braginskii_coefficients``: Return a subset of Braginskii transport coefficients.
- ``whistler_dispersion``: Return the whistler-wave frequency for wavenumber ``k``.
- ``hall_shock_speed``: Return a characteristic Hall shock speed used in tests.

## dpf2.physics.hooks
*Source:* `src/dpf2/physics/hooks.py`

_No module docstring available._

### Functions
- ``neutral_density_source``: Rate of change of neutral density.
- ``wall_ablation_source``: Mass and energy sources due to wall ablation.

## dpf2.physics.lhdi_resistivity
*Source:* `src/dpf2/physics/lhdi_resistivity.py`

_No module docstring available._

### Functions
- ``_to_array``: Return ``val`` as a floating point array.
- ``compute_effective_eta``: Return an effective resistivity and axial electric-field surge.

## dpf2.physics.lower_hybrid_drift
*Source:* `src/dpf2/physics/lower_hybrid_drift.py`

_No module docstring available._

### Classes
- ``LowerHybridDrift``: Minimal lower-hybrid drift instability model.

### Functions
- ``_to_array``: Return ``val`` as a floating point array.

## dpf2.physics.m0_instability
*Source:* `src/dpf2/physics/m0_instability.py`

_No module docstring available._

### Classes
- ``MZeroInstability``: Very simple ``m=0`` (sausage) instability growth model.

### Functions
- ``_to_array``: Return ``val`` as a floating point array.

## dpf2.physics.material_interactions
*Source:* `src/dpf2/physics/material_interactions/__init__.py`

Material interaction helpers coupling sputtering to impurity tracking.

### Classes
- ``ImpurityState``: Track impurity densities and compute effective charge.

## dpf2.physics.material_interactions.material_properties
*Source:* `src/dpf2/physics/material_interactions/material_properties.py`

Minimal material property database with provenance.

### Classes
- ``MaterialProperties``: Container for simple material properties.

### Functions
- ``get_material_properties``: Return properties for a material by name.

## dpf2.physics.mhd
*Source:* `src/dpf2/physics/mhd.py`

Three-dimensional resistive magnetohydrodynamics utilities.

### Classes
- ``ResistiveMHD``: Conservative 3‑D resistive MHD system with optional physics extensions.

## dpf2.physics.neutral_gas
*Source:* `src/dpf2/physics/neutral_gas/__init__.py`

Neutral gas physics models and utilities.

## dpf2.physics.neutral_gas.fluid
*Source:* `src/dpf2/physics/neutral_gas/fluid.py`

_No module docstring available._

### Classes
- ``NeutralGasFluid``: Zero‑dimensional neutral gas model.

## dpf2.physics.neutral_gas.swarm
*Source:* `src/dpf2/physics/neutral_gas/swarm.py`

_No module docstring available._

### Classes
- ``SwarmParameters``: _No docstring._

### Functions
- ``compute_swarm_parameters``: Compute simple swarm parameters from a cross‑section table.
- ``validate_swarm_parameters``: Validate swarm parameters against reference values.

## dpf2.physics.pic
*Source:* `src/dpf2/physics/pic.py`

_No module docstring available._

### Classes
- ``SimplePIC``: Very small 1‑D PIC model used for tests and examples.
- ``HybridPIC``: Hybrid kinetic solver blending PIC particles with a fluid response.

### Functions
- ``lhdi_resistivity``: Return a simple lower-hybrid drift resistivity estimate.

## dpf2.physics.pic_driver
*Source:* `src/dpf2/physics/pic_driver.py`

_No module docstring available._

### Classes
- ``PicDriver``: Minimal interface for an external PIC driver.
- ``PhysicalPICDriver``: Lightweight physical PIC backend used in tests.

## dpf2.physics.radiation
*Source:* `src/dpf2/physics/radiation.py`

_No module docstring available._

### Classes
- ``RadiationTransport``: High-level driver for multi-group radiation diffusion.

## dpf2.physics.radiation_mhd
*Source:* `src/dpf2/physics/radiation_mhd.py`

_No module docstring available._

### Classes
- ``AMRGrid``: Placeholder structure representing an AMR hierarchy.
- ``RadiationMHDState``: Container holding the solver state variables.
- ``RadiationMHDSolver``: Light‑weight 3‑D radiation‑MHD solver interface.

## dpf2.physics.simple_plasma
*Source:* `src/dpf2/physics/simple_plasma.py`

_No module docstring available._

### Classes
- ``ZeroDPlasma``: Very small plasma model used for tests and examples.

## dpf2.physics.warpx_picmi
*Source:* `src/dpf2/physics/warpx_picmi.py`

_No module docstring available._

### Classes
- ``WarpXPicmiDriver``: Light‑weight PIC driver using the WarpX PICMI interface.

## dpf2.physics_models
*Source:* `src/dpf2/physics_models.py`

_No module docstring available._

### Classes
- ``PhysicsModels``: Validated physics configuration for DPF simulations.

### Functions
- ``to_camel_case``: _No docstring._

## dpf2.pinch_models
*Source:* `src/dpf2/pinch_models.py`

_No module docstring available._

### Classes
- ``PinchResult``: _No docstring._
- ``PinchModelBase``: Base interface for pinch dynamics models.
- ``AnalyticPinchModel``: Very simple analytic model of the DPF pinch.
- ``SemiAnalyticPinchModel``: Cylindrical collapse model with simple pressure balance.
- ``MHDPinchModel``: Pinch model driven by the simplified Hall-MHD solver.
- ``HybridPinchModel``: Hybrid pinch model that swaps regions between PIC and fluid solvers.

## dpf2.plasma_model
*Source:* `src/dpf2/plasma_model.py`

_No module docstring available._

### Functions
- ``advance_plasma_with_circuit``: Advance a plasma solver and return updated coupling terms.
- ``advance_plasmas_with_circuit``: Advance multiple plasma solvers in parallel.

## dpf2.post
*Source:* `src/dpf2/post/__init__.py`

Post-processing utilities for simulation output.

## dpf2.post.visualize
*Source:* `src/dpf2/post/visualize.py`

Visualization utilities using VTK.

### Functions
- ``generate_mesh``: Create a VTK PolyData mesh from ``points`` and ``polys``.
- ``render_video``: Render ``mesh_file`` to ``output`` video using VTK.

## dpf2.postprocessing_settings
*Source:* `src/dpf2/postprocessing_settings.py`

_No module docstring available._

### Classes
- ``PostprocessingSettings``: Configuration for DPF postprocessing and analysis.

## dpf2.prepulse
*Source:* `src/dpf2/prepulse.py`

_No module docstring available._

### Classes
- ``PrePulseResult``: _No docstring._
- ``PrePulseBreakdownModel``: Minimal pre-pulse breakdown model with :math:`J imes B` force.

## dpf2.radiation
*Source:* `src/dpf2/radiation/__init__.py`

_No module docstring available._

### Classes
- ``RadiationBase``: _No docstring._
- ``BremsstrahlungModel``: _No docstring._
- ``MonteCarloRadiation``: _No docstring._

### Functions
- ``create_radiation``: _No docstring._

## dpf2.radiation.metadata
*Source:* `src/dpf2/radiation/metadata.py`

_No module docstring available._

### Classes
- ``DatasetMetadata``: Minimal provenance for a radiation table.

### Functions
- ``_require_metadata_fields``: Validate DOI and version fields and build :class:`DatasetMetadata`.
- ``_load``: Load and validate a metadata JSON file.
- ``load_chianti_metadata``: Load metadata for a CHIANTI table.

## dpf2.radiation.multigroup
*Source:* `src/dpf2/radiation/multigroup.py`

_No module docstring available._

### Classes
- ``MultiGroupDiffusion``: Diffusion model for user-defined radiation energy groups.

## dpf2.radiation.power
*Source:* `src/dpf2/radiation/power.py`

Elementary radiation power models.

### Functions
- ``bremsstrahlung_power``: Return volumetric bremsstrahlung power.
- ``line_radiation_power``: Placeholder line-radiation loss model.

## dpf2.radiation.xray_emission_model
*Source:* `src/dpf2/radiation/xray_emission_model.py`

_No module docstring available._

### Classes
- ``Line``: Atomic line description.

### Functions
- ``cr_line_emission``: Return line emissivity for the requested species.

## dpf2.radiation_transport
*Source:* `src/dpf2/radiation_transport.py`

_No module docstring available._

### Classes
- ``RadiationTransport``: Validated radiation transport configuration.

### Functions
- ``export_mcnp_source``: Write a simple MCNP ``SDEF`` style source definition file.
- ``export_geant4_source``: Write a JSON particle source for Geant4 applications.
- ``ingest_mcnp_tally``: Read a simple tally output from MCNP and apply optional efficiency.
- ``ingest_geant4_tally``: Load a Geant4 tally JSON file and optionally scale by detector efficiency.

## dpf2.rlc_solver
*Source:* `src/dpf2/rlc_solver.py`

Simple solver for a distributed series RLC circuit.

### Classes
- ``DistributedRLCSolution``: Container returned by :func:`solve_distributed_circuit`.

### Functions
- ``solve_distributed_circuit``: Integrate an RLC network using a very small nodal analysis scheme.

## dpf2.scaling_laws
*Source:* `src/dpf2/scaling_laws.py`

_No module docstring available._

### Functions
- ``_load_config``: _No docstring._
- ``compare_to_scaling``: Compare simulation outputs to simple scaling law predictions.
- ``_fit_power_law``: Fit ``y = k * x**m`` returning ``(k, m)``.
- ``sweep_yield_scaling``: Generate ``Y_n`` scaling data for a parameter sweep.

## dpf2.simulation
*Source:* `src/dpf2/simulation/__init__.py`

Legacy simulation components for compatibility.

### Classes
- ``_ModuleProxy``: Lazily import simulation submodules on first attribute access.

## dpf2.simulation.adaptive_mesh_refinement
*Source:* `src/dpf2/simulation/adaptive_mesh_refinement.py`

_No module docstring available._

### Classes
- ``AdvancedWarpXSimulation``: Advanced production simulation using WarpX for plasma research.

## dpf2.simulation.circuit
*Source:* `src/dpf2/simulation/circuit.py`

_No module docstring available._

### Classes
- ``SwitchModel``: A more sophisticated switch model that transitions from high resistance to low resistance based on…
- ``TransmissionLineModel``: A more accurate transmission line model using the telegrapher's equations.
- ``CircuitModel``: High-fidelity RLC circuit dynamically coupled to plasma inductance/resistance.

## dpf2.simulation.collision_model
*Source:* `src/dpf2/simulation/collision_model.py`

Collision model utilities for particle and fluid simulations.

### Classes
- ``CollisionOperator``: Interface for collision models.
- ``CrossSectionData``: 1D tabulated cross-section with simple interpolation.
- ``CollisionProcess``: Base class for individual collisional processes.
- ``BetheBlochStopping``: _No docstring._
- ``ElectronIonCollision``: Electron–ion collisions using Spitzer frequencies.
- ``ElectronNeutralCollision``: Electron–neutral collisions with a constant cross-section.
- ``IonizationProcess``: Ionization of neutrals by electron impact.
- ``RecombinationProcess``: Radiative recombination of ions and electrons.
- ``DDFusion``: Deuterium–Deuterium fusion reactions (simplified).
- ``FokkerPlanckOperator``: Simple Fokker–Planck velocity-space diffusion.
- ``AnisotropyRelaxation``: Relax anisotropy in the velocity distribution.
- ``CollisionalRadiativeNetwork``: Very small collisional–radiative network model.
- ``CollisionModel``: Aggregate collision model for fluid simulations.

### Functions
- ``lnLambda_strong``: _No docstring._
- ``nu_ei_spitzer``: _No docstring._
- ``nu_ee``: _No docstring._
- ``nu_ii``: _No docstring._
- ``nu_en``: Electron-neutral collision frequency.
- ``relax_ei_implicit``: _No docstring._
- ``braginskii_coeffs``: Computes Braginskii transport coefficients.

## dpf2.simulation.config_schema
*Source:* `src/dpf2/simulation/config_schema.py`

_No module docstring available._

### Classes
- ``CircuitConfig``: Configuration for the circuit model.
- ``CollisionConfig``: Configuration for the collision model.
- ``MaterialOpacity``: Per-material opacity definition across radiation groups.
- ``RadiationConfig``: Configuration for the radiation model.
- ``PICConfig``: Configuration for the Particle-in-Cell (PIC) model.
- ``HybridConfig``: Configuration for the hybrid (fluid-PIC) coupling.
- ``DiagnosticsConfig``: Configuration for the diagnostics.
- ``GeometryConfig``: Configuration for the simulation geometry.
- ``FieldManagerConfig``: Configuration for the FieldManager.
- ``AMRConfig``: Configuration for adaptive mesh refinement.
- ``MaterialConfig``: Selection of component materials and initial lifecycle parameters.
- ``SimulationConfig``: Main configuration for the DPF simulation.
- ``ServerConfig``: _No docstring._
- ``SheathConfig``: _No docstring._
- ``TurbulenceConfig``: _No docstring._

## dpf2.simulation.constants
*Source:* `src/dpf2/simulation/constants.py`

Central repository for physical constants.

## dpf2.simulation.diagnostic_plots
*Source:* `src/dpf2/simulation/diagnostic_plots.py`

Plotting utilities for diagnostic outputs.

### Functions
- ``_get_group``: _No docstring._
- ``plot_interferometry``: Plot phase shift versus time.
- ``plot_xray_signal``: Plot X-ray detector signal over time.
- ``plot_neutron_tof``: Plot neutron time-of-flight histogram.

## dpf2.simulation.diagnostics
*Source:* `src/dpf2/simulation/diagnostics.py`

_No module docstring available._

### Classes
- ``Diagnostic``: Base class for all simulation diagnostics.
- ``Interferometry``: _No docstring._
- ``XrayDetector``: _No docstring._
- ``NeutronDetector``: _No docstring._
- ``ModeAnalysis``: _No docstring._
- ``ThomsonScattering``: _No docstring._
- ``Diagnostics``: _No docstring._

### Functions
- ``gaussian_noise``: Return a callable generating Gaussian noise with given ``std``.
- ``poisson_noise``: Return a callable generating Poisson noise with expectation ``lam``.

## dpf2.simulation.dpf_simulation
*Source:* `src/dpf2/simulation/dpf_simulation.py`

DPF Simulation Launcher: Comprehensive multi-physics configuration.

### Classes
- ``_ConfigNamespace``: Lightweight mapping that mimics a Pydantic model.
- ``ConfigurationError``: _No docstring._
- ``InitializationError``: _No docstring._
- ``DPFSimulation``: _No docstring._

### Functions
- ``_namespace_to_plain``: _No docstring._
- ``_as_namespace``: Recursively convert ``data`` into a :class:`_ConfigNamespace`.
- ``parse_arguments``: Parses command-line arguments.
- ``load_config_from_json``: Loads configuration from a JSON file and validates it.
- ``main``: _No docstring._

## dpf2.simulation.dpf_simulator_amrex_backend
*Source:* `src/dpf2/simulation/dpf_simulator_amrex_backend.py`

_No module docstring available._

### Classes
- ``AMReXGridManager``: Minimal grid manager emulating AMReX refinement
- ``DPFSimulatorAMReXBackend``: _No docstring._

### Functions
- ``_parse_cli``: _No docstring._
- ``main``: _No docstring._

## dpf2.simulation.dpf_simulator_full_backend
*Source:* `src/dpf2/simulation/dpf_simulator_full_backend.py`

_No module docstring available._

### Classes
- ``ConfigurationError``: _No docstring._
- ``InitializationError``: _No docstring._
- ``DPFSimulatorBackend``: Unified Dense Plasma Focus simulation orchestrator.

### Functions
- ``_generate_boundary_conditions``: Generates boundary conditions for the fluid solver.
- ``_estimate_total_steps``: Estimate how many steps the simulation will take.

## dpf2.simulation.dpf_simulator_server
*Source:* `src/dpf2/simulation/dpf_simulator_server.py`

_No module docstring available._

### Classes
- ``SimulationInterface``: Thread safe wrapper around :class:`DPFSimulation`.
- ``SimulationManager``: _No docstring._

### Functions
- ``_handle_config_error``: Return a JSON response for configuration errors.
- ``_handle_sim_error``: Return a JSON response for simulation errors.
- ``load_config``: _No docstring._
- ``requires_auth``: _No docstring._
- ``apply_resource_limits``: Apply CPU time and address space limits for the current process.
- ``limit_simulations``: _No docstring._
- ``_validate_sim_parameters``: _No docstring._
- ``start_simulation``: Launch a new simulation. Expects JSON body:
- ``stop_simulation``: Gracefully stop a running simulation by setting its end time to now.
- ``export_results``: Export the HDF5 diagnostics for a completed simulation.
- ``simulation_updates``: WebSocket endpoint streaming summary diagnostics at ~10 Hz.
- ``main``: _No docstring._

## dpf2.simulation.eos
*Source:* `src/dpf2/simulation/eos.py`

_No module docstring available._

### Classes
- ``TabulatedEOS``: Tabulated Equation of State (EOS) for plasma simulations.

### Functions
- ``parse_mixture_fractions``: Parse mixture fraction definitions into a normalised dictionary.

## dpf2.simulation.eos_selector
*Source:* `src/dpf2/simulation/eos_selector.py`

Selects and initializes the appropriate Equation of State (EOS) model.

### Functions
- ``select_eos``: Selects and returns an initialized Equation of State object.

## dpf2.simulation.fluid_solver_high_order
*Source:* `src/dpf2/simulation/fluid_solver_high_order.py`

High-order fluid solver for magnetized plasmas.

### Classes
- ``FluidSolverHighOrder``: Prototype high-order MHD solver.

### Functions
- ``divergence``: _No docstring._
- ``curl``: _No docstring._
- ``weno5_reconstruct``: _No docstring._
- ``weno5_reconstruct_3d``: Reconstructs flux at i+1/2 for a 3D array u of shape (nx, ny, nz) using WENO5 scheme with improved…

## dpf2.simulation.gpu_diagnostics
*Source:* `src/dpf2/simulation/gpu_diagnostics.py`

GPU-accelerated diagnostic helpers.

### Classes
- ``GPUKineticEnergyDiagnostic``: Minimal diagnostic recording kinetic energy for a species.

### Functions
- ``kinetic_energy``: Return total kinetic energy using CUDA when available.

## dpf2.simulation.hybrid_controller
*Source:* `src/dpf2/simulation/hybrid_controller.py`

Hybrid controller coordinating fluid and PIC solvers.

### Classes
- ``HybridController``: Orchestrates hybrid fluid–PIC simulations, managing coupling and transitions.
- ``AsyncHybridController``: Asynchronous hybrid controller running fluid and PIC solvers concurrently.

### Functions
- ``compute_transition_mask``: _No docstring._
- ``bump_weight``: _No docstring._

## dpf2.simulation.hybrid_pic_solver
*Source:* `src/dpf2/simulation/hybrid_pic_solver.py`

Hybrid 2D/3D Particle-in-Cell module.

### Classes
- ``_FluidSolver``: Minimal protocol for fluid solvers used by :class:`HybridPICSolver`.
- ``_ParticleSolver``: Minimal protocol for particle solvers used by :class:`HybridPICSolver`.
- ``HybridPICSolver``: Hybrid fluid/PIC plasma solver.

## dpf2.simulation.load_balance_metrics
*Source:* `src/dpf2/simulation/load_balance_metrics.py`

_No module docstring available._

### Classes
- ``LoadBalanceMetrics``: Compute simple cell and particle balance metrics across MPI ranks.

## dpf2.simulation.models
*Source:* `src/dpf2/simulation/models.py`

_No module docstring available._

### Classes
- ``PhysicsModule``: Abstract base class for all physics modules in the DPF simulation.

## dpf2.simulation.module_registry
*Source:* `src/dpf2/simulation/module_registry.py`

_No module docstring available._

### Classes
- ``ModuleDependencyError``: Raised when a module's dependency is not met.
- ``ModuleInitializationError``: Raised when a module fails to initialize.
- ``ModuleConfigurationError``: Raised when a module's configuration is invalid.
- ``ModuleRegistry``: A robust module registry for managing physics modules in the DPF simulation.

## dpf2.simulation.openpmd_io
*Source:* `src/dpf2/simulation/openpmd_io.py`

_No module docstring available._

### Classes
- ``OpenPMDWriter``: Minimal openPMD-compliant writer for field and particle data.

## dpf2.simulation.pic_solver
*Source:* `src/dpf2/simulation/pic_solver.py`

Particle-in-Cell (PIC) solver implementing classical electromagnetic dynamics,

### Classes
- ``MZeroInstability``: Simple exponential m=0 instability growth model.
- ``AnomalousResistivity``: Mechanism-based anomalous resistivity model.
- ``LHDIResistivity``: Lower-hybrid drift instability based resistivity model.
- ``PICSolver``: Classical PIC solver with optional WarpX coupling.

## dpf2.simulation.radiation_model
*Source:* `src/dpf2/simulation/radiation_model.py`

Radiation transport model used by the DPF2 simulations.

### Classes
- ``Photon``: Lightweight Monte Carlo photon particle.
- ``RadiationModel``: Multi-group radiation transport model with simplified coupling.

### Functions
- ``klein_nishina_cross_section``: _No docstring._
- ``pair_production_cross_section``: Very approximate Bethe-Heitler like cross section.

## dpf2.simulation.setup
*Source:* `src/dpf2/simulation/setup.py`

_No module docstring available._

## dpf2.simulation.sheath_model
*Source:* `src/dpf2/simulation/sheath_model.py`

_No module docstring available._

### Classes
- ``BohmSheath``: Apply simple Bohm sheath boundary conditions.
- ``PlasmaSheathFormation``: A high-fidelity model for plasma sheath formation, including:

## dpf2.simulation.solver_selector
*Source:* `src/dpf2/simulation/solver_selector.py`

_No module docstring available._

### Functions
- ``select_solver``: Selects and returns a solver based on the specified backend.
- ``initialize_solver``: Initializes the selected solver with the given configuration.

## dpf2.simulation.turbulence_model
*Source:* `src/dpf2/simulation/turbulence_model.py`

_No module docstring available._

### Classes
- ``TurbulenceModel``: RANS k-epsilon turbulence model with optional wall functions.

### Functions
- ``compute_strain_rate_tensor``: Computes the strain rate tensor using Numba for acceleration.
- ``compute_laplacian``: Computes the Laplacian of an array using Numba for acceleration.

## dpf2.simulation.utils
*Source:* `src/dpf2/simulation/utils.py`

_No module docstring available._

### Classes
- ``FieldManager``: Manages electromagnetic fields (E and B) and related operations.
- ``SimulationState``: Represents the state of the simulation at a given time.

## dpf2.simulation.warp_piclibrary
*Source:* `src/dpf2/simulation/warp_piclibrary.py`

Placeholder library for handling PIC collisions specifically within WarpX.

### Classes
- ``PICCollisionHandler``: High level interface for Monte Carlo collisions in WarpX.

## dpf2.simulation.warpx_wrapper
*Source:* `src/dpf2/simulation/warpx_wrapper.py`

Lightweight WarpX wrapper.

### Classes
- ``Field``: Thin wrapper for field and particle arrays, matching fluid solver style.
- ``WarpXWrapper``: Simplified interface to WarpX for grid setup and particle integration.

### Functions
- ``_resample_array``: Resample ``arr`` to ``new_shape`` using a simple separable scheme.
- ``_compute_kinetic_energy``: Compute kinetic energy for ``vel`` with optional relativistic correction.

## dpf2.simulation_engine
*Source:* `src/dpf2/simulation_engine.py`

_No module docstring available._

### Classes
- ``SimulationResults``: _No docstring._
- ``EnsembleResults``: Statistics aggregated from multiple realizations.
- ``SimulationEngine``: Execute a minimal Dense Plasma Focus simulation.

### Functions
- ``_capacitor_energy``: Return stored energy in a capacitor using Numba for speed.

## dpf2.simulation_settings
*Source:* `src/dpf2/simulation_settings.py`

_No module docstring available._

### Classes
- ``SimulationSettings``: Simulation control parameters.

## dpf2.solvers
*Source:* `src/dpf2/solvers/__init__.py`

Plasma solver implementations.

## dpf2.solvers.axisymmetric_hlld
*Source:* `src/dpf2/solvers/axisymmetric_hlld.py`

Axisymmetric HLLD MHD solver with constrained transport.

### Classes
- ``AxisymmetricHLLD``: Minimal 2-D axisymmetric MHD solver.

## dpf2.solvers.muscl_hancock
*Source:* `src/dpf2/solvers/muscl_hancock.py`

MUSCL-Hancock scheme for MHD.

### Classes
- ``MUSCLHancock``: Second-order MUSCL-Hancock scheme with HLL Riemann solver.

## dpf2.synthetic_diagnostics
*Source:* `src/dpf2/synthetic_diagnostics/__init__.py`

Synthetic diagnostics package with compatibility shim.

## dpf2.synthetic_diagnostics
*Source:* `src/dpf2/synthetic_diagnostics.py`

Legacy synthetic diagnostics module.

## dpf2.synthetic_diagnostics.core
*Source:* `src/dpf2/synthetic_diagnostics/core.py`

_No module docstring available._

### Classes
- ``AngularDistribution``: Simple histogram of particle counts versus angle.
- ``SyntheticInstrument``: Per-instrument overrides for synthetic diagnostics.
- ``SyntheticDiagnostics``: Synthetic diagnostic modeling configuration.

### Functions
- ``generate_tof_spectrum``: Create a synthetic time-of-flight spectrum from neutron energies.
- ``beam_target_angular_spectrum``: Convenience wrapper exposing :func:`dd_beam_target_angular_spectrum`.
- ``directional_yields``: Return forward, radial and backward yield components.
- ``flashover_delay_stats``: Return simple statistics for flashover delays.
- ``flashover_jitter_stats``: Return jitter statistics for flashover hold-off voltages.
- ``export_directional_yields``: Write directional yield ``totals`` to ``path`` in JSON format.
- ``synthetic_tof_trace``: Generate a synthetic time-of-flight trace from a history of states.
- ``autocorrelated_tof_iv_report``: Export an auto-correlated ToF versus I–V spike report.
- ``anisotropy_report``: Compute yield ratios and correlate ToF counts with ``I*V`` spikes.
- ``_cr39_image``: Return a simple Gaussian spot image scaled by peak current.
- ``_rcf_image``: Return a ring-shaped image scaled by peak voltage.
- ``_faraday_iedf``: Generate a synthetic ion energy distribution function.
- ``_faraday_eedf``: Generate a synthetic electron energy distribution function.
- ``run_diagnostic_calculations``: Compute enabled synthetic diagnostic signals.
- ``_export_csv``: _No docstring._
- ``_export_hdf5``: _No docstring._
- ``export_diagnostic_data``: Write diagnostic ``data`` to ``output_dir`` according to ``cfg``.
- ``_sd_model_validate``: _No docstring._

## dpf2.synthetic_diagnostics.modes
*Source:* `src/dpf2/synthetic_diagnostics/modes/__init__.py`

Synthetic diagnostics for modal analysis.

### Functions
- ``plot_growth_rates``: Plot modal growth rates and return the path to the figure.
- ``write_growth_rates``: Compute growth rates and write them to ``outdir``.

## dpf2.ui
*Source:* `src/dpf2/ui/__init__.py`

User interface panels.

## dpf2.ui.verification_panel
*Source:* `src/dpf2/ui/verification_panel.py`

Simple CLI/CLI helpers for running verification problems.

### Classes
- ``VerificationPanelUI``: Launch verification tests and report pass/fail status.

### Functions
- ``_main``: _No docstring._

## dpf2.units_settings
*Source:* `src/dpf2/units_settings.py`

_No module docstring available._

### Classes
- ``UnitsSettings``: Unit system configuration for DPF simulations.

## dpf2.uq
*Source:* `src/dpf2/uq/__init__.py`

Uncertainty quantification utilities.

## dpf2.uq.analysis
*Source:* `src/dpf2/uq/analysis.py`

Post-processing helpers for uncertainty quantification.

### Functions
- ``_to_matrix``: Convert ``samples`` to a list-of-lists without requiring ``numpy``.
- ``sobol_indices``: Estimate first-order Sobol indices from ``samples`` and ``values``.
- ``uncertainty_band``: Compute mean, standard deviation and a central interval for ``values``.
- ``propagate_yield_pinch``: Propagate parameter samples to yield and pinch-time uncertainties.

## dpf2.uq.calibration
*Source:* `src/dpf2/uq/calibration.py`

Calibration routines for inferring model parameters from diagnostics.

### Functions
- ``bayesian_calibration``: Infer model parameters from experimental ``data`` using MCMC.
- ``nested_calibration``: Calibrate model parameters using a basic nested sampler.
- ``emcee_calibrate``: Infer parameters using the :mod:`emcee` ensemble sampler.
- ``dynesty_calibrate``: Infer parameters using :mod:`dynesty` nested sampling.
- ``emcee_calibrate_mass_current``: Estimate mass and current scaling factors using :mod:`emcee`.
- ``dynesty_calibrate_mass_current``: Estimate mass and current factors using :mod:`dynesty` nested sampling.
- ``emcee_calibrate_waveform``: Infer mass/current scaling factors from a current waveform using MCMC.
- ``dynesty_calibrate_waveform``: Calibrate waveform scaling factors using :mod:`dynesty`.
- ``calibrate_waveform``: Calibrate waveform scaling factors using the chosen sampler.

## dpf2.uq.inference
*Source:* `src/dpf2/uq/inference.py`

Advanced parameter inference routines leveraging external samplers.

### Functions
- ``emcee_infer``: Infer parameters using the :mod:`emcee` ensemble sampler.
- ``dynesty_infer``: Infer parameters using the :mod:`dynesty` nested sampler.
- ``emcee_infer_waveform``: Infer mass and current scaling from waveform data using MCMC.
- ``dynesty_infer_waveform``: Infer waveform scaling factors using :mod:`dynesty` nested sampling.

## dpf2.uq.samplers
*Source:* `src/dpf2/uq/samplers.py`

Sampling schemes for uncertainty quantification.

### Functions
- ``latin_hypercube``: Generate samples using Latin hypercube sampling.
- ``sobol_sample``: Generate Sobol sequence samples.

## dpf2.uq.sampling
*Source:* `src/dpf2/uq/sampling.py`

Deprecated module. Use :mod:`dpf2.uq.samplers` instead.

## dpf2.utils
*Source:* `src/dpf2/utils/__init__.py`

Utility helpers for dpf2.

## dpf2.utils.pydantic_compat
*Source:* `src/dpf2/utils/pydantic_compat.py`

Compatibility helpers bridging Pydantic v1 and v2 APIs.

### Functions
- ``model_validator``: Polyfill for the :func:`pydantic.v2` ``model_validator`` decorator.

## dpf2.validation
*Source:* `src/dpf2/validation/__init__.py`

Validation helpers and numerical regression panels.

## dpf2.validation.numerics_panel
*Source:* `src/dpf2/validation/numerics_panel.py`

Lightweight numerical regression tests for standard MHD problems.

### Classes
- ``NumericsPanel``: Run small validation problems and gather diagnostic metrics.

### Functions
- ``compute_metrics``: Return basic diagnostic metrics for a magnetic field.
- ``_divergence``: Compute a simple discrete divergence.
- ``_spectrum_1d``: Naive discrete Fourier spectrum along the flattened array.
- ``_flatten``: _No docstring._

## dpf2.validation_suite
*Source:* `src/dpf2/validation_suite.py`

_No module docstring available._

### Classes
- ``ValidationSuite``: Validated configuration for benchmarking simulation outputs.

### Functions
- ``_load_profile_csv``: Load a time-series profile from a two-column CSV file.
- ``load_benchmark_dataset``: Return benchmark GV timing and L(t)/I(t) profiles from ``dataset_dir``.
- ``compare_gv_timing``: Absolute difference between simulated and reference GV timing.
- ``compare_profiles``: RMSE between a simulated profile and reference profile.
- ``compute_error_metrics``: Compute error metrics for GV timing and L(t)/I(t) profiles.
- ``load_pinch_dataset``: Load benchmark pinch traces from ``dataset_dir``.
- ``_rmse``: Compute RMSE between two time series.
- ``_peak_time``: _No docstring._
- ``_peak_timing_error``: Absolute difference in peak times between two profiles.
- ``_integrated_energy``: _No docstring._
- ``_energy_balance_error``: Difference in discharge energy between simulation and reference.
- ``compute_pinch_error_metrics``: Compute RMSE, peak timing and energy balance errors for pinch traces.
- ``load_validation_dataset``: Load waveform and yield data for ``device`` from packaged CSV files.
- ``resample_profile``: Resample ``profile`` onto ``new_t`` using ``method``.
- ``score_simulation``: Compute per-observable and aggregate validation scores.
- ``evaluate_benchmark``: Compare simulation outputs against expected benchmark values.

## dpf2.verification
*Source:* `src/dpf2/verification/__init__.py`

Verification helpers and regression panels.

## dpf2.verification.panel
*Source:* `src/dpf2/verification/panel.py`

Verification panel executing small MHD benchmark problems.

### Classes
- ``VerificationPanel``: Run verification problems and gather diagnostic metrics.

### Functions
- ``_divergence``: Compute a simple discrete divergence.
- ``_spectrum_1d``: Naive discrete Fourier spectrum along the flattened array.
- ``_flatten``: _No docstring._
- ``compute_metrics``: Return basic diagnostic metrics for a magnetic field.
- ``_observed_orders``: Compute observed order of accuracy from error data.

## dpf2.version
*Source:* `src/dpf2/version.py`

_No module docstring available._

## dpf2.visualization
*Source:* `src/dpf2/visualization/__init__.py`

Visualization helpers for DPF2.

## dpf2.visualization.sheath
*Source:* `src/dpf2/visualization/sheath.py`

Sheath visualization utilities.

### Classes
- ``SheathField``: Container for the vector field used in the animation.

### Functions
- ``_sheath_field``: Generate a toy sheath vector field.
- ``sheath_velocity_field``: Return the synthetic sheath velocity field.
- ``jxb_field``: Compute the J×B drift field for the toy sheath.
- ``animate_sheath``: Animate a simple sheath evolution.
- ``animate_discharge_phases``: Animate the canonical DPF discharge phases.

## dpf2.visualization.widgets
*Source:* `src/dpf2/visualization/widgets.py`

Interactive widgets for visualization.

### Functions
- ``sheath_widget``: Return a widget controlling :func:`animate_sheath`.

## dpf2.warpx_settings
*Source:* `src/dpf2/warpx_settings.py`

_No module docstring available._

### Classes
- ``AdaptiveTimeStep``: Configuration for adaptive time stepping.
- ``SpeciesEntry``: _No docstring._
- ``WarpXSettings``: Validated WarpX PIC solver configuration.

## dpf2.web
*Source:* `src/dpf2/web/__init__.py`

Lightweight web dashboard for running simulations.

## dpf2.web.app
*Source:* `src/dpf2/web/app.py`

Minimal Flask application for running DPF simulations.

### Functions
- ``_update_config_from_form``: Populate a :class:`DPFConfig` instance from form fields.
- ``_parse_sweep_values``: _No docstring._
- ``create_app``: _No docstring._

## dpf2.web.lab_mode_api
*Source:* `src/dpf2/web/lab_mode_api.py`

Utilities for managing lab-mode manifests via a simple API.

### Functions
- ``_config_hash``: Return a stable hash of ``config`` for provenance purposes.
- ``generate_manifest``: Generate a single manifest describing a run.
- ``generate_manifest_bundle``: Generate manifests for a batch of configurations.
- ``export_manifest_bundle``: Write a zip bundle containing inputs and manifests for each run.

## dpf2.web.plots
*Source:* `src/dpf2/web/plots.py`

_No module docstring available._

### Functions
- ``plot_current_voltage``: Overlay current and voltage versus time on shared axes.
- ``plot_vector_field_overlay``: Create a simple vector field overlay with an optional background image.
- ``plot_plasma_inductance_comparison``: Plot field-derived and circuit-derived plasma inductance.

## dpf2.web.sandbox
*Source:* `src/dpf2/web/sandbox.py`

_No module docstring available._

### Functions
- ``main``: Generate simple current/voltage and vector-field plots for students.

## dpf2.xray_emission_model
*Source:* `src/dpf2/xray_emission_model.py`

X-ray emission configuration model for DPF simulations.

### Classes
- ``XrayEmissionModel``: Validated X-ray emission configuration.
