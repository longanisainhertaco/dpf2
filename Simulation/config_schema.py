# config_schema.py
from pydantic import BaseModel, validator, Field, root_validator
from typing import List, Dict, Optional, Any, Tuple
import numpy as np

class CircuitConfig(BaseModel):
    """Configuration for the circuit model."""
    C: float = Field(..., gt=0, description="Capacitance [F]")
    V0: float = Field(..., gt=0, description="Initial voltage [V]")
    L0: float = Field(..., gt=0, description="External inductance [H]")
    R0: float = Field(..., ge=0, description="External resistance [Ω]")
    anode_radius: float = Field(..., gt=0, description="Anode radius [m]")
    cathode_radius: float = Field(..., gt=0, description="Cathode radius [m]")
    ESR: float = Field(0.0, ge=0, description="Equivalent series resistance [Ω]")
    ESL: float = Field(0.0, ge=0, description="Equivalent series inductance [H]")
    switch_resistance: float = Field(0.0, ge=0, description="Resistance of the switch [Ω]")
    switch_inductance: float = Field(0.0, ge=0, description="Inductance of the switch [H]")
    transmission_line_impedance: float = Field(50.0, gt=0, description="Impedance of the transmission line [Ω]")
    transmission_line_length: float = Field(1.0, gt=0, description="Length of the transmission line [m]")

    @validator('anode_radius')
    def check_anode_radius(cls, v, values):
        if 'cathode_radius' in values and v >= values['cathode_radius']:
            raise ValueError('anode_radius must be smaller than cathode_radius')
        return v

class CollisionConfig(BaseModel):
    """Configuration for the collision model."""
    lnΛ: float = Field(10.0, gt=0, description="Coulomb logarithm")
    sigma_in: float = Field(1e-19, gt=0, description="Ionization cross-section [m^2]")
    electron_neutral_collision_enabled: bool = Field(False, description="Enable electron-neutral collisions")
    electron_neutral_cross_section: float = Field(1e-19, gt=0, description="Electron-neutral cross-section [m^2]")
    charge_exchange_enabled: bool = Field(False, description="Enable charge exchange collisions")
    charge_exchange_cross_section: float = Field(1e-19, gt=0, description="Charge exchange cross-section [m^2]")

class RadiationConfig(BaseModel):
    """Configuration for the radiation model."""
    use_line_radiation: bool = Field(False, description="Enable line radiation")
    line_cooling_curve: Optional[str] = Field(None, description="Path to the HDF5 file containing the line cooling curve")
    telemetry_port: int = Field(..., description="Port for telemetry streaming")
    photon_params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for photon Monte Carlo")
    opacity_model: str = Field("constant", description="Model for opacity calculation")
    opacity_params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for opacity model")
    group_opacities: List[float] = Field(..., description="Opacities for each radiation group")
    adios_engine: str = Field("BP4", description="ADIOS2 engine")
    adios_parameters: Dict[str, Any] = Field(default_factory=dict, description="ADIOS2 parameters")
    adios_file: str = Field("radiation.bp", description="ADIOS2 output file")
    gaunt_factor: float = Field(1.2, description="Gaunt factor for Bremsstrahlung")
    num_photons_per_cell: int = Field(10, description="Number of photons to emit per cell per time step")
    bremsstrahlung_enabled: bool = Field(True, description="Enable Bremsstrahlung radiation")
    synchrotron_enabled: bool = Field(True, description="Enable Synchrotron radiation")
    compton_scattering_enabled: bool = Field(False, description="Enable Compton scattering")
    dynamic_opacities_enabled: bool = Field(False, description="Enable dynamic opacities")
    photon_monte_carlo_enabled: bool = Field(False, description="Enable photon Monte Carlo")
    eddington_tensor_closure_enabled: bool = Field(False, description="Enable Eddington tensor closure")
    dynamic_line_radiation_enabled: bool = Field(False, description="Enable dynamic line radiation")
    photon_absortion_enabled: bool = Field(False, description="Enable photon absortion")
    photon_emission_enabled: bool = Field(False, description="Enable photon emission")
    photon_transport_enabled: bool = Field(False, description="Enable photon transport")
    # Add other parameters as needed

class PICConfig(BaseModel):
    """Configuration for the Particle-in-Cell (PIC) model."""
    num_particles: int = Field(..., gt=0, description="Number of particles")
    use_hybrid_electrons: bool = True
    particle_shape: int = Field(2, description="Order of the particle shape function")
    max_grid_size: int = Field(64, description="Maximum grid size for adaptive mesh refinement")
    max_level: int = Field(0, description="Maximum AMR level")
    initial_grid_level: int = Field(0, description="Initial AMR level")
    time_step_scaling_factor: float = Field(1.0, description="Scaling factor for the time step")
    particle_relocation_method: str = Field("momentum_conserving", description="Method for particle relocation")
    field_gathering_order: int = Field(2, description="Order of the field gathering scheme")
    particle_pusher_algo: str = Field("boris", description="Algorithm for pushing particles")
    current_deposition_algo: str = Field("esirkepov", description="Algorithm for current deposition")
    field_smoothing_iterations: int = Field(0, description="Number of iterations for field smoothing")
    field_smoothing_strength: float = Field(0.0, description="Strength of field smoothing")
    particle_injection_style: str = Field("full_box", description="Style of particle injection")
    particle_injection_temperature: float = Field(1.0, description="Temperature of injected particles")
    particle_injection_density: float = Field(1.0, description="Density of injected particles")
    particle_injection_velocity: List[float] = Field([0.0, 0.0, 0.0], description="Velocity of injected particles")
    particle_injection_position: List[float] = Field([0.0, 0.0, 0.0], description="Position of injected particles")
    # Add other parameters as needed

class HybridConfig(BaseModel):
    """Configuration for the hybrid (fluid-PIC) coupling."""
    switch_radius: float = Field(..., gt=0, description="Switch radius")
    coupling_buffer: int = Field(..., gt=0, description="Coupling buffer")
    max_subcycles: int = Field(..., gt=0, description="Maximum number of PIC subcycles")
    coupling_tolerance: float = Field(1e-6, description="Tolerance for the coupling iteration")
    target_volume_fraction: float = Field(0.5, description="Target volume fraction for the PIC region")
    transition_criteria_gradient_threshold: float = Field(0.1, description="Gradient threshold for transition criteria")
    transition_criteria_knudsen_number_threshold: float = Field(0.1, description="Knudsen number threshold for transition criteria")
    transition_criteria_hall_parameter_threshold: float = Field(0.1, description="Hall parameter threshold for transition criteria")
    transition_criteria_non_maxwellianity_factor: float = Field(1.0, description="Non-Maxwellianity factor for transition criteria")
    # Add other parameters as needed

class DiagnosticsConfig(BaseModel):
    """Configuration for the diagnostics."""
    hdf5_filename: str = "diagnostics.h5"
    field_diagnostic_interval: int = Field(10, description="Interval for field diagnostics")
    particle_diagnostic_interval: int = Field(10, description="Interval for particle diagnostics")
    energy_diagnostic_interval: int = Field(10, description="Interval for energy diagnostics")
    mode_analysis_enabled: bool = Field(False, description="Enable mode analysis diagnostic")
    thomson_scattering_enabled: bool = Field(False, description="Enable Thomson scattering diagnostic")
    # Add other parameters as needed

class GeometryConfig(BaseModel):
    """Configuration for the simulation geometry."""
    enable_amr: bool = False
    amr_criteria: List[str] = []
    eb_geometry_file: Optional[str] = None
    electrode_positions: Optional[List[float]] = None
    boundary_conditions: Optional[Dict[str, str]] = None
    domain_length: List[float] = Field(..., description="Length of the simulation domain in each dimension")
    domain_width: List[float] = Field(..., description="Width of the simulation domain in each dimension")
    domain_height: List[float] = Field(..., description="Height of the simulation domain in each dimension")
    geometry_type: str = Field("cartesian", description="Type of geometry to use")
    # Add other parameters as needed

class FieldManagerConfig(BaseModel):
    """Configuration for the FieldManager."""
    boundary_conditions: Dict[str, str] = Field(
        default_factory=lambda: {
            'x_lo': 'periodic', 'x_hi': 'periodic',
            'y_lo': 'periodic', 'y_hi': 'periodic',
            'z_lo': 'periodic', 'z_hi': 'periodic'
        },
        description="Boundary conditions for the electromagnetic fields."
    )
    field_smoothing_enabled: bool = Field(False, description="Enable field smoothing")
    field_smoothing_iterations: int = Field(0, description="Number of iterations for field smoothing")
    field_smoothing_strength: float = Field(0.0, description="Strength of field smoothing")
    # Add other parameters as needed

class SimulationConfig(BaseModel):
    """Main configuration for the DPF simulation."""
    grid_shape: List[int] = Field(..., min_items=3, max_items=3, description="Number of grid points (nx, ny, nz)")
    dx: float = Field(..., gt=0, description="Grid spacing in x-direction [m]")
    dy: float = Field(..., gt=0, description="Grid spacing in y-direction [m]")
    dz: float = Field(..., gt=0, description="Grid spacing in z-direction [m]")
    domain_lo: Tuple[float, float, float] = Field(..., description="Lower corner of the domain (x0, y0, z0).")
    sim_time: float = Field(..., gt=0, description="Simulation time [s]")
    dt_init: Optional[float] = Field(None, gt=0, description="Initial time step [s]")
    circuit: CircuitConfig
    collision: Optional[CollisionConfig] = None
    radiation: Optional[RadiationConfig] = None
    pic: Optional[PICConfig] = None
    hybrid: Optional[HybridConfig] = None
    diagnostics: Optional[DiagnosticsConfig] = None
    geometry: Optional[GeometryConfig] = None
    field_manager: FieldManagerConfig = Field(default_factory=FieldManagerConfig, description="Configuration for the FieldManager.")
    provenance: Optional[Dict[str, Any]] = None
    telemetry: Optional[Dict[str, Any]] = None
    io: Optional[Dict[str, Any]] = None
    simulation_name: str = Field("DPF Simulation", description="Name of the simulation")
    simulation_description: str = Field("A Dense Plasma Focus simulation", description="Description of the simulation")
    # Add other parameters as needed

    @validator('grid_shape')
    def check_grid_shape(cls, v):
        if any(n <= 0 for n in v):
            raise ValueError('grid_shape must contain positive integers')
        return v

    @root_validator
    def check_hybrid_requirements(cls, values):
        if values.get('hybrid') and not values.get('pic'):
            raise ValueError('Hybrid coupling requires PIC module')
        return values

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(5000, gt=0, lt=65536, description="Port for the server to listen on")
    admin_username: str = "admin"
    admin_password_hash: str = Field(..., description="Hashed password for the admin user")
    max_simultaneous_simulations: int = Field(5, gt=0, description="Maximum number of simultaneous simulations")
    telemetry_interval: float = Field(0.1, gt=0, description="Interval for sending telemetry data [s]")
    data_directory: str = Field("data", description="Directory to store simulation data")
    # Add other parameters as needed

class SheathConfig(BaseModel):
    ion_density: float = Field(..., description="Ion density at the sheath edge")
    electron_density: float = Field(..., description="Electron density at the sheath edge")
    sheath_voltage: float = Field(..., description="Sheath voltage")
    ion_temperature: float = Field(..., description="Ion temperature")
    electron_temperature: float = Field(..., description="Electron temperature")
    ion_mass: float = Field(..., description="Ion mass")
    dx: float = Field(..., description="Grid spacing")
    max_sheath_thickness: float = Field(1e-3, description="Maximum sheath thickness")
    num_grid_points: int = Field(200, description="Number of grid points in the sheath")
    plasma_edge_potential: float = Field(0.0, description="Potential at the plasma edge")
    opacity_model: str = Field("constant", description="Model for opacity calculation")
    opacity_params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for opacity model")
    # Add other parameters as needed

class TurbulenceConfig(BaseModel):
    C_mu: float = Field(0.09, gt=0, description="Model constant C_mu")
    C_epsilon1: float = Field(1.44, gt=0, description="Model constant C_epsilon1")
    C_epsilon2: float = Field(1.92, gt=0, description="Model constant C_epsilon2")
    sigma_k: float = Field(1.0, gt=0, description="Turbulent Prandtl number for k")
    sigma_epsilon: float = Field(1.3, gt=0, description="Turbulent Prandtl number for epsilon")
    initial_k: float = Field(1e-6, gt=0, description="Initial turbulent kinetic energy")
    initial_epsilon: float = Field(1e-8, gt=0, description="Initial dissipation rate")
    wall_function_type: str = Field("none", description="Type of wall function to use (none, log_law, power_law)")
    compressibility_correction: bool = Field(False, description="Enable compressibility correction")
    wall_function_kappa: float = Field(0.41, description="Von Karman constant for wall functions")
    wall_function_E: float = Field(9.8, description="Empirical constant for wall functions")
    compressibility_alpha: float = Field(1.0, description="Compressibility correction coefficient")
    compressibility_beta: float = Field(1.0, description="Compressibility correction coefficient")
    # Add other parameters as needed
