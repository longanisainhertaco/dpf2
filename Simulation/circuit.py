import numpy as np
import sympy as sp
import logging
from scipy.special import erf, erfi
from scipy.constants import mu_0, c, epsilon0
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from typing import Dict, Any, Optional
from models import SimulationState
from utils import FieldManager

# Physical constants (moved to the top)
from constants import e, me, epsilon0

# Configure logging
logger = logging.getLogger(__name__)

# === Symbolic ODE Derivation for RLC ===
# States: Q (charge on C), I (circuit current)
# Equations:
#   dQ/dt = -I
#   dI/dt = [ -R_tot*I - Q/C ] / L_tot
Q_sym, I_sym, R_sym, L_sym, C_sym, Lp_sym, Rp_sym, Ls_sym, Cs_sym = sp.symbols('Q I R L C Lp Rp Ls Cs')
dQ_dt_expr = -I_sym
dI_dt_expr = (-R_sym * I_sym - Q_sym / C_sym) / (L_sym + Ls_sym)
# Lambdify for numeric evaluation
_rhs = sp.lambdify((Q_sym, I_sym, R_sym, L_sym, C_sym, Lp_sym, Rp_sym, Ls_sym, Cs_sym),
                   (dQ_dt_expr, dI_dt_expr), 'numpy')
# Clean up
del Q_sym, I_sym, R_sym, L_sym, C_sym, Lp_sym, Rp_sym, Ls_sym, Cs_sym, dQ_dt_expr, dI_dt_expr

class SwitchModel:
    """
    A more sophisticated switch model that transitions from high resistance to low resistance based on voltage and current.
    """
    def __init__(self, initial_resistance: float = 1e6, final_resistance: float = 1e-3,
                 transition_voltage: float = 1e3, transition_current: float = 1e3,
                 transition_time: float = 1e-9, voltage_slew_rate: float = 1e12, current_slew_rate: float = 1e12):
        """
        Initializes the SwitchModel.

        Args:
            initial_resistance: Initial resistance of the switch [Ω].
            final_resistance: Final resistance of the switch [Ω].
            transition_voltage: Voltage at which the switch starts to close [V].
            transition_current: Current at which the switch starts to close [A].
            transition_time: Time over which the switch transitions from high to low resistance [s].
            voltage_slew_rate: Maximum rate of change of voltage across the switch [V/s].
            current_slew_rate: Maximum rate of change of current through the switch [A/s].
        """
        self.initial_resistance = initial_resistance
        self.final_resistance = final_resistance
        self.transition_voltage = transition_voltage
        self.transition_current = transition_current
        self.transition_time = transition_time
        self.voltage_slew_rate = voltage_slew_rate
        self.current_slew_rate = current_slew_rate
        self.is_closed = False  # Flag to indicate if the switch is closed
        self.start_time = 0.0  # Time when the switch started closing

    def get_resistance(self, voltage: float, current: float, dt: float) -> float:
        """
        Returns the resistance of the switch at a given time, based on voltage and current.

        Args:
            voltage: The voltage across the switch [V].
            current: The current flowing through the switch [A].
            dt: The time step [s].

        Returns:
            The resistance of the switch [Ω].
        """
        if not self.is_closed:
            # Check if the switch should start closing based on voltage or current
            if abs(voltage) >= self.transition_voltage or abs(current) >= self.transition_current:
                self.is_closed = True
                self.start_time = 0.0  # Time when the switch started closing

        if self.is_closed:
            # Limit the voltage and current slew rates
            max_voltage_change = self.voltage_slew_rate * dt
            max_current_change = self.current_slew_rate * dt

            # Linear transition from initial to final resistance over transition_time
            if self.start_time < self.transition_time:
                resistance = self.initial_resistance - (self.initial_resistance - self.final_resistance) * (self.start_time / self.transition_time)
                self.start_time += dt
                return resistance
            else:
                return self.final_resistance
        else:
            return self.initial_resistance

    def get_inductance(self, voltage: float, current: float, dt: float) -> float:
        """
        Returns the inductance of the switch at a given time (assumed to be constant).

        Args:
            voltage: The voltage across the switch [V].
            current: The current flowing through the switch [A].
            dt: The time step [s].

        Returns:
            The inductance of the switch [H].
        """
        return 0.0  # Switch inductance is assumed to be negligible

class TransmissionLineModel:
    """
    A more accurate transmission line model using the telegrapher's equations.
    """
    def __init__(self, impedance: float = 50.0, length: float = 1.0, velocity_factor: float = 0.7,
                 resistance_per_length: float = 0.1, conductance_per_length: float = 1e-6,
                 capacitance_per_length: float = 100e-12, inductance_per_length: float = 250e-9):
        """
        Initializes the TransmissionLineModel.

        Args:
            impedance: Characteristic impedance of the transmission line [Ω].
            length: Length of the transmission line [m].
            velocity_factor: Velocity factor of the transmission line.
            resistance_per_length: Resistance per unit length [Ω/m].
            conductance_per_length: Conductance per unit length [S/m].
            capacitance_per_length: Capacitance per unit length [F/m].
            inductance_per_length: Inductance per unit length [H/m].
        """
        self.impedance = impedance
        self.length = length
        self.velocity_factor = velocity_factor
        self.resistance_per_length = resistance_per_length
        self.conductance_per_length = conductance_per_length
        self.capacitance_per_length = capacitance_per_length
        self.inductance_per_length = inductance_per_length
        self.delay = length / (c * velocity_factor)  # Transmission line delay
        self.history_V = []  # Store past voltage values
        self.history_I = []  # Store past current values
        self.num_segments = 100  # Number of segments to discretize the transmission line

    def get_reflected_current(self, voltage: float, current: float, dt: float) -> float:
        """
        Returns the reflected current from the transmission line.

        Args:
            voltage: The voltage at the load end of the transmission line [V].
            current: The current flowing into the transmission line [A].
            dt: The time step [s].

        Returns:
            The reflected current [A].
        """
        # Store the current and voltage values in the history
        self.history_V.append(voltage)
        self.history_I.append(current)

        # Calculate the index for the delayed current
        delay_steps = int(self.delay / dt)
        if len(self.history_V) > delay_steps and len(self.history_I) > delay_steps:
            delayed_voltage = self.history_V[-delay_steps]
            delayed_current = self.history_I[-delay_steps]

            # Calculate the reflection coefficient
            reflection_coefficient = (self.impedance - 50.0) / (self.impedance + 50.0)  # Example reflection coefficient

            # Calculate the reflected voltage and current
            reflected_voltage = reflection_coefficient * delayed_voltage
            reflected_current = -reflection_coefficient * delayed_current  # Inverted sign for current

            return reflected_current
        else:
            return 0.0  # No reflection if the delay hasn't passed yet

class CircuitModel:
    """
    High-fidelity RLC circuit dynamically coupled to plasma inductance/resistance.
    """

    def __init__(self, C: float, L0: float, R0: float, anode_radius: float, cathode_radius: float, collision_model: Any, field_manager: FieldManager,
                 V0: float = 0.0, ESR: float = 0.0, ESL: float = 0.0, switch_model: Optional[SwitchModel] = None, transmission_line: bool = False,
                 initial_Q: Optional[float] = None, initial_I: float = 0.0, parasitic_inductance: float = 0.0, stray_capacitance: float = 0.0,
                 transmission_line_impedance: float = 50.0, transmission_line_length: float = 1.0, transmission_line_velocity_factor: float = 0.7,
                 switch_initial_resistance: float = 1e6, switch_final_resistance: float = 1e-3,
                 switch_transition_voltage: float = 1e3, switch_transition_current: float = 1e3,
                 switch_transition_time: float = 1e-9, switch_voltage_slew_rate: float = 1e12, switch_current_slew_rate: float = 1e12):
        """
        Initializes the CircuitModel.

        Args:
            C: Capacitance [F].
            L0: External inductance [H].
            R0: External resistance [Ω].
            anode_radius: Anode radius [m].
            cathode_radius: Cathode radius [m].
            collision_model: Collision model object (must have a spitzer_resistivity method).
            field_manager: FieldManager object for accessing field data.
            V0: Initial voltage [V].
            ESR: Equivalent series resistance [Ω].
            ESL: Equivalent series inductance [H].
            switch_model: Optional switch model object.
            transmission_line: Whether to include a transmission line model.
            initial_Q: Initial charge on the capacitor [C]. If None, defaults to C * V0.
            initial_I: Initial current in the circuit [A].
            parasitic_inductance: Parasitic inductance [H].
            stray_capacitance: Stray capacitance [F].
            transmission_line_impedance: Impedance of the transmission line [Ω].
            transmission_line_length: Length of the transmission line [m].
            transmission_line_velocity_factor: Velocity factor of the transmission line.
            switch_initial_resistance: Initial resistance of the switch [Ω].
            switch_final_resistance: Final resistance of the switch [Ω].
            switch_transition_voltage: Voltage at which the switch starts to close [V].
            switch_transition_current: Current at which the switch starts to close [A].
            switch_transition_time: Time over which the switch transitions from high to low resistance [s].
            switch_voltage_slew_rate: Maximum rate of change of voltage across the switch [V/s].
            switch_current_slew_rate: Maximum rate of change of current through the switch [A/s].
        """
        if not all(isinstance(x, (int, float)) and x >= 0 for x in [C, L0, R0, anode_radius, cathode_radius, V0, ESR, ESL, parasitic_inductance, stray_capacitance, transmission_line_impedance, transmission_line_length, transmission_line_velocity_factor, switch_initial_resistance, switch_final_resistance, switch_transition_voltage, switch_transition_current, switch_transition_time, switch_voltage_slew_rate, switch_current_slew_rate]):
            raise ValueError("Capacitance, inductance, resistance, radii, and initial voltage must be non-negative numbers.")
        if anode_radius >= cathode_radius:
            raise ValueError("Anode radius must be smaller than cathode radius.")
        if not hasattr(collision_model, 'spitzer_resistivity'):
            raise ValueError("Collision model must have a spitzer_resistivity method.")

        self.C = C
        self.L0 = L0
        self.R0 = R0
        self.a = anode_radius
        self.b = cathode_radius
        self.collision_model = collision_model
        self._lnΛ = 10.0  # Default Coulomb logarithm
        self.ESR = ESR
        self.ESL = ESL
        self.Ls = parasitic_inductance
        self.Cs = stray_capacitance

        # Initial state: Q = C*V0; I = 0
        self.Q = initial_Q if initial_Q is not None else C * V0
        self.I = initial_I

        # Switch model
        if switch_model is None:
            self.switch_model = SwitchModel(switch_initial_resistance, switch_final_resistance, switch_transition_voltage, switch_transition_current, switch_transition_time, switch_voltage_slew_rate, switch_current_slew_rate)
        else:
            self.switch_model = switch_model

        # Transmission line model
        if self.transmission_line:
            self.transmission_line_model = TransmissionLineModel(transmission_line_impedance, transmission_line_length, transmission_line_velocity_factor)
        else:
            self.transmission_line_model = None

        logger.info("CircuitModel initialized.")

    def plasma_inductance(self, state: SimulationState) -> float:
        """
        Calculates the plasma inductance using a more sophisticated model.

        Args:
            state: The current state of the simulation.

        Returns:
            Plasma inductance [H].
        """
        try:
            z = state.sheath_position  # Access sheath position from SimulationState
            if not isinstance(z, (int, float)) or z <= 0:
                raise ValueError("Sheath position must be a positive number.")

            # More accurate inductance calculation (example)
            # This is a placeholder; replace with a more sophisticated model
            L_plasma = mu_0 / (2 * np.pi) * z * np.log(self.b / self.a) * (1 + 0.1 * np.sin(2 * np.pi * z / self.b))
            return L_plasma
        except AttributeError:
            raise AttributeError("SimulationState object must have a 'sheath_position' attribute.")
        except ValueError as e:
            raise ValueError(f"Invalid sheath position: {e}")

    def plasma_resistance(self, state: SimulationState) -> float:
        """
        Calculates the plasma resistance, considering current density distribution.

        Args:
            state: The current state of the simulation.

        Returns:
            Plasma resistance [Ω].
        """
        try:
            Te = state.electron_temperature  # Access electron temperature from SimulationState
            ne = state.density / 1.67e-27  # Assuming proton mass
            z = state.sheath_position  # Access sheath position from SimulationState

            if not all(isinstance(x, (int, float)) and x > 0 for x in [Te, ne, z]):
                raise ValueError("Electron temperature, density, and sheath position must be positive numbers.")

            η = self.collision_model.spitzer_resistivity(ne, Te, self._lnΛ)  # [Ω·m]

            # Assume a Gaussian current density profile
            sigma = 0.5 * (self.b - self.a)  # Example width
            J0 = self.I / (np.pi * sigma**2) # Estimate peak current density

            # Integrate the resistivity over the current distribution
            R_plasma = η * z / (2 * np.pi * sigma**2) * (1 - erf((self.b - self.a) / sigma))
            return R_plasma

        except AttributeError:
            raise AttributeError("SimulationState object must have 'electron_temperature', 'density', and 'sheath_position' attributes.")
        except ValueError as e:
            raise ValueError(f"Invalid plasma parameters: {e}")

    def step(self, state: SimulationState, dt: float, method: str = "RK4") -> float:
        """
        Integrates the circuit ODEs over dt.

        Args:
            state: The current state of the simulation.
            dt: The time step [s].
            method: The integration method to use ("RK4" or "trapezoidal").

        Returns:
            The circuit current [A].
        """
        if not isinstance(dt, (int, float)) or dt <= 0:
            raise ValueError("Time step (dt) must be a positive number.")

        if method == "RK4":
            return self._step_rk4(state, dt)
        elif method == "trapezoidal":
            return self._step_trapezoidal(state, dt)
        elif method == "half_step":
            return self._half_step(state, dt)
        else:
            raise ValueError(f"Invalid method: {method}")

    def _step_rk4(self, state: SimulationState, dt: float) -> float:
        """
        Integrates the circuit ODEs over dt using RK4.

        Args:
            state: The current state of the simulation.
            dt: The time step [s].

        Returns:
            The circuit current [A].
        """
        C = self.C
        L0 = self.L0
        R0 = self.R0
        ESR = self.ESR
        ESL = self.ESL
        Ls = self.Ls
        Cs = self.Cs

        # Helper: compute RHS given Q, I
        def rhs(Q, I, R, L, Lp, Rp, Ls, Cs):
            return _rhs(Q, I, R, L, C, Lp, Rp, Ls, Cs)

        # Compute time-varying L_plasma & R_plasma at current state
        try:
            Lp = self.plasma_inductance(state)
            Rp = self.plasma_resistance(state)
        except (ValueError, AttributeError) as e:
            raise RuntimeError(f"Error calculating plasma inductance/resistance: {e}")

        # Total parameters
        R_tot = R0 + Rp + ESR
        L_tot = L0 + Lp + ESL + Ls

        # Apply switch model (if any)
        if self.switch_model:
            R_tot += self.switch_model.get_resistance(self.get_voltage(), self.I, dt)
            L_tot += self.switch_model.get_inductance(self.get_voltage(), self.I, dt)

        # Apply transmission line model (if any)
        if self.transmission_line_model:
            reflected_current = self.transmission_line_model.get_reflected_current(self.get_voltage(), self.I, dt)
            # Adjust the total current based on the reflected current
            I0 = self.I - reflected_current

        # RK4 for [Q, I]
        Q0, I0 = self.Q, self.I
        try:
            k1_Q, k1_I = rhs(Q0, I0, R_tot, L_tot, Lp, Rp, Ls, Cs)
            k2_Q, k2_I = rhs(Q0 + 0.5 * dt * k1_Q, I0 + 0.5 * dt * k1_I, R_tot, L_tot, Lp, Rp, Ls, Cs)
            k3_Q, k3_I = rhs(Q0 + 0.5 * dt * k2_Q, I0 + 0.5 * dt * k2_I, R_tot, L_tot, Lp, Rp, Ls, Cs)
            k4_Q, k4_I = rhs(Q0 + dt * k3_Q, I0 + dt * k3_I, R_tot, L_tot, Lp, Rp, Ls, Cs)
        except Exception as e:
            raise RuntimeError(f"Error during RK4 integration: {e}")

        self.Q = Q0 + dt * (k1_Q + 2 * k2_Q + 2 * k3_Q + k4_Q) / 6
        self.I = I0 + dt * (k1_I + 2 * k2_I + 2 * k3_I + k4_I) / 6

        # Check for negative voltage
        voltage = self.get_voltage()
        if voltage < 0:
            logger.warning(f"Negative voltage detected: {voltage:.3e} V")

        return self.I

    def _step_trapezoidal(self, state: SimulationState, dt: float) -> float:
        """
        Integrates the circuit ODEs over dt using trapezoidal, with dynamic L_tot, R_tot.

        Args:
            state: The current state of the simulation.
            dt: The time step [s].

        Returns:
            The circuit current [A].
        """
        try:
            C = self.C
            L0 = self.L0
            R0 = self.R0
            ESR = self.ESR
            ESL = self.ESL
            Ls = self.Ls
            Cs = self.Cs

            # Compute time-varying L_plasma & R_plasma at current state
            Lp = self.plasma_inductance(state)
            Rp = self.plasma_resistance(state)

            # Total parameters
            R_tot = R0 + Rp + ESR
            L_tot = L0 + Lp + ESL + Ls

            # Apply switch model (if any)
            if self.switch_model:
                R_tot += self.switch_model.get_resistance(self.I, dt)
                L_tot += self.switch_model.get_inductance(self.I, dt)

            # Apply transmission line model (if any)
            if self.transmission_line_model:
                reflected_current = self.transmission_line_model.get_reflected_current(self.I, dt)
                # Adjust the total current based on the reflected current
                I0 -= reflected_current

            # trapezoidal
            Q0, I0 = self.Q, self.I
            dIdt = (-R_tot * I0 - Q0 / C) / L_tot
            self.Q -= dt * (I0 + 0.5*dt*dIdt)
            self.I += dt * dIdt

            # Check for negative voltage
            voltage = self.Q / self.C
            if voltage < 0:
                logger.warning(f"Negative voltage detected: {voltage:.3e} V")

            return self.I
        except (ValueError, AttributeError, RuntimeError) as e:
            raise RuntimeError(f"Error during trapezoidal integration: {e}")

    def _half_step(self, state: SimulationState, dt: float) -> float:
        """
        Integrate circuit ODEs over dt using trapezoidal, with dynamic L_tot, R_tot.

        Args:
            state: The current state of the simulation.
            dt: The time step [s].

        Returns:
            The circuit current [A].
        """
        try:
            C = self.C
            L0 = self.L0
            R0 = self.R0
            ESR = self.ESR
            ESL = self.ESL
            Ls = self.Ls
            Cs = self.Cs

            # Compute time-varying L_plasma & R_plasma at current state
            Lp = self.plasma_inductance(state)
            Rp = self.plasma_resistance(state)

            # Total parameters
            R_tot = R0 + Rp + ESR
            L_tot = L0 + Lp + ESL + Ls

            # Apply switch model (if any)
            if self.switch_model:
                R_tot += self.switch_model.get_resistance(self.I, dt)
                L_tot += self.switch_model.get_inductance(self.I, dt)

            # Apply transmission line model (if any)
            if self.transmission_line_model:
                reflected_current = self.transmission_line_model.get_reflected_current(self.I, dt)
                # Adjust the total current based on the reflected current
                I0 -= reflected_current

            # trapezoidal
            Q0, I0 = self.Q, self.I
            dIdt = (-R_tot * I0 - Q0 / C) / L_tot
            self.Q -= dt * I0
            self.I += dt * dIdt

            # Check for negative voltage
            voltage = self.Q / self.C
            if voltage < 0:
                logger.warning(f"Negative voltage detected: {voltage:.3e} V")

            return self.I
        except (ValueError, AttributeError, RuntimeError) as e:
            raise RuntimeError(f"Error during half_step integration: {e}")

    def get_current(self) -> float:
        """Return instantaneous circuit current [A]."""
        return self.I

    def get_voltage(self) -> float:
        """Return capacitor voltage V = Q/C."""
        return self.Q / self.C

    def initialize(self):
        """
        Initializes the circuit model.
        """
        logger.info("CircuitModel initialized.")

    def finalize(self):
        """
        Finalizes the circuit model.
        """
        logger.info("CircuitModel finalized.")

    def configure(self, config: Dict[str, Any]):
        """Configures the circuit model."""
        try:
            for key, value in config.items():
                setattr(self, key, value)
            logger.info(f"CircuitModel configured with: {config}")
        except Exception as e:
            logger.error(f"Error configuring CircuitModel: {e}")

    def validate(self, analytical_waveform_data):
        """Validates the circuit model against analytical discharge waveforms."""
        try:
            # Load analytical waveform data
            analytical_time = analytical_waveform_data['time']
            analytical_current = analytical_waveform_data['current']

            # Simulate the circuit using the same time steps as the analytical data
            simulated_time = []
            simulated_current = []
            dt = analytical_time[1] - analytical_time[0]  # Assuming uniform time steps
            t = 0.0
            while t <= analytical_time[-1]:
                simulated_time.append(t)
                # Create a dummy SimulationState
                class DummySimulationState:
                    def __init__(self):
                        self.sheath_position = 0.0
                        self.electron_temperature = 1.0
                        self.density = 1.0
                state = DummySimulationState()
                self.step(state, dt)
                simulated_current.append(self.get_current())
                t += dt

            # Compare the simulated and analytical waveforms
            simulated_current_interp = np.interp(analytical_time, simulated_time, simulated_current)
            error = np.sqrt(np.mean((np.array(simulated_current_interp) - analytical_current)**2))

            logger.info(f"Validation error: {error:.3e}")

            # You can add a threshold for the error to determine if the model is valid
            if error > 0.1:
                logger.warning("Validation failed: error exceeds threshold.")
            else:
                logger.info("Validation successful.")

        except Exception as e:
            logger.error(f"Error during validation: {e}")

    def check_component_limits(self):
        """Checks if the circuit components are within their operating limits."""
        try:
            voltage = self.get_voltage()
            current = self.get_current()

            # Example checks (replace with your specific component limits)
            if abs(voltage) > 1e6:
                logger.warning(f"Voltage exceeds limit: {voltage:.3e} V")
            if abs(current) > 1e5:
                logger.warning(f"Current exceeds limit: {current:.3e} A")

        except Exception as e:
            logger.error(f"Error checking component limits: {e}")
