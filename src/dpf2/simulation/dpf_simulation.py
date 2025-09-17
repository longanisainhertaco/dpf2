#!/usr/bin/env python3
"""
DPF Simulation Launcher: Comprehensive multi-physics configuration.
"""
import os
import sys
import time
import json
import socket
import subprocess
import logging
import argparse
import numpy as np
import random
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, Mapping

try:  # Prefer package-local imports
    from .config_schema import SimulationConfig, FieldManagerConfig, PICConfig, AMRConfig
except Exception:  # pragma: no cover - fallback when loaded as a script
    from config_schema import SimulationConfig, FieldManagerConfig  # type: ignore
    class PICConfig:  # type: ignore
        pass
    class AMRConfig:  # type: ignore
        enable: bool = False
        max_level: int = 1
        refinement_threshold: float = 1.0
        diag_interval: int = 10

_AMR = AMRConfig
from .module_registry import ModuleRegistry
from .collision_model import CollisionModel
from .radiation_model import RadiationModel
from .hybrid_controller import HybridController
from .eos_selector import select_eos
from .solver_selector import select_solver
from .circuit import CircuitModel
from .utils import FieldManager, SimulationState
try:
    from .diagnostics import Diagnostics
except ModuleNotFoundError:  # pragma: no cover - compatibility with legacy layout
    from ..diagnostics import Diagnostics  # type: ignore
from .pic_solver import PICSolver
from ..core.bases import CouplingState
from ..materials import (
    MaterialLibrary,
    ComponentMaterialState,
    MaterialDamageModel,
)
try:
    from ..exceptions import SimulationRuntimeError
except Exception:  # pragma: no cover - fallback for standalone usage
    class SimulationRuntimeError(RuntimeError):
        pass

logger = logging.getLogger("DPFSimulationWrapper")

def _namespace_to_plain(value: Any) -> Any:
    if isinstance(value, _ConfigNamespace):
        return value.dict()
    if isinstance(value, Mapping):
        return {k: _namespace_to_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_namespace_to_plain(v) for v in value]
    return value


class _ConfigNamespace(SimpleNamespace):
    """Lightweight mapping that mimics a Pydantic model."""

    def dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial accessors
        return {k: _namespace_to_plain(v) for k, v in self.__dict__.items()}

    # ``pydantic`` 2.x uses ``model_dump`` while 1.x exposes ``dict``.
    model_dump = dict  # type: ignore[assignment]


def _as_namespace(data: Mapping[str, Any]) -> _ConfigNamespace:
    """Recursively convert ``data`` into a :class:`_ConfigNamespace`."""

    converted = {}
    for key, value in data.items():
        if isinstance(value, Mapping):
            converted[key] = _as_namespace(value)
        elif isinstance(value, list):
            converted[key] = [
                _as_namespace(item) if isinstance(item, Mapping) else item
                for item in value
            ]
        else:
            converted[key] = value
    return _ConfigNamespace(**converted)


# Custom Exceptions
class ConfigurationError(Exception):
    pass

class InitializationError(Exception):
    pass


class DPFSimulation:
    def __init__(
        self,
        config: SimulationConfig | Mapping[str, Any] | Any,
        *,
        field_manager: FieldManager | None = None,
    ) -> None:
        self.config = self._load_config(config)
        self.modules: Dict[str, Any] = {}
        self.step_count = 0
        self.current_time = 0.0
        self.dt = getattr(self.config, "dt_init", None)
        self.amr_config = getattr(self.config, "amr", _AMR())
        self.material_model: MaterialDamageModel | None = None
        self._external_field_manager = field_manager

        self.field_manager: FieldManager | None = None
        self.state: SimulationState | None = None
        self.solver = None
        self.pic_solver = None
        self.diagnostics = None

        # Initialize modules
        self.registry = ModuleRegistry()
        self.register_modules()
        self.initialize_modules()

    def register_modules(self) -> None:
        """Registers available modules with the registry."""

        self.registry.register(CollisionModel, field_manager_required=True)
        self.registry.register(RadiationModel, field_manager_required=True)
        self.registry.register(HybridController, field_manager_required=True)

    def initialize_modules(self) -> None:
        """Initializes modules based on the configuration."""

        try:
            cfg = self.config
            grid_shape = tuple(getattr(cfg, "grid_shape", ()))
            if len(grid_shape) != 3:
                raise ConfigurationError("grid_shape must contain three entries")
            dx = float(getattr(cfg, "dx", 0.0))
            dy = float(getattr(cfg, "dy", 0.0))
            dz = float(getattr(cfg, "dz", 0.0))
            domain_lo = tuple(getattr(cfg, "domain_lo", (0.0, 0.0, 0.0)))
            if self.dt is None:
                spacing = [v for v in (dx, dy, dz) if v]
                self.dt = min(spacing) * 0.1 if spacing else 1e-9

            eos_backend = getattr(cfg, "eos_backend", "tabulated")
            mixture = (
                getattr(cfg, "mixture_fractions", None)
                if getattr(cfg, "enable_eos_mixture", False)
                else None
            )
            table_file = getattr(cfg, "table_file", None)
            self.eos = select_eos(
                backend=eos_backend,
                table_file=table_file,
                mixture_fractions=mixture,
            )

            self.field_manager = self._ensure_field_manager(
                grid_shape, dx, dy, dz, domain_lo
            )
            self.state = SimulationState(
                grid_shape=grid_shape,
                dx=dx,
                dy=dy,
                dz=dz,
                domain_lo=domain_lo,
                boundary_conditions=self.field_manager.boundary_conditions,
                field_manager=self.field_manager,
                amr_config=self.amr_config,
            )

            backend = getattr(cfg, "solver_backend", "pic")
            solver_config = {
                "grid_shape": grid_shape,
                "dx": dx,
                "dy": dy,
                "dz": dz,
            }
            self.solver = select_solver(
                backend=backend,
                config=solver_config,
                field_manager=self.field_manager,
            )
            self.gamma = getattr(self.solver, "gamma", 5.0 / 3.0)

            self.pic_solver = self._instantiate_pic_solver(self.dt)

            collision_cfg = getattr(cfg, "collision", None)
            if collision_cfg:
                self.modules["collision"] = self.registry.create(
                    CollisionModel,
                    self._section_dict(collision_cfg),
                    field_manager=self.field_manager,
                )

            radiation_cfg = getattr(cfg, "radiation", None)
            if radiation_cfg:
                self.modules["radiation"] = self.registry.create(
                    RadiationModel,
                    self._section_dict(radiation_cfg),
                    field_manager=self.field_manager,
                )

            circuit_cfg = getattr(cfg, "circuit", None)
            if circuit_cfg is None:
                raise ConfigurationError("Circuit configuration is required")
            circuit_dict = self._section_dict(circuit_cfg)
            self.circuit = CircuitModel(
                collision_model=self.modules.get("collision"),
                field_manager=self.field_manager,
                **circuit_dict,
            )

            hybrid_cfg = getattr(cfg, "hybrid", None)
            if hybrid_cfg:
                hybrid_dict = self._section_dict(hybrid_cfg)
                hybrid_dict.update(
                    fluid_solver=self.solver,
                    pic_solver=self.pic_solver,
                    circuit_model=self.circuit,
                    radiation_model=self.modules.get("radiation"),
                    field_manager=self.field_manager,
                )
                self.modules["hybrid"] = self.registry.create(
                    HybridController,
                    hybrid_dict,
                    field_manager=self.field_manager,
                )

            diag_cfg = getattr(cfg, "diagnostics", None)
            if diag_cfg:
                diag_dict = self._section_dict(diag_cfg)
                diag_config: Dict[str, Any] = {}
                diag_config.update(circuit_dict)
                if collision_cfg:
                    diag_config.update(self._section_dict(collision_cfg))
                if radiation_cfg:
                    diag_config.update(self._section_dict(radiation_cfg))
                if getattr(cfg, "pic", None):
                    diag_config.update(self._section_dict(cfg.pic))  # type: ignore[arg-type]
                if hybrid_cfg:
                    diag_config.update(self._section_dict(hybrid_cfg))

                full_interval = int(diag_dict.get("field_diagnostic_interval", 10))
                adaptive_threshold = float(
                    diag_dict.get("adaptive_interval_threshold", 0.1)
                )
                self.modules["diagnostics"] = Diagnostics(
                    hdf5_filename=diag_dict.get("hdf5_filename", "diagnostics.h5"),
                    config=diag_config,
                    domain_lo=domain_lo,
                    grid_shape=grid_shape,
                    dx=dx,
                    gamma=self.gamma,
                    field_manager=self.field_manager,
                    full_interval=full_interval,
                    adaptive_interval_threshold=adaptive_threshold,
                )
                self.diagnostics = self.modules["diagnostics"]

            self.material_model = self._build_material_model()

        except Exception as exc:
            raise InitializationError(f"Failed to initialize modules: {exc}") from exc

    def _ensure_field_manager(
        self,
        grid_shape: tuple[int, int, int],
        dx: float,
        dy: float,
        dz: float,
        domain_lo: tuple[float, float, float],
    ) -> FieldManager:
        if self._external_field_manager is not None:
            if not hasattr(self._external_field_manager, "boundary_conditions"):
                setattr(self._external_field_manager, "boundary_conditions", {})
            return self._external_field_manager

        fm_cfg = self._section_dict(getattr(self.config, "field_manager", None))
        boundary = fm_cfg.get("boundary_conditions", {})
        fm = FieldManager(
            grid_shape=grid_shape,
            dx=dx,
            dy=dy,
            dz=dz,
            domain_lo=domain_lo,
            boundary_conditions=boundary,
        )
        if not hasattr(fm, "boundary_conditions"):
            setattr(fm, "boundary_conditions", boundary)
        return fm

    def _instantiate_pic_solver(self, base_dt: float | None):
        pic_cfg_obj = getattr(self.config, "pic", None)
        if not pic_cfg_obj:
            return None

        pic_cfg = self._section_dict(pic_cfg_obj)
        pic_cfg.setdefault("grid_shape", tuple(getattr(self.config, "grid_shape", ())))
        pic_cfg.setdefault(
            "grid_spacing",
            (
                float(getattr(self.config, "dx", 0.0)),
                float(getattr(self.config, "dy", 0.0)),
                float(getattr(self.config, "dz", 0.0)),
            ),
        )
        if base_dt is not None:
            pic_cfg.setdefault("max_dt", base_dt)
        try:
            if hasattr(PICConfig, "model_validate"):
                pic_model = PICConfig.model_validate(pic_cfg)  # type: ignore[attr-defined]
            else:
                pic_model = PICConfig(**pic_cfg)  # type: ignore[call-arg]
        except Exception:
            pic_model = _as_namespace(pic_cfg)
        return PICSolver(config=pic_model, field_manager=self.field_manager)  # type: ignore[arg-type]

    def _build_material_model(self) -> MaterialDamageModel | None:
        materials_cfg = getattr(self.config, "materials", None)
        if not materials_cfg:
            return None
        materials_dict = self._section_dict(materials_cfg)
        components = materials_dict.get("components", {})
        if not components:
            return None
        initial_state = materials_dict.get("initial_state", {})
        component_states: Dict[str, ComponentMaterialState] = {}
        for comp, mat_ref in components.items():
            material_id = None
            if isinstance(mat_ref, Mapping):
                material_id = mat_ref.get("material_id")
            else:
                material_id = getattr(mat_ref, "material_id", None)
            if not material_id:
                continue
            material = MaterialLibrary.get(material_id)
            init = initial_state.get(comp, {})
            component_states[comp] = ComponentMaterialState(
                material=material,
                erosion=float(init.get("erosion", 0.0)),
                film_thickness=float(init.get("film_thickness", 0.0)),
            )
        if not component_states:
            return None
        plasma_model = self.modules.get("collision")
        return MaterialDamageModel(component_states, plasma_model=plasma_model)

    @staticmethod
    def _section_dict(section: Any) -> Dict[str, Any]:
        if section is None:
            return {}
        if isinstance(section, Mapping):
            return dict(section)
        if hasattr(section, "model_dump"):
            data = section.model_dump()
            return dict(data) if isinstance(data, Mapping) else data
        if hasattr(section, "dict"):
            data = section.dict()
            return dict(data) if isinstance(data, Mapping) else data
        if hasattr(section, "__dict__"):
            return {k: getattr(section, k) for k in vars(section)}
        return {}

    @staticmethod
    def _load_config(config: Any) -> Any:
        if isinstance(config, SimulationConfig):
            return config

        serialisable: Any = None
        if isinstance(config, Mapping):
            serialisable = dict(config)
        elif hasattr(config, "model_dump"):
            serialisable = config.model_dump()
        elif hasattr(config, "dict"):
            serialisable = config.dict()
        elif hasattr(config, "__dict__"):
            serialisable = {k: getattr(config, k) for k in vars(config)}
        else:
            serialisable = config

        for attr in ("model_validate", "parse_obj"):
            fn = getattr(SimulationConfig, attr, None)
            if callable(fn):
                try:
                    return fn(serialisable)  # type: ignore[misc]
                except Exception:
                    continue

        if isinstance(serialisable, Mapping):
            return _as_namespace(serialisable)
        return config

    def run(self) -> None:
        """Runs the simulation."""

        if self.solver is None or self.field_manager is None:
            raise InitializationError("Simulation modules are not fully initialised")

        try:
            sim_time = float(getattr(self.config, "sim_time", 0.0))
            while self.current_time < sim_time:
                # --- determine timestep ---
                dt_candidate = None
                if hasattr(self.solver, "compute_optimal_dt"):
                    dt_candidate = self.solver.compute_optimal_dt()
                elif hasattr(self.solver, "compute_dt"):
                    dt_candidate = self.solver.compute_dt()
                if dt_candidate is not None:
                    self.dt = float(dt_candidate)
                if self.dt is None or self.dt <= 0.0:
                    raise RuntimeError("Invalid time step")
                if self.current_time + self.dt > sim_time:
                    self.dt = sim_time - self.current_time

                # --- advance primary solver or hybrid controller ---
                if "hybrid" in self.modules:
                    try:
                        self.modules["hybrid"].apply(self.state, self.dt)
                    except Exception as exc:
                        logger.error(f"Hybrid controller error: {exc}")
                else:
                    self.solver.step(self.dt)
                    if self.pic_solver:
                        self.pic_solver.step()

                # --- material damage model ---
                if self.material_model and hasattr(self.material_model, "apply"):
                    self.material_model.apply(self.solver, self.dt)

                # --- collision and radiation modules ---
                for name in ("collision", "radiation"):
                    module = self.modules.get(name)
                    if module:
                        try:
                            module.apply(self.state, self.dt)
                        except Exception as exc:
                            logger.error(f"{name.capitalize()} module error: {exc}")

                # --- circuit update ---
                try:
                    current = getattr(self.solver, "current", None)
                    if callable(current):
                        current = current()
                    if current is None:
                        try:
                            current = self.field_manager.get_J()
                        except Exception:
                            current = 0.0
                    if isinstance(current, np.ndarray):
                        current = float(np.mean(current))

                    back_emf = getattr(self.solver, "back_emf", None)
                    if callable(back_emf):
                        back_emf = back_emf()
                    if back_emf is None:
                        back_emf = 0.0

                    feedback = getattr(self.solver, "circuit_feedback", None)

                    if isinstance(feedback, CouplingState):
                        coupling = feedback
                        coupling.current = current
                        coupling.voltage = (
                            self.circuit.get_voltage()
                            if hasattr(self.circuit, "get_voltage")
                            else 0.0
                        )
                    else:
                        feedback = feedback or {}
                        coupling = CouplingState(
                            Lp=feedback.get("Lp", 0.0),
                            emf=feedback.get("emf", 0.0),
                            current=current,
                            voltage=(
                                self.circuit.get_voltage()
                                if hasattr(self.circuit, "get_voltage")
                                else 0.0
                            ),
                        )

                    if hasattr(self.circuit, "step"):
                        self.circuit.step(coupling, back_emf, self.dt)
                except Exception as exc:
                    logger.error(f"Circuit step failed: {exc}")

                # --- diagnostics and checkpointing ---
                checkpoint_data = {}
                for name, module in self.modules.items():
                    if hasattr(module, "checkpoint"):
                        try:
                            checkpoint_data[name] = module.checkpoint()
                        except Exception as exc:
                            logger.error(f"Checkpoint failed for {name}: {exc}")
                diagnostics = self.diagnostics or self.modules.get("diagnostics")
                if diagnostics:
                    try:
                        diagnostics.record(
                            self.current_time,
                            self.circuit,
                            self.solver,
                            self.pic_solver,
                            self.modules.get("radiation"),
                            checkpoint_id=self.step_count,
                            state=self.state,
                        )
                        if hasattr(diagnostics, "checkpoints"):
                            diagnostics.checkpoints.append(checkpoint_data)
                    except Exception as exc:
                        logger.error(f"Diagnostics error: {exc}")

                # --- advance time ---
                self.step_count += 1
                self.current_time += self.dt
                logger.info(f"Step {self.step_count}: time={self.current_time:.3e}")

        except Exception as e:
            raise SimulationRuntimeError(
                f"Simulation failed at step {self.step_count}: {e}"
            )

    def finalize(self):
        """Finalizes the simulation."""
        logger.info("Simulation completed.")

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="DPF Simulation Launcher")
    parser.add_argument("--config-file", type=str, required=True, help="Path to the JSON configuration file")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                        help="Global logging level")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument(
        "--enable-tracing",
        action="store_true",
        help="Enable OpenCensus tracing for simulation stages",
    )
    parser.add_argument("--relativistic-corrections", action="store_true", help="Enable relativistic PIC corrections")
    parser.add_argument("--quantum-emission", action="store_true", help="Enable quantum emission module")
    parser.add_argument("--time-dependent-boundaries", action="store_true", help="Enable time-dependent PIC boundaries")
    args = parser.parse_args()
    return args

def load_config_from_json(filepath):
    """Loads configuration from a JSON file and validates it."""
    try:
        with open(filepath, 'r') as f:
            config_data = json.load(f)
            config = SimulationConfig(**config_data)  # Validate the config
            return config
    except FileNotFoundError:
        raise ConfigurationError(f"Configuration file not found: {filepath}")
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Error decoding JSON from {filepath}: {e}")
    except Exception as e:
        raise ConfigurationError(f"Error validating configuration: {e}")

def main():
    args = parse_arguments()

    # Configure logging
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # Seed RNGs
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")

    # Load and validate configuration
    try:
        config = load_config_from_json(args.config_file)
        if config.pic:
            if args.relativistic_corrections:
                config.pic.relativistic_corrections = True
            if args.quantum_emission:
                config.pic.quantum_emission = True
            if args.time_dependent_boundaries:
                config.pic.time_dependent_boundaries = True
    except ConfigurationError as e:
        logger.error(e)
        sys.exit(1)

    tracer = None
    if args.enable_tracing:
        try:
            from opencensus.trace.tracer import Tracer

            tracer = Tracer()
        except ModuleNotFoundError:
            logger.error(
                "OpenCensus is required for tracing but is not installed."
            )
            sys.exit(1)

    # Instantiate and run the simulation
    try:
        if tracer:
            with tracer.span(name="initialize"):
                sim = DPFSimulation(config)
            with tracer.span(name="run"):
                sim.run()
            with tracer.span(name="finalize"):
                sim.finalize()
        else:
            sim = DPFSimulation(config)
            sim.run()
            sim.finalize()
    except InitializationError as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)
    except SimulationRuntimeError as e:
        logger.error(f"Simulation failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
