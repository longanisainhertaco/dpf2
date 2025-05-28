# solver_selector.py
from fluid_solver_high_order import FluidSolverHighOrder
from pic_solver import PICSolver
# from amrex_solver import FluidSolverAMReX  # Keep for potential future use, but comment out
from utils import FieldManager  # Import FieldManager
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

def select_solver(backend: str, config: Dict[str, Any], field_manager: FieldManager) -> PhysicsModule:
    """
    Selects and returns a solver based on the specified backend.

    Args:
        backend: The name of the backend to use ('high_order', 'pic', 'amrex').
        config: The configuration dictionary for the solver.
        field_manager: The FieldManager object.

    Returns:
        A PhysicsModule instance (FluidSolverHighOrder, PICSolver, or FluidSolverAMReX).

    Raises:
        ValueError: If an invalid backend is specified.
    """
    try:
        if backend == 'high_order':
            solver = FluidSolverHighOrder(
                geom=None,  # Geometry will be set later
                config=config,
                field_manager=field_manager  # Pass FieldManager to FluidSolverHighOrder
            )
        elif backend == 'pic':
            solver = PICSolver(
                config=config,
                field_manager=field_manager
            )
        # elif backend == 'amrex':  # Keep for potential future use, but comment out
        #     solver = FluidSolverAMReX(
        #         geom=None,  # Geometry will be set later
        #         config=config,
        #         field_manager=field_manager  # Pass FieldManager to FluidSolverAMReX
        #     )
        else:
            raise ValueError(f"Invalid solver backend: {backend}")

        logger.info(f"Selected solver: {backend}")
        return solver
    except Exception as e:
        logger.error(f"Error creating solver: {e}")
        raise

def initialize_solver(solver: PhysicsModule, config: Dict[str, Any]) -> None:
    """
    Initializes the selected solver with the given configuration.

    Args:
        solver: The solver instance to initialize.
        config: The configuration dictionary for the solver.
    """
    try:
        solver.configure(config)
        solver.initialize()
        logger.info(f"Initialized solver: {solver.__class__.__name__}")
    except Exception as e:
        logger.error(f"Error initializing solver: {e}")
        raise
