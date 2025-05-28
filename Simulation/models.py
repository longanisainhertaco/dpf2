# models.py
from abc import ABC, abstractmethod
from typing import Dict, Any
from utils import SimulationState  # Import SimulationState
import logging

logger = logging.getLogger(__name__)

class PhysicsModule(ABC):
    """
    Abstract base class for all physics modules in the DPF simulation.

    This class defines the interface that all physics modules must implement.
    It includes methods for initialization, finalization, applying the module,
    getting diagnostics, checkpointing, restarting, and configuring.
    """

    @abstractmethod
    def apply(self, state: SimulationState, dt: float) -> None:
        """
        Applies the physics module to the simulation state.

        Args:
            state: The current simulation state.
            dt: The time step.
        """
        pass

    def initialize(self) -> None:
        """
        Optional method for module initialization.

        This method can be overridden to perform setup tasks, such as
        loading data or connecting to resources.
        """
        logger.info(f"Initializing module: {self.__class__.__name__}")
        pass

    def finalize(self) -> None:
        """
        Optional method for module finalization.

        This method can be overridden to perform cleanup tasks, such as
        closing connections or saving data.
        """
        logger.info(f"Finalizing module: {self.__class__.__name__}")
        pass

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Optional method to return diagnostic information.

        Returns:
            A dictionary containing diagnostic information.
        """
        logger.debug(f"Getting diagnostics from module: {self.__class__.__name__}")
        return {}

    def checkpoint(self) -> Dict[str, Any]:
        """
        Optional method to return a dictionary of data to checkpoint.

        Returns:
            A dictionary containing data to be saved for checkpointing.
        """
        logger.debug(f"Checkpointing module: {self.__class__.__name__}")
        return {}

    def restart(self, data: Dict[str, Any]) -> None:
        """
        Optional method to load data from a checkpoint.

        Args:
            data: A dictionary containing data loaded from a checkpoint.
        """
        logger.info(f"Restarting module: {self.__class__.__name__}")
        pass

    def configure(self, config: Dict[str, Any]) -> None:
        """
        Optional method to configure the module.

        Args:
            config: A dictionary containing configuration parameters.
        """
        logger.info(f"Configuring module: {self.__class__.__name__}")
        pass
