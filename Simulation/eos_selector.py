# eos_selector.py
"""
Selects and initializes the appropriate Equation of State (EOS) model.
"""
import logging
from typing import Optional, Dict, Any
from eos import TabulatedEOS # Assuming TabulatedEOS is the primary implementation

logger = logging.getLogger(__name__)

# Define an EOS base class if strict typing/interface is desired, otherwise rely on duck typing
# from models import PhysicsModule # Example if EOS should conform to PhysicsModule

def select_eos(backend: str, table_file: Optional[str] = None, mixture_fractions: Optional[Dict[str, float]] = None, **kwargs: Any) -> Any:
    """
    Selects and returns an initialized Equation of State object.

    Args:
        backend (str): The type of EOS backend to use (e.g., 'tabulated').
        table_file (Optional[str]): Path to the HDF5 file for tabulated EOS.
        mixture_fractions (Optional[Dict[str, float]]): Fractions for mixture EOS (currently not implemented).
        **kwargs: Additional arguments specific to the EOS backend.

    Returns:
        An instance of the selected EOS class (e.g., TabulatedEOS).

    Raises:
        ValueError: If the backend is unknown or required parameters are missing.
        NotImplementedError: If a requested feature (like mixtures) is not implemented.
    """
    logger.info(f"Selecting EOS backend: {backend}")

    if backend == 'tabulated':
        if table_file is None:
            logger.error("Tabulated EOS backend selected, but 'table_file' not provided.")
            raise ValueError("Missing 'table_file' for tabulated EOS backend.")
        try:
            eos_instance = TabulatedEOS(filename=table_file)
            logger.info(f"Instantiated TabulatedEOS from file: {table_file}")

            if mixture_fractions is not None:
                logger.warning("Mixture fractions provided but not currently implemented in TabulatedEOS.")
                # raise NotImplementedError("Mixture EOS is not implemented yet.")

            return eos_instance
        except FileNotFoundError:
            logger.error(f"EOS table file not found: {table_file}")
            raise
        except Exception as e:
            logger.error(f"Failed to instantiate TabulatedEOS: {e}")
            raise

    # Add other EOS backends here if needed
    # elif backend == 'ideal_gas':
    #     try:
    #         from ideal_gas_eos import IdealGasEOS # Example
    #         eos_instance = IdealGasEOS(**kwargs)
    #         logger.info("Instantiated IdealGasEOS")
    #         return eos_instance
    #     except ImportError:
    #         logger.error("IdealGasEOS module not found.")
    #         raise ValueError("IdealGasEOS module required but not found.")
    #     except Exception as e:
    #         logger.error(f"Failed to instantiate IdealGasEOS: {e}")
    #         raise

    else:
        logger.error(f"Unknown EOS backend specified: {backend}")
        raise ValueError(f"Unknown EOS backend: {backend}")