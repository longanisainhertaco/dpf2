# eos_selector.py
"""
Selects and initializes the appropriate Equation of State (EOS) model.
"""
import logging
from typing import Any, Optional, Union

from .eos import TabulatedEOS, parse_mixture_fractions
from dpf2.eos.ideal_gas import IdealGasEOS

logger = logging.getLogger(__name__)

def select_eos(
    backend: str,
    table_file: Optional[str] = None,
    mixture_fractions: Optional[Union[str, dict[str, float]]] = None,
    **kwargs: Any,
) -> Any:
    """
    Selects and returns an initialized Equation of State object.

    Args:
        backend (str): The type of EOS backend to use (e.g., 'tabulated').
        table_file (Optional[str]): Path to the HDF5 file for tabulated EOS.
        mixture_fractions (Optional[Union[str, dict[str, float]]]): Fractions for
            mixture EOS. Can be provided as a dictionary or as a comma separated
            string in the form ``"species1:frac1,species2:frac2"``.
        **kwargs: Additional arguments specific to the EOS backend.

    Returns:
        An instance of the selected EOS class (e.g., TabulatedEOS).

    Raises:
        ValueError: If the backend is unknown or required parameters are missing.
        NotImplementedError: If a requested feature (like mixtures) is not implemented.
    """
    logger.info(f"Selecting EOS backend: {backend}")

    fractions = None
    if mixture_fractions is not None:
        fractions = parse_mixture_fractions(mixture_fractions)

    if backend == 'tabulated':
        if table_file is None:
            logger.error("Tabulated EOS backend selected, but 'table_file' not provided.")
            raise ValueError("Missing 'table_file' for tabulated EOS backend.")

        try:
            eos_instance = TabulatedEOS(
                filename=table_file,
                mixture_fractions=fractions,
            )
            logger.info(f"Instantiated TabulatedEOS from file: {table_file}")
            return eos_instance
        except FileNotFoundError:
            logger.error(f"EOS table file not found: {table_file}")
            raise
        except Exception as e:
            logger.error(f"Failed to instantiate TabulatedEOS: {e}")
            raise

    elif backend == 'ideal_gas':
        try:
            eos_instance = IdealGasEOS(**kwargs)
            logger.info("Instantiated IdealGasEOS")
            return eos_instance
        except Exception as e:
            logger.error(f"Failed to instantiate IdealGasEOS: {e}")
            raise

    else:
        logger.error(f"Unknown EOS backend specified: {backend}")
        raise ValueError(f"Unknown EOS backend: {backend}")
