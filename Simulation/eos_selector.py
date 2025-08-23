# eos_selector.py
"""
Selects and initializes the appropriate Equation of State (EOS) model.
"""
import logging
from typing import Optional, Dict, Any, Union
from eos import TabulatedEOS  # Assuming TabulatedEOS is the primary implementation

logger = logging.getLogger(__name__)

# Define an EOS base class if strict typing/interface is desired, otherwise rely on duck typing
# from models import PhysicsModule # Example if EOS should conform to PhysicsModule

def _parse_mixture_fractions(mixture_fractions: Union[str, Dict[str, float]]) -> Dict[str, float]:
    """Parse mixture fraction input into a validated dictionary.

    Args:
        mixture_fractions (Union[str, Dict[str, float]]): Definition of mixture fractions
            either as a dictionary or as a comma separated string of
            ``species:fraction`` pairs.

    Returns:
        Dict[str, float]: Normalized mixture fraction dictionary.

    Raises:
        ValueError: If the provided definition is invalid or fractions do not
            sum to one.
        TypeError: If ``mixture_fractions`` is not a string or dictionary.
    """

    if mixture_fractions is None:
        return {}

    parsed: Dict[str, float] = {}

    if isinstance(mixture_fractions, str):
        for part in mixture_fractions.split(','):
            part = part.strip()
            if not part:
                continue
            try:
                species, frac = part.split(':')
            except ValueError as exc:  # not enough values to unpack
                raise ValueError(
                    "Mixture fractions must be in 'species:fraction' format"
                ) from exc
            species = species.strip()
            if not species:
                raise ValueError("Species name in mixture fractions cannot be empty")
            try:
                parsed[species] = float(frac)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid fraction value for species '{species}': {frac}"
                ) from exc
    elif isinstance(mixture_fractions, dict):
        for species, frac in mixture_fractions.items():
            species = str(species)
            try:
                parsed[species] = float(frac)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid fraction value for species '{species}': {frac}"
                ) from exc
    else:
        raise TypeError("mixture_fractions must be a dict or a string")

    if not parsed:
        raise ValueError("No valid mixture fractions provided")

    if any(frac < 0.0 for frac in parsed.values()):
        raise ValueError("Mixture fractions must be non-negative")

    total = sum(parsed.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Mixture fractions must sum to 1")

    return parsed


def select_eos(
    backend: str,
    table_file: Optional[str] = None,
    mixture_fractions: Optional[Union[str, Dict[str, float]]] = None,
    **kwargs: Any,
) -> Any:
    """
    Selects and returns an initialized Equation of State object.

    Args:
        backend (str): The type of EOS backend to use (e.g., 'tabulated').
        table_file (Optional[str]): Path to the HDF5 file for tabulated EOS.
        mixture_fractions (Optional[Union[str, Dict[str, float]]]): Fractions for
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

    if backend == 'tabulated':
        if table_file is None:
            logger.error("Tabulated EOS backend selected, but 'table_file' not provided.")
            raise ValueError("Missing 'table_file' for tabulated EOS backend.")
        try:
            parsed_mixture = _parse_mixture_fractions(mixture_fractions) if mixture_fractions is not None else None
            eos_instance = TabulatedEOS(filename=table_file, mixture_fractions=parsed_mixture)
            logger.info(f"Instantiated TabulatedEOS from file: {table_file}")
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