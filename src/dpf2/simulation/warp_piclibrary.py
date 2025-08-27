# warp_piclibrary.py
"""
Placeholder library for handling PIC collisions specifically within WarpX.

This module is intended to provide interfaces or handlers that integrate
custom or standard collision models with the WarpX particle data structures
and simulation loop.
"""

import logging
import math
import random
from statistics import mean
from typing import Any, Callable, Sequence, Tuple

# ``numpy`` is an optional dependency of the project.  The unit tests in this
# kata run without the real package, therefore the implementation below relies
# only on the Python standard library.  If ``numpy`` is available it can still
# be used transparently as the container objects returned by WarpX often expose
# list-like interfaces.
try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - fallback when numpy is absent
    np = None  # type: ignore

# We only import pywarpx lazily.  The real package is optional for the test
# environment and is not required for the light-weight mocks used here.  If it
# is available the handler will attempt to use the native MCC collision
# routines instead of the simplified Python implementation below.
try:  # pragma: no cover - optional dependency
    from pywarpx import picmi  # type: ignore
except Exception:  # pragma: no cover - the tests run without pywarpx installed
    picmi = None

logger = logging.getLogger(__name__)

class PICCollisionHandler:
    """High level interface for Monte Carlo collisions in WarpX.

    The handler encapsulates logic for querying particle information from a
    WarpX simulation object and applying Monte Carlo collision steps.  If a
    native WarpX collision routine is available (e.g., through
    ``pywarpx``/``picmi``) it will be used; otherwise a light‑weight Python
    implementation is executed.  The latter is sufficient for unit testing and
    simple examples.
    """

    def __init__(
        self,
        collision_freq_func: Callable,
        species_pairs: Sequence[Tuple[str, str]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a new collision handler.

        Parameters
        ----------
        collision_freq_func:
            Callable returning a collision frequency given plasma properties.
            The callable must accept at least ``ne`` and ``Te`` and may accept
            the keyword arguments ``species1`` and ``species2``.
        species_pairs:
            Optional sequence of pairs of species names to collide when
            :meth:`apply_collisions` is invoked without explicit pairs.
        **kwargs:
            Additional keyword arguments forwarded to ``collision_freq_func``.
        """

        self.collision_freq_func = collision_freq_func
        self.kwargs = kwargs
        self.species_pairs = list(species_pairs or [])

        logger.info(
            "PICCollisionHandler initialized with frequency function: %s",
            collision_freq_func,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def apply_collisions(
        self,
        warp_instance: Any,
        dt: float,
        species_pairs: Sequence[Tuple[str, str]] | None = None,
    ) -> None:
        """Apply collisions for the configured species pairs.

        Parameters
        ----------
        warp_instance:
            WarpX simulation object exposing particle containers.
        dt:
            Time step over which to apply collisions.
        species_pairs:
            Optional explicit list of species pairs.  When omitted the pairs
            supplied during initialisation are used.
        """

        pairs = list(species_pairs or self.species_pairs)
        for sp1, sp2 in pairs:
            self._apply_pair(sp1, sp2, warp_instance, dt)

    # ------------------------------------------------------------------
    # Implementation helpers
    # ------------------------------------------------------------------
    def _apply_pair(
        self, species1_name: str, species2_name: str, warp_instance: Any, dt: float
    ) -> None:
        """Apply one collision step between two species."""

        logger.info(
            "Applying collisions between %s and %s with dt=%s",
            species1_name,
            species2_name,
            dt,
        )

        # If the WarpX object exposes its own MCC collision routine we defer to
        # it.  This mirrors the behaviour of ``picmi.MCCCollision``.
        if hasattr(warp_instance, "do_mcc_collisions"):
            warp_instance.do_mcc_collisions(
                species1_name, species2_name, dt, self.collision_freq_func, **self.kwargs
            )
            return

        # --- Retrieve particle containers ----------------------------------------------
        if not hasattr(warp_instance, "get_particle_container"):
            raise AttributeError("WarpX instance missing 'get_particle_container'")

        try:
            species1 = warp_instance.get_particle_container(species1_name)
            species2 = warp_instance.get_particle_container(species2_name)
        except Exception as exc:  # pragma: no cover - defensive programming
            logger.error("Unsupported species requested: %s", exc)
            raise ValueError(
                f"Unknown species pair ({species1_name}, {species2_name})"
            ) from exc

        if not (
            hasattr(species1, "get_velocities") and hasattr(species2, "get_velocities")
        ):
            raise AttributeError("Particle containers must implement get_velocities")

        # Work with explicit ``float`` copies so we can safely modify the arrays
        v1 = [list(v) for v in species1.get_velocities()]
        v2 = [list(v) for v in species2.get_velocities()]

        # Optional: retrieve particle weights if WarpX exposes them
        w1 = list(species1.get_weights()) if hasattr(species1, "get_weights") else None
        w2 = list(species2.get_weights()) if hasattr(species2, "get_weights") else None

        if len(v1) == 0 or len(v2) == 0:
            logger.debug("One of the species has no particles; skipping collisions")
            return

        # --- Estimate plasma parameters -------------------------------------------------
        n1 = int(sum(w1)) if w1 is not None else len(v1)
        n2 = int(sum(w2)) if w2 is not None else len(v2)

        volume = None
        if hasattr(warp_instance, "get_volume"):
            try:
                volume = float(warp_instance.get_volume())
            except Exception:  # pragma: no cover - best effort
                volume = None
        elif hasattr(warp_instance, "volume"):
            volume = float(warp_instance.volume)

        if volume and volume > 0.0:
            ne = max(n1, n2) / volume
        else:  # pragma: no cover - logging path
            logger.warning(
                "WarpX instance missing volume; using particle count for density"
            )
            ne = float(max(n1, n2))

        k_B = 1.380649e-23  # Boltzmann constant
        m1 = getattr(species1, "mass", 1.0)
        def norm(vec):
            return math.sqrt(sum(comp * comp for comp in vec))

        speeds1 = [norm(v) for v in v1]
        Te = m1 * mean(s * s for s in speeds1) / (3.0 * k_B)

        # Collision frequency and probability
        freq = float(
            self.collision_freq_func(
                ne,
                Te,
                species1=species1_name,
                species2=species2_name,
                **self.kwargs,
            )
        )
        if freq < 0:
            raise ValueError("Collision frequency must be non-negative")
        prob = max(0.0, min(1.0, 1.0 - math.exp(-freq * dt)))
        if prob <= 0.0:
            logger.debug("Zero collision probability; skipping")
            return

        # --- Determine colliding pairs --------------------------------------------------
        num_pairs = int(min(n1, n2))
        if num_pairs == 0:
            return
        rand = [random.random() for _ in range(num_pairs)]
        colliding = [i for i, r in enumerate(rand) if r < prob]

        if colliding:
            m2 = getattr(species2, "mass", 1.0)

            def random_dir() -> list[float]:
                while True:
                    vec = [random.gauss(0.0, 1.0) for _ in range(3)]
                    n = norm(vec)
                    if n > 0:
                        return [c / n for c in vec]

            for idx in colliding:
                dir_rel = random_dir()
                rel_v = [v1[idx][j] - v2[idx][j] for j in range(3)]
                rel_speed = norm(rel_v)
                v_cm = [
                    (m1 * v1[idx][j] + m2 * v2[idx][j]) / (m1 + m2)
                    for j in range(3)
                ]
                new_rel = [dir_rel[j] * rel_speed for j in range(3)]
                v1[idx] = [
                    v_cm[j] + (m2 / (m1 + m2)) * new_rel[j] for j in range(3)
                ]
                v2[idx] = [
                    v_cm[j] - (m1 / (m1 + m2)) * new_rel[j] for j in range(3)
                ]

        # --- Write updated velocities back ----------------------------------------------
        if not (
            hasattr(species1, "set_velocities") and hasattr(species2, "set_velocities")
        ):
            raise AttributeError("Particle containers must implement set_velocities")

        species1.set_velocities(v1)
        species2.set_velocities(v2)

    def setup_warpx_collisions(self, warp_instance: Any, species_pairs: list):
        """
        Sets up collision interactions within the WarpX simulation environment.

        This might involve configuring WarpX's internal collision modules.

        Args:
            warp_instance (Any): The WarpX simulation object.
            species_pairs (list): A list of tuples, where each tuple contains the names
                                   of two species that should collide, e.g., [('electrons', 'ions')].
        """
        logger.info(f"Setting up WarpX collisions for pairs: {species_pairs}")

        for sp1, sp2 in species_pairs:
            try:
                if hasattr(warp_instance, "add_collision_operator"):
                    try:
                        warp_instance.add_collision_operator(
                            sp1, sp2, self.collision_freq_func, self.kwargs
                        )
                    except TypeError:
                        warp_instance.add_collision_operator(
                            sp1, sp2, self.collision_freq_func, **self.kwargs
                        )
                else:
                    import picmi  # type: ignore

                    coll = picmi.MCCCollision(
                        name=f"coll_{sp1}_{sp2}",
                        species=[warp_instance.species[sp1], warp_instance.species[sp2]],
                        **self.kwargs,
                    )
                    if hasattr(warp_instance, "add_collision"):
                        warp_instance.add_collision(coll)
                    else:
                        warp_instance.collisions = getattr(warp_instance, "collisions", [])
                        warp_instance.collisions.append(coll)
            except Exception as e:  # pragma: no cover - logging path
                logger.warning(
                    f"Failed to add WarpX collision between {sp1} and {sp2}: {e}"
                )

# --- Example Usage Pattern (as inferred from collision_model.py) ---
# Assuming 'ne', 'Te', 'Z' are numpy arrays or floats
# def nu_ei_spitzer(ne, Te, lnL=10.0, Z=1.0):
#     # ... calculation ...
#     return calculated_frequency

# In collision_model.py:
# handler = PICCollisionHandler(lambda ne, Te, Z=1.0: nu_ei_spitzer(ne, Te, Z))

# In the PIC solver loop (e.g., within WarpXWrapper.step):
# collision_handler.apply_collisions('electrons', 'ions', self.warp, self.dt)
