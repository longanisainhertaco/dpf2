from typing import Dict, Callable, Optional

from .state import ComponentMaterialState


class MaterialDamageModel:
    """Minimal material damage model.

    Updates component material states from surface fluxes and temperatures
    provided by a solver.  Impurities may be injected back into a plasma
    model if it exposes an ``inject_impurities`` method.
    """

    def __init__(
        self,
        components: Dict[str, ComponentMaterialState],
        plasma_model: Optional[object] = None,
    ) -> None:
        self.components = components
        self.plasma_model = plasma_model

    def apply(self, solver: object, dt: float) -> None:
        """Update material states for one timestep.

        The solver may expose optional hooks:

        ``surface_flux(name)`` -> incident particle flux for sputtering
        ``surface_temperature(name)`` -> surface temperature
        ``deposition_flux(name)`` -> redeposition flux from plasma
        ``evaporation_rate(name, temperature)`` -> mass loss rate from evaporation
        """

        flux_fn: Callable[[str], float] = getattr(solver, "surface_flux", lambda name: 0.0)
        temp_fn: Callable[[str], float] = getattr(solver, "surface_temperature", lambda name: 300.0)
        dep_fn: Callable[[str], float] = getattr(solver, "deposition_flux", lambda name: 0.0)
        evap_fn: Callable[[str, float], float] = getattr(
            solver, "evaporation_rate", lambda name, temp: 0.0
        )

        # First pass: handle local sputtering/evaporation and record mass to redistribute
        redistributed: Dict[str, float] = {}
        for name, state in self.components.items():
            flux = flux_fn(name)
            temperature = temp_fn(name)
            state.record_temperature(temperature)

            sputtered = flux * state.material.sputter_yield * dt
            redep = dep_fn(name) * dt
            evap = evap_fn(name, temperature) * dt

            net_sputter = max(sputtered - redep, 0.0)
            state.erode(net_sputter + evap)
            if redep > 0.0:
                state.redeposit(redep)

            redistributed[name] = net_sputter

            if self.plasma_model and hasattr(self.plasma_model, "inject_impurities"):
                try:
                    if redep > 0.0:
                        self.plasma_model.inject_impurities(name, -redep)
                    if evap > 0.0:
                        self.plasma_model.inject_impurities(name, evap)
                except Exception:
                    pass

        # Second pass: distribute net sputtered material as contamination films
        ncomp = len(self.components)
        if ncomp > 1:
            for src_name, amount in redistributed.items():
                if amount <= 0.0:
                    continue
                share = amount / (ncomp - 1)
                for dest_name, state in self.components.items():
                    if dest_name == src_name:
                        continue
                    state.deposit(share)
