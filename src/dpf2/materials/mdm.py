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
        flux_fn: Callable[[str], float] = getattr(solver, "surface_flux", lambda name: 0.0)
        temp_fn: Callable[[str], float] = getattr(solver, "surface_temperature", lambda name: 300.0)

        for name, state in self.components.items():
            flux = flux_fn(name)
            temperature = temp_fn(name)
            state.record_temperature(temperature)

            erosion = flux * state.material.sputter_yield * dt
            if erosion > 0.0:
                state.erode(erosion)
                if self.plasma_model and hasattr(self.plasma_model, "inject_impurities"):
                    try:
                        self.plasma_model.inject_impurities(name, erosion)
                    except Exception:
                        pass
