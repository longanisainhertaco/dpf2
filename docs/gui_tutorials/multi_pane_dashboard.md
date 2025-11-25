# Multi-pane regime dashboard

The GUI dashboard now streams **four synchronized panes**: discharge current,
bank voltage, sheath radius, and a dimensionless **regime gate** that blends a
Lundquist surrogate with the magnetisation product \(\omega_c \tau_e\).

Use the guided steps in the *Tutorial Flow* overlay to replay the scenarios
below. Each step highlights the physics signal, the expected numerical
implication, and what to check in the plots:

1. **Charge and breakdown** – a fast voltage ramp inflates the sheath. The gate
   stays below unity, signalling collisional breakdown where diffusive
   resistivity dominates and first-order advection is stable.
2. **Rundown and compression** – higher pressure slows the current crest while
   narrowing the WebGL sheath. The gate climbs as the sheath magnetises; tighten
   Courant safety factors to maintain CFL limits.
3. **Pinch optimisation** – voltage and pressure push the sheath into the
   pinch. The regime line crosses the dashed threshold, indicating collisionless
   motion and the need for higher-order MHD fluxes and Hall terms.
4. **Afterglow** – both drive knobs drop. The gate falls and the current tail
   aligns with the sheath pane, signalling a return to fluid-valid dynamics.

### Export and reproducibility

Click *Save snapshot* in the GUI to capture the four-pane state alongside the
Regime Dashboard indicators. Each save writes both a PNG and a JSON manifest
with the slider settings, making it simple to compare current/voltage/sheath
traces across runs.
