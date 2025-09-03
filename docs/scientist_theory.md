# Scientific Theory and API Links

Researchers can use DPF2 to explore magneto-hydrodynamic behavior in dense
plasma focus devices. The core solver integrates resistive MHD equations with
radiation transport and kinetics models.

## Governing Equations

The simulation evolves density $\rho$, velocity $\vec{u}$, and magnetic field
$\vec{B}$ under the conservation laws:

- Mass: $\partial_t \rho + \nabla \cdot (\rho\vec{u}) = 0$
- Momentum: $\partial_t (\rho\vec{u}) + \nabla \cdot (\rho\vec{u}\vec{u}) = -\nabla p + \vec{J}\times\vec{B}$
- Energy: includes Joule heating, radiation, and fusion source terms.

Additional models handle ionization, fusion yields, and radiation transport.

## Braginskii Transport

In magnetized plasmas the transport of particles, momentum, and energy is highly
anisotropic. Braginskii derived a fluid closure that expands the distribution
function in powers of the small parameter $\omega_{c}\tau$. The resulting
coefficients describe cross\-field viscosity, parallel heat conduction, and
resistivity:

- The parallel heat flux follows $\vec{q}_{\parallel}=-\kappa_{\parallel}
  \nabla_{\parallel} T$ with $\kappa_{\parallel} \gg \kappa_{\perp}$.
- Viscous stresses couple to velocity gradients through species\-dependent
  tensors, capturing gyroviscous effects.
- Finite resistivity produces magnetic field diffusion on the scale
  $\eta \nabla^{2}\vec{B}$ and sets the Lundquist number.

These closures are implemented in DPF2 to capture collisional transport in the
pinch column.

## Instability Physics

Dense plasma focus devices are prone to macroscopic instabilities that shape the
current sheath and pinch. The code tracks perturbations associated with common
magneto\-hydrodynamic modes:

- **Sausage ($m=0$)** instabilities modulate the radius and can trigger
  localized density spikes.
- **Kink ($m=1$)** modes bend the column and may lead to beam formation.
- **Rayleigh–Taylor** growth occurs when the accelerating sheath pushes against
  the denser plasma, characterized by $\gamma \sim \sqrt{Akg}$.

Linear growth rates and nonlinear saturation are available through diagnostic
outputs so researchers can relate simulation results to experimental behavior.

## API Highlights

The [API reference](api_reference.md) details programmatic access. Key modules:

- [`dpf2.physics`](api_reference.md#dpf2.physics): primitive variables and EOS.
- [`dpf2.solvers`](api_reference.md#dpf2.solvers): time integration schemes.
- [`dpf2.diagnostics`](api_reference.md#dpf2.diagnostics): data extraction and analysis.

These interfaces allow scientists to probe fields, currents, and reaction rates
across the simulation domain.
