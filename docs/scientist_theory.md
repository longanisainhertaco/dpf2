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

## API Highlights

The [API reference](api_reference.md) details programmatic access. Key modules:

- [`dpf2.physics`](api_reference.md#dpf2.physics): primitive variables and EOS.
- [`dpf2.solvers`](api_reference.md#dpf2.solvers): time integration schemes.
- [`dpf2.diagnostics`](api_reference.md#dpf2.diagnostics): data extraction and analysis.

These interfaces allow scientists to probe fields, currents, and reaction rates
across the simulation domain.
