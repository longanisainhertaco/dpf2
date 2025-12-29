# Hybrid Core

This package separates fluid and kinetic species handlers to support hybrid-PIC workflows.

- `fluid/`: Fluid electron and bulk plasma models (MHD closures, Ohm's law variants, resistivity models).
- `kinetic/`: Ion kinetic modules (particle pushers, collision operators, fusion reaction sampling).

Each subpackage is intended to remain decoupled except through clearly defined field/current exchange interfaces to support Energy Conserving Semi-Implicit Methods (ECSIM) and explicit PIC back-ends.
