# DPF2 High-Performance Simulation Roadmap

This document outlines possible steps to evolve **dpf2** from the current proof-of-concept code into a comprehensive dense plasma focus (DPF) simulation framework.  The goal is to match or exceed existing high-level codes by adopting proven practices from projects such as WarpX and AMReX.

## 1. Core Simulation Engine

### Electromagnetic Solvers and Particle Pushers
- Implement modular Maxwell solvers (Yee, CKC, PSATD) with options for finite-difference or spectral discretisations.
- Provide a choice of particle pushers: Boris, Vay and Higuera–Cary algorithms are recommended for relativistic accuracy.
- Use charge-conserving current deposition (e.g. Esirkepov scheme) to maintain numerical stability.  Pair these with divergence-cleaning techniques or Maxwell solvers that conserve \(\nabla\cdot B=0\) by construction.

### Multi-Physics Models
- **MHD module**: support ideal and resistive MHD.  Include full Braginskii transport coefficients (anisotropic viscosity and thermal conduction).  Implement the stress tensor using a closure that naturally reduces to the ideal form at high conductivity.  Add shock-capturing methods (e.g. HLLD) and CT/FLUX-CT for magnetic divergence control.
- **PIC module**: support advanced Monte Carlo collision models for neutral and charged particles (Nanbu or Takizuka–Abe).  Implement field ionisation models (ADK, PPT, BSI) with Monte Carlo selection of ionisation events.
- **Hybrid module**: kinetic ions (PIC) with a fluid electron model.  Couple species via a generalised Ohm's law containing Hall term and resistivity.  Electron pressure and energy may be solved with a fluid energy equation or obtained from an EOS.

## 2. DPF Specific Physics

### Equation of State (EOS)
- Interface with tabulated EOS libraries (SESAME or FPEOS).  Provide interpolation in temperature–density space and allow table selection via the configuration file.
- Implement analytic models for warm dense matter (e.g. average-atom).  The EOS module should expose pressure, internal energy and derivatives to both MHD and kinetic solvers.

### Collisional–Radiative Model (CRM)
- Track multiple ionisation stages and include collisional ionisation, excitation, radiative and three-body recombination, auto-ionisation and Auger processes.  Manage atomic data tables (cross-sections, rate coefficients) through a lightweight database.

### Radiation Transport
- Provide bremsstrahlung, line and recombination radiation sources.  Implement transport algorithms selectable at run-time: FLD/M1 for optically thick, Monte Carlo or escape probability for thin regimes.  Support frequency groups with tabulated opacities.

### Transport Coefficients
- Include Braginskii coefficients for viscosity and thermal conduction; support temperature and density dependent collision frequencies for warm dense conditions.  Allow user-defined anomalous resistivity/viscosity models to represent micro-instabilities.

### Axial and Poloidal Magnetic Fields
- Full 3‑D geometry should be supported.  Seed fields can be specified or evolved self-consistently.  Capture their effect on ion acceleration and neutron production.

## 3. Numerical Methods and HPC Integration

### Adaptive Mesh Refinement
- Adopt a block-structured AMR similar to AMReX.  Refinement criteria may include current density, magnetic field gradients and plasma density.  Refinement and coarsening should be fully dynamic.

### Parallelisation and Performance
- Design for MPI based domain decomposition with optional GPU acceleration via CUDA, HIP or SYCL.  Leverage AMReX for backend abstractions where possible.  Apply load balancing across AMR levels and particle populations.

### Boundary Conditions
- Provide absorbing and perfectly conducting boundary conditions for fields (e.g. PML or PEC).  Particle boundaries should support reflection, absorption or user-defined scripts.

## 4. Software Architecture

- Organise the code into clearly separated modules: numerics (grid, solvers), physics models (MHD, PIC, ionisation, radiation), I/O and diagnostics.  Use C++ for performance-critical kernels, Python bindings for scripting and high-level orchestration, and Fortran modules if legacy models are reused.
- Define explicit APIs for data exchange between modules; for example, field arrays may follow the openPMD standard.
- Configuration files can be JSON/YAML with optional Python scripting à la PICMI to allow advanced users to construct complex setups.

### Diagnostics and Output
- Full diagnostics: openPMD plotfiles containing fields \(E,B,J,\rho\) and particle information (positions, momenta, weights, IDs).
- Reduced diagnostics: global energy balances, spectra, probe signals.  Provide hooks for custom analysis.
- Boundary particle diagnostics to record species and energies striking electrodes.  Field probes should be configurable at arbitrary points.

## 5. DPF‑Specific Output

### Neutron Yield
- Compute yields from beam–target and thermonuclear reactions.  Use tabulated cross‑sections for D–D and D–T.  Separate contributions by tracking fast ion distributions and thermal reaction rates.

### X‑ray Emission
- Compute bremsstrahlung from thermal and beam electrons.  Include line emission from impurities and anode materials.  Couple this module to the radiation transport system for reabsorption and spectral formation.

### Synthetic Diagnostics
- Provide synthetic signals mimicking probes, interferometers, neutron ToF and imaging diagnostics.  Instrument response functions can be supplied as user input for each detector type.

## 6. Advanced Features

- Support embedded boundaries to model complex electrode geometries.  A simple sputtering/erosion model can feed impurity sources into the plasma.
- Expose APIs for future machine‑learning integration (e.g. surrogate models for expensive physics or data-driven optimisation).
- Allow coupling to external circuit solvers beyond simple RLC models by exchanging time-dependent impedance and currents.

## 7. GUI Integration Considerations

- Output data should be written in a format easily consumed by a Unity-based GUI (openPMD/HDF5).  Expose a lightweight WebSocket or shared-memory interface to stream reduced diagnostics for real-time visualisation.
- Provide Python bindings for all configuration parameters so that GUI front‑ends can construct input files programmatically.

---
This roadmap is intended as a high-level guide.  Detailed implementation should proceed iteratively with validation against benchmark cases and comparison to experimental data.
