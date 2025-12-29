# Gap Analysis: Chicago vs. Ansys Charge Plus vs. DPF-NextGen

## Overview
- **Chicago (Sandia)**: Hybrid PIC with fluid electrons and explicit/implicit ion pushers. Relies on CUBIT for block-structured meshing and manual problem setup. Strong heritage in HEDP but limited CAD-native workflows and requires explicit particle push for stability at sub-ns time steps.
- **Ansys Charge Plus**: Commercial electrostatic PIC with Energy Conserving Semi-Implicit Method (ECSIM) solver, native CAD ingestion, and multiphysics coupling. Optimized for electronic device and plasma processing regimes; limited transparency around fusion extensions.
- **DPF-NextGen Goal**: Achieve ion-kinetic fidelity on par with Chicago while adopting CAD-native, automated meshing and ECSIM time-stepping to shorten setup and extend stable step sizes into the microsecond regime relevant to DPF discharge evolution.

## Focus Area Findings

### Meshing
- **Chicago (CUBIT block meshing)**: Uses structured blocks with manual partitioning and entity tagging. Suitable for canonical coaxial geometries but burdensome for complex electrode/insulator assemblies and parametric sweeps. Geometry fidelity depends on manual defeaturing and re-blocking.
- **Ansys Charge Plus (voxel/cut-cell automation)**: Automates voxelization and cut-cell generation directly from CAD, enabling rapid turnaround and consistent boundary representation across design variants. Reduced need for meshing expertise, and cut-cell methods preserve embedded boundary fidelity.
- **Gap / Requirement for DPF-NextGen**: Provide CAD-native ingest (OpenCASCADE) and automated voxel/cut-cell meshing to cut human meshing effort by ~90% while maintaining fidelity for tight electrode gaps and gas feeds. Must expose geometry provenance for reproducibility and uncertainty quantification.

### Time-Stepping
- **Chicago (explicit/implicit hybrid PIC)**: Fluid electrons with explicit Boris ion pusher typically constrained by electron plasma frequency and sheath resolution, forcing sub-ns steps. Implicit options reduce stiffness but often compromise energy conservation and can require nonlinear solves per step.
- **Ansys Charge Plus (ECSIM)**: ECSIM splits particle advance and field solve such that discrete energy is conserved without resolving fastest plasma oscillations. Semi-implicit field update damps high-frequency numerical instabilities, enabling stable steps orders of magnitude larger than explicit Boris while maintaining correct momentum exchange.
- **Gap / Requirement for DPF-NextGen**: Adopt ECSIM for ion kinetics with fluid-electron closures. Target microsecond-scale steps during current rise/fall to capture macroscopic DPF evolution without resolving every electron plasma oscillation, retaining energy conservation and acceptable phase accuracy.

### Fusion Physics
- **Current state (commercial/legacy)**: Commercial PIC tools rarely expose validated p–B¹¹ fusion models; cross-sections often limited to D–T and D–D. Chicago heritage emphasizes D–D/D–T; p–B¹¹ options (with quantum corrections) are typically absent.
- **Gap / Requirement for DPF-NextGen**: Implement quantum-corrected p–B¹¹ cross-sections (e.g., including electron screening and resonant contributions) and sampling for hybrid-PIC reactions. Provide transparent data provenance and sensitivity knobs for burn-rate studies.

## Implications for DPF-NextGen Roadmap
1. **CAD-first pipeline**: Integrate STEP import via OpenCASCADE and automated voxel/cut-cell meshing to lower setup time by ~90% compared to manual CUBIT block meshing.
2. **ECSIM-based hybrid core**: Couple ECSIM ion pushers with fluid-electron closures to achieve stable microsecond-scale steps while conserving energy, surpassing explicit Boris limits.
3. **Fusion model completeness**: Ship validated p–B¹¹ cross-sections and sampling utilities as first-class citizens, addressing a key physics gap in both legacy and commercial tools.
