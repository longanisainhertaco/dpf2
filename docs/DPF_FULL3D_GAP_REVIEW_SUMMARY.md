# Full-3D High-Fidelity DPF Simulator Gap Review

This document summarizes the current state of the DPF simulation prototype and outlines the path toward a production-grade, high-fidelity capability.

## Top Gaps
- **Circuit–plasma coupling:** current implementation lacks dynamic inductance and back-EMF terms.
- **MHD core:** only a placeholder resistive model exists; no Hall, radiation, or two-temperature physics.
- **Geometry & AMR:** code is restricted to idealized meshes without CAD import or refinement.
- **HPC & reproducibility:** no MPI/GPU support, limited I/O, and missing CI/testing workflow.

## Roadmap Highlights
1. **Foundations** – Develop a verified 2‑D resistive/Hall MHD solver with robust \(\nabla\cdot B\) control, coupled to a dynamic RLC circuit.
2. **Physics Completion** – Add tabular EOS, non‑LTE radiation, neutral/ablation models, and synthetic diagnostics validated against reference devices.
3. **Predictive Production** – Introduce 3‑D geometry with AMR, hybrid PIC pinch window, and full circuit–plasma–vacuum co-simulation producing neutron yield within a factor of two of experiment.

## Key Risks and Mitigations
| Risk | Mitigation |
|------|------------|
| Unstable magnetic divergence | Employ constrained transport or hyperbolic cleaning. |
| Radiation losses missing | Implement optically thin and gray radiation modules. |
| Lack of scaling | Adopt an MPI+GPU framework such as AMReX. |
| Validation data gaps | Establish data sharing with PF‑400 and PF‑1000 facilities. |

## Immediate Backlog
1. Implement 2‑D resistive MHD solver with HLLD + constrained transport.
2. Extend circuit model to include time-varying \(L\) and back‑EMF coupling.
3. Integrate tabular EOS and ionization models.
4. Set up continuous integration and dependency pinning.

This summary is derived from the gap analysis and roadmap produced during the review process and serves as a starting point for coordinated development.
