# Executive Summary
The current DPF2 prototype is a Python-based toy model that couples a fixed-parameter RLC circuit to analytic pinch surrogates. It lacks 3-D geometry, fluid or kinetic plasma solvers, radiation, or realistic transport. Nearly all capabilities required for a predictive high-fidelity DPF simulator are missing. Primary risks are the absence of a scalable multiphysics core and limited validation data. The shortest path to a first credible 3‑D result is to implement a resistive Hall‑MHD module with AMR and circuit coupling on simplified coaxial geometry, followed by incremental addition of radiation, EOS tables, and synthetic diagnostics. Assumptions: no existing CAD, experimental data, or parallel infrastructure are provided.

## Rapid Inventory
| # | Target Capability | Status |
|---|-------------------|--------|
|1|Circuit–plasma co-simulation with mutual coupling|Missing|
|2|Resistive/Hall/Nernst two-temperature radiation-MHD|Missing|
|3|Ionization/recombination & tabular EOS|Missing|
|4|Radiation transport (optically-thin + gray/multigroup)|Missing|
|5|Neutrals & material ablation|Missing|
|6|Sheath & snowplow physics with instabilities|Missing|
|7|Neutron yield via beam–target + thermonuclear|Partial|
|8|Hybrid kinetic/PIC closure|Missing|
|9|Vacuum EM & return path|Missing|
|10|High-order shock-capturing with CT/hyperbolic cleaning|Missing|
|11|AMR resolving 10–100 µm|Missing|
|12|Implicit/IMEX solvers with MPI+GPU|Missing|
|13|Robust BCs (conducting/open/sheath)|Missing|
|14|Reproducible I/O, restart, provenance|Partial|
|15|Synthetic diagnostics suite|Partial|
|16|Cross-validation against ≥2 facilities|Missing|

## Gap Analysis by Domain
- **Circuit–Plasma**: only static RLC circuit; no dynamic inductance or back‑EMF. *Solution*: develop coupled circuit module using PETSc ODE with dL/dt from plasma state.
- **MHD/Transport**: no fluid solver. *Solution*: implement 3‑D resistive Hall‑MHD (see new `HallMHDSolver` skeleton) with Braginskii transport and CT.
- **Radiation**: configuration schema exists but no solver. *Solution*: start with optically thin loss module; later gray FLD or M1.
- **EOS/Conductivity**: only ideal gas; no ionization. *Solution*: integrate SESAME/LEOS tables with Saha/CR model.
- **Neutrals/Ablation**: absent. *Solution*: add neutral gas dynamics and surface ablation models.
- **Sheath/Snowplow**: analytic pinch ignores sheath. *Solution*: include 1‑D shock-capturing snowplow phase and electrode sheath boundary models.
- **Instabilities/Subgrid**: none. *Solution*: resolve m=0/m=1 with 3‑D AMR; develop anomalous resistivity closure.
- **Kinetic/PIC-Hybrid**: none. *Solution*: embed PIC window (WarpX-style) or subgrid model for fast ions.
- **Vacuum EM**: missing. *Solution*: couple to FDTD vacuum solver or boundary impedance model.
- **Geometry/Mesh/AMR**: no CAD or meshing. *Solution*: import STL/STEP with embedded-boundary AMR (AMReX).
- **Numerics/Solvers**: no Riemann solvers, ∇·B control, or implicit sources. *Solution*: HLLD + CT, IMEX-ARK with PETSc.
- **HPC/Performance**: purely serial Python. *Solution*: C++/CUDA kernels via AMReX; MPI parallelism.
- **I/O/Repro**: JSON results only; no restart or provenance. *Solution*: HDF5/openPMD outputs with checkpoints.
- **Synthetic Diagnostics**: partial (neutron yield, simple outputs). *Solution*: implement interferometry, XRD, B-dot, neutron TOF synthetic probes.
- **V&V/UQ**: none. *Solution*: assemble experimental datasets and add UQ framework (Dakota/UQTk).
- **DevOps/QA**: unit tests exist but lack CI and coverage. *Solution*: set up GitHub Actions, code style checks, coverage.
- **UX/Workflow**: basic CLI; no scan or optimization framework. *Solution*: add parameter sweep API and hooks for optimizers.

## Risk Register
| Risk | Likelihood | Impact | Early Signal | Mitigation |
|------|------------|--------|-------------|------------|
|AMR Hall‑MHD solver fails to converge|Medium|High|Divergent residuals in first 3-D runs|Use verified libraries (AMReX) and start with 2-D tests|
|Lack of experimental data for validation|High|High|Cannot match published scaling|Establish collaborations, compile open datasets|
|Performance shortfall on GPUs|Medium|High|Scaling <50% on multi-GPU|Profile early; leverage existing GPU kernels|
|Radiation model stiffness|Medium|Medium|Time steps collapse with radiation enabled|Use IMEX schemes and operator splitting|
|Project scope creep|High|Medium|Milestones slip|Phased roadmap with gated reviews|

## Roadmap
### Foundations (Months 0‑6)
- **Milestones**: minimal 3-D coaxial geometry; Hall‑MHD solver with CT; RLC circuit coupling.
- **Exit Criteria**: stable 3‑D rundown/pinch without radiation; ∇·B < 1e‑6 relative; current trace within 20% of analytic.

### Physics Completion (Months 6‑18)
- **Milestones**: add tabular EOS & ionization, optically thin radiation, AMR, synthetic diagnostics, begin PIC window.
- **Exit Criteria**: first Hall‑MHD pinch with AMR; non‑LTE radiation in 2-D slice; synthetic neutron yield within ×3 of reference.

### Predictive Production (Months 18‑36)
- **Milestones**: full non‑LTE radiation, hybrid PIC, vacuum EM, full facility validation, UQ pipeline.
- **Exit Criteria**: match I(t) & dI/dt within 10%, neutron yield ×2 accuracy, strong scaling ≥70% to 64 GPUs.

## Compute & Data Budget
Assumptions: 3-D AMR run with 10^8 cells, 5 species, 100 ns physical time.
- **Single 3-D run**: ~200 GPU-hrs, 0.5 TB memory, 1 TB storage.
- **V&V campaign (10 shots)**: ~2000 GPU-hrs, 5 TB storage.
- **100-shot param scan**: ~20,000 GPU-hrs, 50 TB storage.

## Acceptance Tests & Metrics
- Match I(t) and dI/dt within ±10% near pinch.
- Reproduce pinch time and radius within ±15%.
- Predict neutron yield and X-ray timing within ×2.
- Recover Y_n vs I_peak scaling across pressure/geometry cases.
- m=0/m=1 onset timing within ±20%.
- Strong scaling ≥70% on N nodes; ≥2× GPU speedup vs CPU.

## Actionable Backlog
| Item | Owner Role | Skills | Est. (pw) | Depends On |
|------|-----------|--------|-----------|------------|
|Implement Hall‑MHD solver with CT|Computational physicist|MHD, AMReX, CUDA|12|None|
|Circuit–plasma coupling with variable inductance|Electrical engineer|Circuit modeling, PETSc|6|Hall‑MHD solver|
|Integrate SESAME EOS & Saha ionization|Plasma physicist|EOS tables, CR modeling|8|Hall‑MHD solver|
|Optically thin radiation losses|Plasma physicist|Radiation transport|4|EOS integration|
|AMR infrastructure and mesh import|HPC engineer|AMReX, CAD|10|Hall‑MHD solver|
|Synthetic diagnostics module|Diagnostics expert|Signal modeling|6|AMR infrastructure|
|Validation dataset ingestion|Data scientist|Data pipelines|3|Diagnostics module|
|Hybrid PIC window|Kinetic specialist|PIC, WarpX|12|Hall‑MHD + AMR|
|UQ and parameter sweep tools|Applied mathematician|UQ, workflow mgmt|5|Diagnostics & data ingest|

## JSON Summary
```json
{
  "inventory": [
    {"domain": "circuit_plasma", "target_capability": "Circuit-plasma mutual coupling", "status": "Missing", "impact": "High", "complexity": "Medium", "recommended_solution": "Time-varying inductance coupled via PETSc ODE", "dependencies": []},
    {"domain": "mhd_twofluid_transport", "target_capability": "Hall-MHD with anisotropic transport", "status": "Missing", "impact": "High", "complexity": "High", "recommended_solution": "Implement AMReX-based solver with CT", "dependencies": []},
    {"domain": "radiation", "target_capability": "Optically thin + gray radiation", "status": "Missing", "impact": "Medium", "complexity": "Medium", "recommended_solution": "Add thin-loss module then FLD/M1", "dependencies": ["mhd_twofluid_transport"]},
    {"domain": "eos_conductivity", "target_capability": "Tabular EOS with ionization", "status": "Missing", "impact": "High", "complexity": "Medium", "recommended_solution": "Integrate SESAME tables and Saha/CR", "dependencies": ["mhd_twofluid_transport"]},
    {"domain": "neutrals_ablation", "target_capability": "Neutral gas & ablation", "status": "Missing", "impact": "Medium", "complexity": "High", "recommended_solution": "Neutral fluid + surface source terms", "dependencies": ["mhd_twofluid_transport"]},
    {"domain": "instabilities_closures", "target_capability": "m=0/m=1 instability capture", "status": "Missing", "impact": "High", "complexity": "High", "recommended_solution": "3-D AMR with anomalous resistivity", "dependencies": ["mhd_twofluid_transport", "geometry_mesh_amr"]},
    {"domain": "kinetic_hybrid", "target_capability": "Hybrid PIC window", "status": "Missing", "impact": "Medium", "complexity": "High", "recommended_solution": "Embed WarpX PIC region", "dependencies": ["mhd_twofluid_transport"]},
    {"domain": "vacuum_em", "target_capability": "3-D vacuum EM return path", "status": "Missing", "impact": "Medium", "complexity": "Medium", "recommended_solution": "FDTD/FEM coupling", "dependencies": []},
    {"domain": "geometry_mesh_amr", "target_capability": "3-D CAD import with AMR", "status": "Missing", "impact": "High", "complexity": "High", "recommended_solution": "AMReX embedded-boundary mesh", "dependencies": []},
    {"domain": "numerics_solvers", "target_capability": "HLLD + CT + IMEX", "status": "Missing", "impact": "High", "complexity": "High", "recommended_solution": "PETSc/ARK solvers", "dependencies": ["mhd_twofluid_transport"]},
    {"domain": "hpc_perf", "target_capability": "MPI+GPU scaling", "status": "Missing", "impact": "High", "complexity": "High", "recommended_solution": "AMReX GPU backend", "dependencies": ["numerics_solvers"]},
    {"domain": "io_repro", "target_capability": "HDF5/openPMD I/O", "status": "Partial", "impact": "Medium", "complexity": "Medium", "recommended_solution": "Add HDF5 checkpoints and provenance", "dependencies": []},
    {"domain": "synthetic_diags", "target_capability": "Comprehensive synthetic diagnostics", "status": "Partial", "impact": "Medium", "complexity": "Medium", "recommended_solution": "Implement interferometry, XRD, B-dot, neutron TOF", "dependencies": ["mhd_twofluid_transport"]},
    {"domain": "vv_uq", "target_capability": "Facility cross-validation and UQ", "status": "Missing", "impact": "High", "complexity": "Medium", "recommended_solution": "Collect data and integrate UQ toolkit", "dependencies": ["synthetic_diags"]},
    {"domain": "devops_qa", "target_capability": "CI and coverage", "status": "Partial", "impact": "Medium", "complexity": "Low", "recommended_solution": "GitHub Actions with pytest/coverage", "dependencies": []},
    {"domain": "ux_workflow", "target_capability": "Parameter scans & optimization", "status": "Missing", "impact": "Low", "complexity": "Medium", "recommended_solution": "Sweep/optimization API", "dependencies": []}
  ],
  "radar_scores": {
    "circuit_plasma": 0,
    "mhd_twofluid_transport": 0,
    "radiation": 0,
    "eos_conductivity": 0,
    "neutrals_ablation": 0,
    "instabilities_closures": 0,
    "kinetic_hybrid": 0,
    "vacuum_em": 0,
    "geometry_mesh_amr": 0,
    "numerics_solvers": 0,
    "hpc_perf": 0,
    "io_repro": 2,
    "synthetic_diags": 2,
    "vv_uq": 0,
    "devops_qa": 1,
    "ux_workflow": 0
  },
  "roadmap": [
    {"phase": "Foundations", "milestone": "Hall-MHD core with circuit coupling", "exit_criteria": "Stable 3-D rundown"},
    {"phase": "Physics Completion", "milestone": "Radiation, EOS tables, AMR, synthetic diagnostics", "exit_criteria": "Hall-MHD pinch with radiation"},
    {"phase": "Predictive Production", "milestone": "Hybrid PIC, vacuum EM, full validation", "exit_criteria": "Shot replay with neutron yield ×2"}
  ],
  "risks": [
    {"risk": "AMR Hall-MHD solver fails to converge", "likelihood": "Medium", "impact": "High", "signal": "Divergent residuals", "mitigation": "Use AMReX & start 2-D"},
    {"risk": "Lack of experimental data", "likelihood": "High", "impact": "High", "signal": "No scaling match", "mitigation": "Acquire datasets"},
    {"risk": "GPU performance shortfall", "likelihood": "Medium", "impact": "High", "signal": "<50% scaling", "mitigation": "Profile & optimize kernels"},
    {"risk": "Radiation stiffness", "likelihood": "Medium", "impact": "Medium", "signal": "Time step collapse", "mitigation": "IMEX & splitting"},
    {"risk": "Scope creep", "likelihood": "High", "impact": "Medium", "signal": "Milestone slip", "mitigation": "Gated roadmap"}
  ],
  "vv_plan": {
    "reference_devices": ["PF-400", "PF-1000"],
    "diagnostics": ["I(t)", "dI/dt", "interferometry", "XRD", "neutron_TOF", "activation", "B-dot"],
    "synthetic_diags": ["interferometry", "XRD", "neutron_TOF", "B-dot"]
  },
  "compute_budget": {
    "single_3d_run_gpu_hours": 200,
    "vv_campaign_gpu_hours": 2000,
    "param_scan_gpu_hours": 20000,
    "memory_gb": 512,
    "storage_tb": 50,
    "assumptions": "10^8 cells, 100 ns, 5 species"
  },
  "backlog": [
    {"item": "Implement Hall-MHD solver with CT", "owner_role": "Computational physicist", "skills": ["MHD", "AMReX", "CUDA"], "estimate_pw": 12, "depends_on": []},
    {"item": "Circuit-plasma coupling with variable inductance", "owner_role": "Electrical engineer", "skills": ["Circuit", "PETSc"], "estimate_pw": 6, "depends_on": ["Implement Hall-MHD solver with CT"]},
    {"item": "Integrate SESAME EOS & Saha ionization", "owner_role": "Plasma physicist", "skills": ["EOS", "CR"], "estimate_pw": 8, "depends_on": ["Implement Hall-MHD solver with CT"]},
    {"item": "Optically thin radiation losses", "owner_role": "Plasma physicist", "skills": ["Radiation"], "estimate_pw": 4, "depends_on": ["Integrate SESAME EOS & Saha ionization"]},
    {"item": "AMR infrastructure and mesh import", "owner_role": "HPC engineer", "skills": ["AMReX", "CAD"], "estimate_pw": 10, "depends_on": ["Implement Hall-MHD solver with CT"]},
    {"item": "Synthetic diagnostics module", "owner_role": "Diagnostics expert", "skills": ["Signal modeling"], "estimate_pw": 6, "depends_on": ["AMR infrastructure and mesh import"]},
    {"item": "Validation dataset ingestion", "owner_role": "Data scientist", "skills": ["Data pipelines"], "estimate_pw": 3, "depends_on": ["Synthetic diagnostics module"]},
    {"item": "Hybrid PIC window", "owner_role": "Kinetic specialist", "skills": ["PIC", "WarpX"], "estimate_pw": 12, "depends_on": ["Implement Hall-MHD solver with CT", "AMR infrastructure and mesh import"]},
    {"item": "UQ and parameter sweep tools", "owner_role": "Applied mathematician", "skills": ["UQ", "workflow"], "estimate_pw": 5, "depends_on": ["Validation dataset ingestion"]}
  ]
}
```
