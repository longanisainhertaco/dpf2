# Tasks for Full-3D High-Fidelity DPF Simulator Gap Review

The following task list follows the instructions in `DPF_FULL3D_GAP_REVIEW_PROMPT.md` and outlines steps needed to produce a comprehensive gap analysis and roadmap for a Dense Plasma Focus simulation prototype.

1. **Collect Required Inputs**
   - Gather prototype overview: architecture, language, dependencies, solvers, licensing.
   - Document current physics scope: circuit coupling, MHD level, radiation, EOS, transport, neutrals, sheath, kinetic modules, nuclear reactions.
   - Record numerical methods: discretization, temporal schemes, Riemann solvers, divergence control, stiffness handling, AMR strategy, mesh type.
   - Capture geometry and boundary conditions information: CAD formats, feed/return paths, vessel details, gas fill, vacuum treatment, material properties.
   - List couplings: external circuit models, vacuum EM region approach, PIC or hybrid models, MCC, radiation transport type.
   - Assemble data available for validation: current traces, interferometry, imaging, X-ray diodes, neutron diagnostics, B-dot probes, facility references.
   - Assess performance and scaling metrics: threading/GPU usage, MPI strategy, scaling plots, memory footprint, wall-clock times.
   - Review software process: tests, CI, documentation, reproducibility, I/O formats, in-situ visualization, restart capability, provenance, UQ hooks.
   - Clarify target use-cases: parameter scans, design optimization, synthetic diagnostics, shot-to-shot prediction.

2. **Conduct Rapid Inventory**
   - Map prototype capabilities against the target checklist, marking each as Present, Partial, Missing, or Unknown.

3. **Perform Gap Analysis by Domain**
   - For each domain listed in the prompt (circuit-plasma, MHD, radiation, etc.), enumerate missing features, technical blockers, and feasible solution paths.
   - Prioritize gaps by impact on predictive fidelity and implementation complexity or risk.

4. **Compile Risk Register**
   - Identify scientific, numerical, and engineering risks with likelihood, impact, early warning signals, and mitigation strategies.

5. **Develop Roadmap to High Fidelity**
   - Propose a three-phase plan: Foundations, Physics Completion, Predictive Production.
   - Define milestones and exit criteria, including demonstration problems such as stable Hall-MHD pinch, non-LTE radiation, hybrid PIC window, neutron yield validation, and full experimental shot replay.

6. **Estimate Compute & Data Budget**
   - Provide node/GPU-hour estimates, memory, and storage for a design-sized run, V&V campaign, and 100-shot parameter scan, noting assumptions.

7. **Define Acceptance Tests & Metrics**
   - Specify quantitative targets for current waveforms, pinch characteristics, neutron yield, scaling laws, instability onset, and scaling efficiency.

8. **Build Actionable Backlog**
   - Convert identified gaps into an ordered backlog with owner roles, required skills, person-week estimates, and dependencies.

9. **Prepare Deliverables**
   - Compile human-readable outputs: Executive Summary, Gap Matrix, Capability Radar, Roadmap, Risk Register, V&V Plan, Compute & Data Budget, and Backlog.
   - Generate machine-readable JSON summary following the schema in the prompt.

These tasks collectively enable a systematic evaluation and planning effort to progress an existing DPF simulation prototype toward a true full-3D, high-fidelity production code.
