# Dense Plasma Focus PIC Tool Blueprint

0) Scope
Goal: A 3-D, fully electromagnetic, relativistic, multi-species PIC code that reproduces DPF’s four phases and, critically, the instability/disruption physics that generate beams and neutrons. It must couple to a pulsed-power circuit, include collisions/chemistry/radiation, and ship with synthetic diagnostics and benchmarks that match experimental DPF signatures (current dip, voltage spike, anisotropic neutron yield).

1) Physics coverage (first-principles)
1.1 Maxwell–Lorentz core (must)
Equations:
∂tE=c^2∇×B−J/ϵ0,
∂tB=−∇×E,
d/dt(γmsvs)=qs(E+vs×B)
with relativistic momentum γmsv.
Why: Only full EM PIC can form the huge axial Ez during disruption and self-consistently accelerate beams.
Implement: PSATD (pseudo-spectral) or high-order FDTD with guard-cell exchange; charge-conserving current deposition (Esirkepov/Villasenor–Buneman).
Verify: EM wave dispersion & energy conservation; discrete continuity equation holds: Δtρ+∇⋅J=0 to round-off.

1.2 Multi-species kinetics (must)
Species: e−, D+, D2+, neutrals D2; optionally He/Ne/Ar for SXR studies; optional trace impurities (Cu/W/C) injected from walls.
Why: Beam-target neutrons depend on non-Maxwellian ion tails and impurity-driven radiation losses (Zeff) that shift pinch conditions.
Implement: Per-species particle containers; variable weights; SoA layout for GPU.

1.3 Collisions & chemistry (must)
Electron/ion Coulomb collisions: Takizuka–Abe / Nanbu binary MCC.
e–neutral / i–neutral: elastic, excitation, ionization (state-resolved cross sections) with probability P=1−exp(−νΔt).
Charge-exchange for ions in D2 / noble gases.
Recombination: radiative and 3-body (pinch edge).
Why: Breakdown → sheath → pinch all depend on collisionality; beam-plasma scattering shapes EDFs and neutron yield partition.
Verify: Recover Spitzer trends in collisional limit; energy/momentum conservation per collision operator (ensemble).

1.4 Surface physics & emission (should→must for breakdown realism)
SEE δ(E), Fowler–Nordheim field emission JFE = A(βE)^2/ϕ exp(−Bϕ^{3/2}/(βE)).
Particle–wall interactions: absorption, reflection, sputter source to plasma.
Why: Triple-point priming and early sheath quality govern later symmetry and pinch timing.
Verify: Field emission onset vs. rise-time; increasing SEE reduces breakdown field in insulator tests.

1.5 Radiation & fusion sources (must)
Bremsstrahlung + line radiation: tabulated/CR-lite model for power sink and synthetic SXR signals.
Neutrons:
Thermonuclear: R_th = n_D^2 ⟨σv⟩(T_i)
Beam–target: R_bt = ∫ n_b(E) n_t σ_DD(E) v(E) dE with EDF from PIC.
Why: Framework requires partition of neutron channels and anisotropy prediction.
Verify: Forward/radial count ratio ≫ 1 for beam-dominated cases; ToF timing vs. I/V spike (lag window).

2) Numerical methods (high-fidelity stack)
2.1 Field solver (must)
Option A: PSATD (Galilean or relativistic) to suppress numerical Cherenkov for fast beams.
Option B: High-order FDTD with binomial filters, CT/GLM cleaning for ∇⋅B=0.
PML absorbing layers; perfect-conductor electrodes with imposed potentials.

2.2 Particle pusher (must)
Relativistic Boris or Vay/Higuera–Cary (reduced dispersion for ultra-relativistic beams).
Shape functions: CIC/TSC (≥ 2nd order) for low noise.
Sub-cycling electrons (optional) if ions set global Δt.

2.3 Geometry (must)
Embedded-boundary (EB-PIC) for coaxial electrodes, insulator, ports; conformal fields at metal boundaries.
Mesh: Cartesian or cylindrical with block-structured AMR tracking sheath/pinch; strict charge conservation across refinement boundaries.

2.4 Time-step and resolution gates (must)
Recommended gates (tool should enforce/warn):
Δt ω_pe ≲ 0.2,
Δt Ω_ce ≲ 0.2,
Δx ≲ min{ α c/ω_pe, β ρ_e }
with α,β ~ 0.5 for instability resolution; report violations near pinch.
Note: Full λ_D resolution at stagnation may be infeasible; tool MUST support PIC-of-the-pinch (local refinement window) and/or implicit EM to relax c-CFL while preserving kinetics.

2.5 Charge & energy conservation (must)
Current deposition: strictly charge-conserving;
Energy drift: <1% over rundown in collisionless benchmarks.

3) Circuit–plasma co-simulation (must)
External driver: multi-section RLC or transmission-line/Blumlein with switch model and jitter.
Two-way coupling: impose electrode potentials from circuit; measure terminal current from PIC (I=∫J·dA) and return V_pl(t) to circuit.
Dynamic inductance diagnostic:
W_m(t)=∫ B^2/(2μ0) dV,
L_p(t)=2 W_m(t) / I^2(t),
and independently via L_rec(t) = (V − I R_ext)/(dI/dt) − L_0.
Why: Reproduces current dip & voltage spike timing against experiment—a top-level acceptance test in your framework.
Verify: Agreement of L_p(t) and L_rec(t) through pinch window.

4) Materials & surfaces (tight coupling) (should→must)
SEE & FE boundary models; sputter/evaporation mass source feeding plasma; optional temperature-dependent conductivity for electrodes (skin effect).
Evolving state: roughness, deposition films on insulator (flashover risk), coating thickness.
Why: Materials alter breakdown, sheath symmetry, Zeff (radiation), and multi-shot stability—explicitly highlighted in your materials roadmap.
Verify: Changing anode tip W→Cu shifts impurity injection and modifies spike amplitude statistically over repeats.

5) Diagnostics & synthetic instruments (must)
Waveforms: I(t), V(t), Ez (axis), L_p(t); spectra of E, B fluctuations.
Particles: EEDF/IEDF (angle-resolved), pitch-angle distributions, phase-space movies.
Neutrons: Y_th, Y_bt, dN/dΩ, ToF; forward/radial/backward counters.
X-ray/SXR: line-of-sight photodiodes, pinhole camera synthesis.
Why: Your rubric requires synthetic diagnostics for apples-to-apples experiment comparisons.
Verify: Cross-correlate neutron peak with voltage spike lag; report anisotropy factor A = N_f / N_r.

6) Multi-scale strategy (must)
PIC-of-the-pinch mode: Pre-pinch macroscopic state imported from MHD/hybrid (density/current profiles, flows); PIC domain spans anode-tip region, evolving through compression → disruption → beam emission.
Optional global PIC: coarse outer domain + AMR focus region.
Why: Enables feasible runs while preserving the kinetic heart of DPF physics.
Verify: Matching rundown timing vs. MHD; identical disruption timing windows in both.

7) HPC & software engineering (must)
Parallelism: MPI + GPU offload (CUDA/HIP/SYCL); domain decomposition with dynamic load balancing (pinch causes extreme clustering).
I/O: HDF5 with in-situ reduction; checkpoint/restart; reproducible seeds.
Units: SI enforced end-to-end; manifest records code hash, deck, cross-section tables.
Profiling: Roofline reports; kernel timings; scalability plots to 10^4+ GPUs/cores.
Why: Pinch-phase PIC is compute-intense (your framework warns of cost); engineering must sustain it.

8) Verification & validation suite (must)
8.1 PIC code-verification
Cold plasma EM wave, Langmuir/ion-acoustic dispersion; two-stream, Buneman, Weibel, lower-hybrid drift (LHDI) growth rates vs. theory.
Sheath formation at a conducting wall; diode with FE/SEE.
Pass: growth rates and frequencies within 5–10%.

8.2 DPF-specific validation
Rundown test: snowplow/GV-derived mass-loading reproduced (position vs. time overlay).
Pinch timing: I(t) dip and V(t) spike within a tight window; Ez burst coincident with spike.
Beam physics: IEDF tail >100 keV–MeV; anisotropic neutron yield A≫1 for beam-dominated shots.
Dynamic inductance: L_p(t) matches reconstruction through spike.
Why: These are acceptance tests from your rubric.

9) Input-deck structure (YAML sketch)
```
geometry:
  type: coaxial
  anode_radius_m: 0.01
  cathode_radius_m: 0.05
  insulator_length_m: 0.03
  eb_surfaces: [metal_anode, metal_cathode, alumina_insulator]
driver:
  type: transmission_line
  sections: [...]
  switch: { model: arc, jitter_ps: 50 }
species:
  - { name: electron, charge_C: -1.602e-19, mass_kg: 9.11e-31, macro_per_cell: 200 }
  - { name: deuteron, charge_C: 1.602e-19, mass_kg: 3.344e-27, macro_per_cell: 100 }
  - { name: D2_neutral, model: DSMC }
collisions:
  coulomb: { model: nanbu, ee: true, ei: true, ii: true }
  mcc: { en: ["elastic","excitation","ionization"], in: ["cx","ionization"] }
boundaries:
  walls: { SEE: true, FE: true, sputter: true }
solver:
  fields: { scheme: PSATD, dt_s: 2.0e-14 }
  pusher: { scheme: Vay }
amr:
  criteria: [gradB, ne]
diagnostics:
  waveforms: [I, V, Ez_axis, Lp]
  particles: [EEDF, IEDF_angle]
  neutrons: { modes: ["thermonuclear","beam_target"], tof_detectors: [...] }
```

10) “Definition of Done” — acceptance gates
Physics closure: All modules in §1–§5 active; multi-species, collisions, fusion partition, synthetic diagnostics.
Disruption signatures: Concurrent I dip, V spike, Ez surge reproduced; effective η_eff = (E·J)/J^2 shows pinch-phase rise.
Anisotropy: Forward/radial neutron counts A = N_f / N_r ≫ 1 in beam-dominated decks.
Inductance: L_p(t) (field-derived) ≈ L_rec(t) (circuit-derived) over the spike.
Scaling: Y_n increases with peak current near optimal pressure; speed parameter stability visible in pre-pinch setup (when using hybrid hand-off).
Numerics: ∇·B within tolerance; energy drift <1% pre-pinch; LHDI/two-stream growth within theory bands.
HPC: Strong/weak scaling achieved to target nodes; restart works across pinch.

11) Feature backlog (priority)
[Critical] PSATD/Galilean field solver + charge-conserving deposition (relativistic pusher).
[Critical] MCC chemistry (ionization, CX), Coulomb collisions, materials SEE/FE, sputter source.
[Critical] Two-way circuit coupling with dynamic L_p(t) diagnostic and reconstruction.
[High] PIC-of-the-pinch deck builder + MHD/hybrid hand-off tooling.
[High] Full neutron pipeline (EDF→Y_bt, anisotropy, ToF) and SXR radiation.
[High] AMR with charge conservation across levels; dynamic load balancing.
[Medium] EB-PIC geometry for complex anode tips, hollow/tapered options.
[Medium] In-situ analysis (phase-space, spectra) to cut I/O.
[Low] Optional implicit EM (Darwin/IMEX) for oversized domains.
