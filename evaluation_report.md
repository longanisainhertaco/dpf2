# Dense Plasma Focus Simulator Evaluation

## Data Flow Architecture Diagram

See [architecture_diagram.mmd](architecture_diagram.mmd) for the Mermaid.js sequence diagram visualizing the flow:
**User → Service A (FastAPI Backend) → Database (JSON Files)**

For detailed architecture analysis and logic errors, see [ARCHITECTURE_ANALYSIS.md](ARCHITECTURE_ANALYSIS.md).

---

## Section 1: Executive Summary & Overall Grade

The Dense Plasma Focus Simulator currently provides a limited subset of the capabilities required by its intended user base. While it includes a basic snowplow-style model and a modern GUI, the absence of self-consistent kinetic physics, sparse diagnostics, and minimal verification infrastructure severely limit its usefulness for high-confidence prediction or educational clarity. Overall, the product resembles a promising prototype rather than a mature research or design tool.

**Overall Letter Grade: D**

The grade reflects the simulator's nascent state: useful for qualitative demonstrations but deficient in the rigorous physics fidelity, parametric flexibility, and workflow support demanded by professional users.

## Section 2: Persona-Specific Sufficiency Analysis

### National Laboratory Scientist — Grade: F

- **Sufficient:**
  - None identified; current release lacks features required for experimental validation.
- **Insufficient:**
  - No Hall-MHD or kinetic engine; cannot model m=0 instabilities or anomalous resistivity.
  - Lacks dual-channel neutron generation and anisotropy diagnostics.
  - No numerics verification suite or reproducibility manifests.

### Design Engineer — Grade: D+

- **Sufficient:**
  - Basic parameter entry for RLC circuits and standard geometries.
  - Simple plotting of current and voltage traces.
- **Insufficient:**
  - No automated sweeps or optimization tools for yield vs. pressure/voltage.
  - Geometry import and material models are absent; cannot analyze tapered or hollow electrodes.
  - Synthetic diagnostics omit detector response functions, preventing one-to-one overlays.

### Student / Educator — Grade: C-

- **Sufficient:**
  - Intuitive GUI with sliders for bank voltage and fill pressure.
  - Animated sheath position during axial rundown.
- **Insufficient:**
  - No guided tutorials linking visuals to underlying physics.
  - Limited plotting; cannot simultaneously view current, voltage, and sheath evolution.

## Section 3: Detailed Categorical Breakdown

### 0. Security & Architecture (NEW ANALYSIS)

**Critical Security Vulnerabilities Identified:**

The web backend (`web/backend/main.py`) contains multiple critical security flaws:

1. **Authentication Bypass (CRITICAL)**: The OAuth2 implementation returns the username as the access token without any JWT signing, expiration, or cryptographic security. Anyone who knows a username can authenticate as that user.

2. **Hardcoded Credentials (CRITICAL)**: Plain-text passwords ("secret") are hardcoded in the source code for both admin and user accounts.

3. **Data Integrity Issue (HIGH)**: The `/results/{run_id}` endpoint returns configuration data instead of actual simulation results, violating the API contract.

4. **Missing Authentication (MEDIUM)**: Snapshot retrieval endpoint (`GET /snapshot/{snap_id}`) has no authentication, allowing anyone to access any snapshot.

5. **Insecure File Upload (MEDIUM)**: The snapshot upload endpoint lacks authentication, file size limits, and validation, creating DoS vulnerabilities.

6. **Predictable Identifiers (MEDIUM)**: Run IDs and snapshot IDs use predictable timestamp-based values, enabling enumeration attacks.

7. **No Rate Limiting (MEDIUM)**: All endpoints lack rate limiting, allowing brute force and DoS attacks.

**Architectural Issues:**

- **No Database**: Despite architectural documentation mentioning a database, the system uses JSON files on the filesystem for all persistence.
- **Mock Implementation**: The `dispatch_to_hpc()` function is a placeholder that only saves configuration files without dispatching actual jobs.
- **In-Memory State**: User credentials and WebSocket clients stored in-memory, lost on restart.
- **Race Conditions**: WebSocket broadcast functions have potential race conditions when clients connect/disconnect during broadcasts.

**Security Grade: F**

The authentication and authorization mechanisms are fundamentally broken and represent an immediate security risk if deployed. See [ARCHITECTURE_ANALYSIS.md](ARCHITECTURE_ANALYSIS.md) for detailed analysis.

### 1. Physics Engine Fidelity & Scope

The simulator offers only a single fluid snowplow model with fixed resistive parameters. Breakdown, axial rundown, and radial compression are treated in a lumped fashion without spatial resolution, leaving key phenomena—such as sheath curvature, Bennett equilibrium, and m-spectrum growth—unmodeled. There is no Hall or two-fluid capability, and the code does not report dimensionless parameters or regime gates. Consequently, scientists cannot interrogate anomalous resistivity mechanisms or validate against published MJOLNIR or PF-1000 datasets. The absence of a numerics verification panel (e.g., Brio–Wu, Orszag–Tang, MMS) further undermines confidence in its solutions.

### 2. Parametric Control & Geometry Definition

Circuit definition is limited to single-stage RLC inputs, with no provision for transmission lines, crowbars, or saturable inductors. Electrode geometry is restricted to idealized Mather and Filippov presets; importing CAD or defining tapered, hollow, or re-entrant shapes is impossible. Gas handling is static and uniform, lacking puff timing or neutral-plasma coupling. Breakdown modeling is simplified to an instantaneous switch closure, neglecting surface flashover statistics and triple-junction field enhancements.

### 3. Diagnostics, Data Output, & Visualization

Data output consists primarily of time-series for current and voltage. There are no synthetic neutron, X-ray, or interferometry diagnostics, and no energy partition tracking. Visualization is limited to a 2D sheath position plot without vector overlays or density maps. Export options are restricted to CSV, and there is no instrument response modeling. Users cannot examine dimensionless regime dashboards or compute wall-plug efficiency and yield/hour metrics.

### 4. User Experience (UX), Documentation, & Workflow

The GUI is adequate for simple parametric adjustments but lacks project management, comparison tools, or parametric sweep orchestration. There is no CLI or API for batch execution, precluding integration with HPC workflows. Documentation is sparse, offering only a quick-start guide with minimal explanation of physics models or numerical schemes. Uncertainty quantification, Bayesian calibration, surrogate modeling, and multi-objective optimization are absent, leaving engineers and scientists without critical design or inference capabilities.

## Section 4: Actionable Recommendations & Development Roadmap

### Security & Infrastructure (IMMEDIATE - Before Any Deployment)

1. **[Priority: CRITICAL]** Replace the authentication system with proper JWT tokens using a signing secret and expiration times.
2. **[Priority: CRITICAL]** Remove hardcoded credentials; use environment variables and hash passwords with bcrypt or argon2.
3. **[Priority: CRITICAL]** Add authentication to all endpoints that handle user data (snapshot retrieval, file uploads).
4. **[Priority: CRITICAL]** Implement proper error handling for all file operations to prevent information leakage.
5. **[Priority: HIGH]** Add rate limiting middleware to prevent brute force and DoS attacks.
6. **[Priority: HIGH]** Use cryptographically secure random identifiers (UUID4) instead of predictable timestamp-based IDs.
7. **[Priority: HIGH]** Implement a proper database (PostgreSQL/SQLite) for user management, sessions, and results storage.
8. **[Priority: HIGH]** Separate configuration storage from results storage; create a proper `/results` vs `/config` endpoint distinction.
9. **[Priority: HIGH]** Add file validation, size limits, and content-type checking to upload endpoints.
10. **[Priority: MEDIUM]** Implement proper session management with secure cookies and CSRF protection.
11. **[Priority: MEDIUM]** Add comprehensive logging of security events and failed authentication attempts.
12. **[Priority: MEDIUM]** Implement proper CORS policies and security headers.

### Physics & Simulation Features

1. **[Priority: Critical]** Implement a Hall-MHD/Two-Fluid engine with Braginskii transport and regime gating (ω_ce τ_e, d_i/L).
2. **[Priority: Critical]** Add a Numerics Verification panel with Brio–Wu, Orszag–Tang, and MMS tests reporting observed order of accuracy.
3. **[Priority: Critical]** Introduce mechanism-resolved anomalous resistivity during the pinch, driven by lower-hybrid drift waves, and expose effective impedance relative to Spitzer values.
4. **[Priority: Critical]** Provide dual-channel neutron yield modeling (thermonuclear vs beam-target) with angular spectra and time-of-flight phasing linked to I–V features.
5. **[Priority: High]** Incorporate vacuum surface flashover modeling with stochastic delay and conditioning curves to capture breakdown statistics.
6. **[Priority: High]** Maintain versioned atomic and collision data (ADAS/CHIANTI/LXCat) with DOIs recorded in run manifests.
7. **[Priority: High]** Develop a neutral gas module (DSMC or hybrid fluid) with validation against LXCat swarm parameters.
8. **[Priority: High]** Implement plasma–material interaction models for sputtering and impurity evolution, affecting Z_eff and radiation output.
9. **[Priority: High]** Enable azimuthal mode decomposition and growth-rate tracking for m=0 and m=1 instabilities.
10. **[Priority: High]** Add a PSATD Maxwell solver with charge-conserving current deposition and divergence monitoring for PIC runs.
11. **[Priority: High]** Support adaptive mesh refinement triggered by λ_D, d_i, |∇p|, and |J| thresholds.
12. **[Priority: High]** Distribute containerized HPC images with documented strong/weak scaling and Roofline analyses.
13. **[Priority: High]** Embed detector response functions and cable dispersion in all synthetic diagnostics.
14. **[Priority: Medium]** Display a live regime dashboard of key dimensionless parameters and flag model-validity violations.
15. **[Priority: Medium]** Introduce OOD detectors and conformal error bands for ML surrogates to guard against extrapolation.
16. **[Priority: Medium]** Compute throughput metrics, including yield/shot, yield/hour, electrode lifetime, and wall-plug efficiency.
17. **[Priority: Low]** Create a “Lab-mode” UI that simulates shot-to-shot jitter and records batch run manifests for reproducibility.

## Appendix A — DPF Physics Acceptance Tests

| Test | Pass/Fail | Notes |
| --- | --- | --- |
| Flashover realism | Fail | Switch closes instantaneously; no surface flashover model. |
| Triple-point fields | Fail | No geometry-dependent field enhancement. |
| Neutral–plasma handoff | Fail | Initial sheath imposed rather than formed self-consistently. |
| Vacuum Surface Flashover stats | Fail | No stochastic delay or conditioning curves. |
| Photoionization option | Fail | No photoionization sources available. |
| GV trajectory check | Fail | No axial position output for comparison. |
| Speed parameter S band | Fail | Simulator does not compute S. |
| Sheath curvature diagnostic | Fail | Curvature not resolved. |
| Bennett consistency gate | Fail | No reporting of Bennett mismatch. |
| Dynamic inductance L_p(t) overlay | Fail | Inductance not reconstructed. |
| Energy partition | Fail | No channel tracking. |
| Hall activation gate | Fail | Hall terms absent. |
| m=0 growth & beam formation | Fail | No instability modeling. |
| Anomalous resistivity mechanism | Fail | Spike absent; fixed resistivity used. |
| m-spectrum growth | Fail | No azimuthal decomposition. |
| Dual-channel neutron yield | Fail | Only total yield reported. |
| Angular distribution | Fail | No angular diagnostics. |
| Time-of-flight phasing | Fail | No ToF outputs. |
| No-gas negative control | Fail | Simulation crashes without gas. |
| No-Hall negative control | Fail | Hall physics unavailable; cannot test. |

## Appendix B — Benchmark Suite Results

No benchmark projects are supplied; consequently, no overlays or tolerance-band comparisons can be performed.

## Appendix C — Numerics Verification & Stability

No Brio–Wu, Orszag–Tang, or MMS tests are available. Divergence control, energy drift analysis, and PSATD vs FDTD dispersion comparisons are absent.

## Appendix D — Performance & Reproducibility

The simulator is distributed only as a Windows executable; container hashes, compiler flags, MPI/HDF5 settings, and scaling plots are not provided. There is no evidence of parallel I/O benchmarks or Roofline analyses.

