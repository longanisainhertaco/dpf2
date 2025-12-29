# Method of Manufactured Solutions (MMS) Tests

Use this directory to house MMS-based verification problems for field solvers, particle pushers, and coupled hybrid-PIC operators. Each test should define:

- The manufactured analytic solution and source terms.
- Discrete error norms and convergence-rate checks.
- Automation hooks (e.g., pytest) for regression testing.
