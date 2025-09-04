# Particle-in-Cell Method

The Particle-in-Cell (PIC) approach resolves kinetic effects by tracking macro-particles on a mesh.

## Workflow

1. Deposit particle charge and current onto the grid.
2. Solve field equations such as Maxwell's equations.
3. Interpolate fields back to particle positions.
4. Push particles via the Lorentz force.

PIC captures beam dynamics and non-equilibrium phenomena beyond fluid theory.
