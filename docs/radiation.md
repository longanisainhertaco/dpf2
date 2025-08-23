# Radiation Loss Models

The Dense Plasma Focus (DPF) pinch loses energy through several radiation
mechanisms.  The simple models used in `dpf2` estimate the power density
:math:`P` (in W/m^3) for each mechanism as a function of electron temperature
``T_e`` [K] and densities ``n_e`` and ``n_i`` [m^-3]:

## Bremsstrahlung

Free–free emission from electron–ion collisions is approximated by

\[
P_{\text{brem}} \approx 1.69\times10^{-32} Z_{\text{eff}} n_e n_i \sqrt{T_e}.
\]

## Line Radiation

Excited ions radiatively decay, modeled here with a crude exponential drop
with temperature using the hydrogenic excitation energy of 13.6 eV:

\[
P_{\text{line}} \approx 10^{-31} n_e n_i \exp(-13.6 / T_{e,\text{eV}}).
\]

## Radiative Recombination

When an electron recombines with an ion it emits a photon.  The associated
power density is estimated by

\[
P_{\text{recomb}} \approx 1.7\times10^{-32} Z_{\text{eff}} n_e n_i / \sqrt{T_e}.
\]

These expressions are adapted from standard plasma-physics references and are
valid only for rough order-of-magnitude estimates.

## References

- J. D. Huba, *NRL Plasma Formulary*, Naval Research Laboratory, 2016.
- P. M. Bellan, *Fundamentals of Plasma Physics*, Cambridge University Press, 2008.
