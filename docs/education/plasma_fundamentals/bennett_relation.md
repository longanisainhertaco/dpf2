# Bennett Equilibrium and Pinch Dynamics

This module derives the Bennett relation and explores its applications to Dense Plasma Focus (DPF) physics.

## Learning Objectives

After completing this module, you will be able to:

1. Derive the Bennett relation from first principles
2. Apply the Bennett relation to calculate equilibrium parameters
3. Understand the limitations and extensions of the Bennett model
4. Analyze DPF performance using Bennett scaling

---

## 1. Introduction to Bennett Equilibrium

### 1.1 Historical Context

The Bennett relation, derived by Willard Bennett in 1934, describes the equilibrium of a current-carrying plasma column. It remains one of the most important results in pinch physics.

### 1.2 The Physical Picture

In a z-pinch:
- Current flows axially
- This creates an azimuthal magnetic field
- The $\vec{J} \times \vec{B}$ force compresses the plasma inward
- At equilibrium, magnetic pressure balances thermal pressure

---

## 2. Derivation of the Bennett Relation

### 2.1 Assumptions

We assume:
- Cylindrical symmetry
- Steady-state equilibrium
- Uniform temperature across the column
- Quasi-neutrality ($n_e \approx Z n_i$)
- Sharp boundary (pressure goes to zero at $r = a$)

### 2.2 Starting Equations

**MHD equilibrium equation** (radial component):

$$\frac{dP}{dr} = J_z B_\theta$$

**Ampère's law**:

$$\frac{1}{r}\frac{d(r B_\theta)}{dr} = \mu_0 J_z$$

**Equation of state** (ideal gas):

$$P = n k_B T (1 + Z)$$

where $Z$ is the average ionization state and the factor $(1+Z)$ accounts for both ions and electrons.

### 2.3 Step-by-Step Derivation

**Step 1**: Express the current density

From Ampère's law:
$$J_z = \frac{1}{\mu_0 r}\frac{d(r B_\theta)}{dr}$$

**Step 2**: Substitute into the equilibrium equation

$$\frac{dP}{dr} = \frac{B_\theta}{\mu_0 r}\frac{d(r B_\theta)}{dr}$$

**Step 3**: Rearrange and integrate

$$\frac{dP}{dr} = \frac{1}{\mu_0}\left(\frac{B_\theta}{r}\frac{d(r B_\theta)}{dr}\right) = \frac{1}{\mu_0}\left(B_\theta \frac{dB_\theta}{dr} + \frac{B_\theta^2}{r}\right)$$

This can be written as:

$$\frac{dP}{dr} = \frac{d}{dr}\left(\frac{B_\theta^2}{2\mu_0}\right) + \frac{B_\theta^2}{\mu_0 r}$$

**Step 4**: Multiply by $2\pi r$ and integrate from 0 to $a$

$$\int_0^a 2\pi r \frac{dP}{dr} dr = \int_0^a 2\pi r \left[\frac{d}{dr}\left(\frac{B_\theta^2}{2\mu_0}\right) + \frac{B_\theta^2}{\mu_0 r}\right] dr$$

**Step 5**: Evaluate the left side using integration by parts

$$\int_0^a 2\pi r \frac{dP}{dr} dr = [2\pi r P]_0^a - \int_0^a 2\pi P \, dr = -\int_0^a 2\pi P \, dr$$

(since $P(a) = 0$ and $P(0)$ is finite)

The integral $\int_0^a 2\pi r P \, dr$ relates to the line density:

$$N_l = \int_0^a 2\pi r n \, dr \quad \text{(particles per unit length)}$$

For uniform temperature:
$$\int_0^a 2\pi r P \, dr = k_B T (1+Z) N_l$$

**Step 6**: Evaluate the right side

The first term:
$$\int_0^a 2\pi r \frac{d}{dr}\left(\frac{B_\theta^2}{2\mu_0}\right) dr = \left[\pi r^2 \frac{B_\theta^2}{2\mu_0}\right]_0^a - \int_0^a \frac{\pi B_\theta^2}{\mu_0} dr$$

At $r = a$: $B_\theta(a) = \mu_0 I/(2\pi a)$

So: $\left[\pi r^2 \frac{B_\theta^2}{2\mu_0}\right]_a = \frac{\mu_0 I^2}{8\pi}$

The second term:
$$\int_0^a \frac{2\pi B_\theta^2}{\mu_0} dr$$

**Step 7**: Combine and simplify

After careful evaluation of all integrals (the magnetic field integrals cancel for uniform current density), we obtain:

$$k_B T (1+Z) N_l = \frac{\mu_0 I^2}{8\pi}$$

### 2.4 The Bennett Relation

Rearranging:

$$\boxed{I^2 = \frac{8\pi k_B T}{\mu_0} N_l (1+Z)}$$

Or equivalently:

$$\boxed{I^2 = \frac{8\pi}{\mu_0} N_l (k_B T_e + Z k_B T_i)}$$

where we allow for different electron and ion temperatures.

---

## 3. Physical Interpretation

### 3.1 Meaning of the Bennett Relation

The Bennett relation states that for a pinch in equilibrium:

$$\text{(Current)}^2 \propto \text{(Temperature)} \times \text{(Line Density)}$$

This means:
- Higher current → higher temperature OR higher density
- For fixed current, hotter plasmas are less dense
- For fixed current, denser plasmas are cooler

### 3.2 The Bennett Temperature

For a given current and line density, the **Bennett temperature** is:

$$T_B = \frac{\mu_0 I^2}{8\pi k_B N_l (1+Z)}$$

This represents the equilibrium temperature if all magnetic energy were thermalized.

### 3.3 Pressure-Current Relation

At the axis of a Bennett pinch:

$$P_0 = \frac{\mu_0 I^2}{4\pi^2 a^2}$$

This shows the extreme pressures achievable in pinches.

---

## 4. Bennett Relation in DPF

### 4.1 Application to DPF Focus

In a DPF, the Bennett relation helps predict:

1. **Final pinch radius**: Given current and desired temperature
2. **Plasma density**: For fusion conditions
3. **Confinement time**: Related to instability growth

### 4.2 Typical DPF Parameters

| Parameter | Small DPF | Large DPF |
|-----------|-----------|-----------|
| Current | 100 kA | 2 MA |
| Line density | $10^{18}$ m⁻¹ | $10^{19}$ m⁻¹ |
| Bennett temperature | 500 eV | 2 keV |
| Pinch radius | 0.5 mm | 2 mm |

### 4.3 Bennett Condition Check

For a DPF with $I = 500$ kA, $N_l = 5 \times 10^{18}$ m⁻¹, $Z = 1$:

$$T_B = \frac{4\pi \times 10^{-7} \times (5 \times 10^5)^2}{8\pi \times 1.38 \times 10^{-23} \times 5 \times 10^{18} \times 2}$$

$$T_B = \frac{10^{-7} \times 2.5 \times 10^{11}}{2 \times 1.38 \times 10^{-4}} \approx 9 \times 10^7 \text{ K} \approx 8 \text{ keV}$$

---

## 5. Extensions and Limitations

### 5.1 Non-Uniform Temperature

For radially varying temperature $T(r)$:

$$I^2 = \frac{8\pi}{\mu_0} \int_0^a 2\pi r \cdot n(r) k_B T(r) (1+Z) \, dr$$

### 5.2 Including Axial Field

With an axial magnetic field $B_z$:

$$P + \frac{B_z^2}{2\mu_0} = \frac{B_\theta^2}{2\mu_0} + \text{(magnetic tension terms)}$$

This modifies the Bennett relation and can provide stabilization.

### 5.3 Dynamic Effects

The Bennett relation describes **equilibrium**. Dynamic effects include:

1. **Snowplow dynamics**: Current sheath sweeps up mass
2. **Radiation losses**: Reduce achievable temperature
3. **Instabilities**: Prevent equilibrium from being reached

### 5.4 Kinetic Effects

At high temperatures, kinetic effects become important:

- Non-Maxwellian distributions
- Ion beams
- Runaway electrons

These are not captured by the fluid Bennett model.

---

## 6. Bennett Scaling Laws

### 6.1 Neutron Yield Scaling

Combining Bennett relation with fusion cross-section scaling:

$$Y_n \propto I^4$$

This is the famous "fourth power" scaling law for DPF neutron yield.

### 6.2 Derivation of I⁴ Scaling

1. From Bennett: $n T \propto I^2/a^2$
2. Fusion rate: $R \propto n^2 \langle\sigma v\rangle$
3. For D-D fusion below 10 keV: $\langle\sigma v\rangle \propto T^2$
4. Therefore: $R \propto n^2 T^2 \propto I^4/a^4$
5. Volume: $V \propto a^2$
6. Total yield: $Y \propto R \cdot V \propto I^4/a^2$

With pinch radius scaling weakly with current, $Y \propto I^4$ emerges.

### 6.3 Practical Limits

The $I^4$ scaling breaks down due to:

- Radiation losses ($\propto n^2 T^{1/2}$)
- Instabilities at high current
- Circuit limitations

---

## 7. Computational Examples

### 7.1 Python Implementation

```python
import numpy as np

def bennett_temperature(I, N_l, Z=1):
    """
    Calculate Bennett equilibrium temperature.
    
    Parameters
    ----------
    I : float
        Current in Amperes
    N_l : float
        Line density in particles/meter
    Z : float
        Average ionization state
    
    Returns
    -------
    T : float
        Temperature in Kelvin
    T_eV : float
        Temperature in electronvolts
    """
    mu_0 = 4 * np.pi * 1e-7  # H/m
    k_B = 1.38e-23  # J/K
    eV_to_K = 11605  # K/eV
    
    T = (mu_0 * I**2) / (8 * np.pi * k_B * N_l * (1 + Z))
    T_eV = T / eV_to_K
    
    return T, T_eV

def bennett_current(T_eV, N_l, Z=1):
    """
    Calculate current required for Bennett equilibrium.
    
    Parameters
    ----------
    T_eV : float
        Temperature in electronvolts
    N_l : float
        Line density in particles/meter
    Z : float
        Average ionization state
    
    Returns
    -------
    I : float
        Current in Amperes
    """
    mu_0 = 4 * np.pi * 1e-7
    k_B = 1.38e-23
    eV_to_K = 11605
    
    T = T_eV * eV_to_K
    I = np.sqrt(8 * np.pi * k_B * T * N_l * (1 + Z) / mu_0)
    
    return I

# Example: DPF conditions
I = 500e3  # 500 kA
N_l = 5e18  # 5×10^18 m^-1
T, T_eV = bennett_temperature(I, N_l, Z=1)
print(f"Bennett temperature: {T_eV:.1f} eV ({T/1e6:.1f} MK)")
```

### 7.2 Using dpf2 for Bennett Analysis

```python
from dpf2.pinch_models import BennettPinch
from dpf2.physics import compute_bennett_temperature

# Create a Bennett pinch model
pinch = BennettPinch(
    current=500e3,  # 500 kA
    line_density=5e18,  # m^-1
    ionization_state=1
)

# Calculate equilibrium properties
T_eq = pinch.equilibrium_temperature()
a_eq = pinch.equilibrium_radius()
P_eq = pinch.axis_pressure()

print(f"Equilibrium temperature: {T_eq:.1f} eV")
print(f"Equilibrium radius: {a_eq*1e3:.2f} mm")
print(f"Axis pressure: {P_eq/1e9:.1f} GPa")
```

---

## 8. Summary

Key points to remember:

1. The **Bennett relation** $I^2 = (8\pi k_B T/\mu_0) N_l$ describes z-pinch equilibrium
2. It connects current, temperature, and density in a fundamental way
3. The **Bennett temperature** is the equilibrium temperature for given current and line density
4. The **I⁴ scaling** of neutron yield derives from the Bennett relation
5. Real DPF plasmas are dynamic and may not reach true Bennett equilibrium

---

## 9. Derivation Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                     BENNETT RELATION DERIVATION                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Start: MHD Equilibrium         ∇P = J × B                      │
│                                                                  │
│  Cylindrical form:              dP/dr = J_z B_θ                 │
│                                                                  │
│  Ampère's law:                  J_z = (1/μ₀r) d(rB_θ)/dr       │
│                                                                  │
│  Combine and integrate:         ∫ P dA = (μ₀I²)/(8π)           │
│                                                                  │
│  Equation of state:             P = nk_BT(1+Z)                  │
│                                                                  │
│  Result:                        I² = (8πk_BT/μ₀) N_l (1+Z)      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Further Reading

1. Bennett, W.H. "Magnetically Self-Focussing Streams" (1934) - The original paper
2. Ryutov, D.D. et al. "The physics of fast Z pinches" (2000)
3. Haines, M.G. "A review of the dense Z-pinch" (2011)

---

## Exercises

1. Derive the Bennett relation for a parabolic density profile $n(r) = n_0(1 - r^2/a^2)$.
2. Calculate the Bennett temperature for the PF-1000 device (I = 2 MA, $N_l = 10^{19}$ m⁻¹).
3. Show that including an axial field $B_z$ modifies the Bennett relation.
4. Using dpf2, compare simulated pinch parameters to Bennett equilibrium predictions.

---

*Previous: [Magnetic Pressure](magnetic_pressure.md)*
