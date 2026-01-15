# Magnetic Pressure and Pinch Physics

This module covers the fundamental concepts of magnetic pressure and how it drives plasma compression in Dense Plasma Focus (DPF) devices.

## Learning Objectives

After completing this module, you will be able to:

1. Calculate magnetic pressure from field strength
2. Explain the pinch effect in z-pinch geometries
3. Apply equilibrium conditions to plasma columns
4. Understand the role of magnetic pressure in DPF operation

---

## 1. Magnetic Pressure

### 1.1 Definition

The **magnetic pressure** is the pressure exerted by a magnetic field on a plasma or conductor. It represents the energy density of the magnetic field:

$$\boxed{P_B = \frac{B^2}{2\mu_0}}$$

where:
- $P_B$ = magnetic pressure (Pa)
- $B$ = magnetic field strength (T)
- $\mu_0 = 4\pi \times 10^{-7}$ H/m (permeability of free space)

### 1.2 Physical Interpretation

The magnetic pressure arises from the Lorentz force on current-carrying plasma:

$$\vec{F} = \vec{J} \times \vec{B}$$

For a current flowing parallel to itself (as in a plasma column), this force is always inward, creating compression.

### 1.3 Numerical Examples

| Current | Radius | B-field | Magnetic Pressure |
|---------|--------|---------|-------------------|
| 100 kA | 1 cm | 2.0 T | 1.6 MPa |
| 500 kA | 5 mm | 20 T | 160 MPa |
| 1 MA | 1 mm | 200 T | 16 GPa |

Note: 1 atmosphere ≈ 0.1 MPa, so DPF pinches can reach millions of atmospheres!

---

## 2. Magnetic Field of a Current Column

### 2.1 Azimuthal Field

For a cylindrical current channel with uniform current density, the magnetic field is:

**Outside the column** ($r > a$):
$$B_\theta(r) = \frac{\mu_0 I}{2\pi r}$$

**Inside the column** ($r < a$):
$$B_\theta(r) = \frac{\mu_0 I r}{2\pi a^2}$$

where $a$ is the column radius and $I$ is the total current.

### 2.2 Magnetic Field Profile

```
    B_θ
     │
     │        ┌─ B_θ = μ₀I/(2πr)  [outside]
     │       /
     │      /
     │     /
     │    / ← B_θ = μ₀Ir/(2πa²)  [inside]
     │   /
     │──/───────────────────────► r
        0   a
            │
         Column
         radius
```

---

## 3. The Z-Pinch Configuration

### 3.1 Geometry

In a z-pinch (the fundamental geometry of DPF):

- Current flows axially (z-direction)
- Magnetic field is azimuthal (θ-direction)
- Pressure gradient is radial (r-direction)

```
      ┌────────────┐
      │   ↑ J_z    │
      │   │        │
      │ ──┼── B_θ  │ ⊙ into page
      │   │        │
      │   ↓        │
      └────────────┘
           ←── F_r (inward)
           
    Axial View:
    
         ⊙ J (out)
        ╱   ╲
       ↓     ↓
    ← F     F →  
       ↑     ↑
        ╲   ╱
      B_θ circles
```

### 3.2 Force Balance

The radial force per unit volume on the plasma is:

$$f_r = J_z B_\theta = -\frac{1}{\mu_0} B_\theta \frac{d B_\theta}{dr} - \frac{B_\theta^2}{\mu_0 r}$$

This can be written as:

$$f_r = -\frac{d}{dr}\left(\frac{B_\theta^2}{2\mu_0}\right) - \frac{B_\theta^2}{\mu_0 r}$$

The first term is the magnetic pressure gradient; the second is the magnetic tension.

---

## 4. Equilibrium Conditions

### 4.1 MHD Equilibrium

In steady state, the momentum equation becomes:

$$\nabla P = \vec{J} \times \vec{B}$$

For a z-pinch in cylindrical coordinates:

$$\frac{dP}{dr} = -J_z B_\theta$$

### 4.2 Pressure Balance

Integrating across the plasma column:

$$P_{plasma}(r=0) - P_{plasma}(r=a) = \int_0^a J_z B_\theta \, dr$$

For a sharp-boundary model with zero external pressure:

$$P_0 = \frac{B_\theta^2(a)}{2\mu_0} = \frac{\mu_0 I^2}{8\pi^2 a^2}$$

where $P_0$ is the pressure on axis.

### 4.3 The Bennett Pinch

For an isothermal plasma column with temperature $T$:

$$P = n k_B T (1 + Z)$$

where $Z$ is the ionization state. This leads to the Bennett relation (covered in detail in the next module):

$$I^2 = \frac{8\pi k_B T}{\mu_0} N_l$$

---

## 5. Pinch Dynamics

### 5.1 Radial Compression

The pinch effect compresses plasma inward. The equation of motion for a thin shell is:

$$\rho \frac{d^2 r}{dt^2} = -\frac{B_\theta^2}{2\mu_0 r}$$

### 5.2 Characteristic Timescale

The **Alfvén time** characterizes pinch dynamics:

$$\tau_A = \frac{a}{v_A} = a \sqrt{\frac{\mu_0 \rho}{B^2}}$$

For typical DPF parameters:
- $a = 1$ mm
- $\rho = 10^{-4}$ kg/m³
- $B = 50$ T

$$\tau_A \approx 10 \text{ ns}$$

### 5.3 Compression Ratio

The final pinch radius is limited by:

1. **Plasma pressure**: As density increases, pressure resists compression
2. **Temperature rise**: Adiabatic compression heats the plasma
3. **Instabilities**: MHD instabilities can disrupt the pinch

Typical compression ratios: 10:1 to 100:1

---

## 6. Pinch Instabilities

### 6.1 Sausage Instability (m=0)

The sausage mode causes axial variations in radius:

```
    Before:          After:
    ┌────────┐      ╭──╮  ╭──╮
    │        │  →   │  ╰──╯  │
    │        │      │  ╭──╮  │
    └────────┘      ╰──╯  ╰──╯
```

Growth rate: $\gamma = k v_A$ where $k$ is the wavenumber.

### 6.2 Kink Instability (m=1)

The kink mode causes lateral displacement:

```
    Before:          After:
    ┌────────┐         ╱──╲
    │        │  →     /    \
    │        │       ╲    ╱
    └────────┘        ╲──╱
```

Stabilized by axial magnetic field when $B_z > B_\theta$.

### 6.3 Rayleigh-Taylor Instability

Occurs during rapid deceleration:

$$\gamma = \sqrt{kg}$$

where $g$ is the deceleration and $k$ is the wavenumber.

---

## 7. Application to DPF

### 7.1 Current Sheath Formation

The DPF discharge creates a current sheath that:

1. Is driven by magnetic pressure behind it
2. Sweeps up and ionizes the fill gas
3. Accelerates toward the axis during radial collapse

### 7.2 Final Focus

At the end of the discharge:

$$P_{magnetic} = \frac{B^2}{2\mu_0} \sim \frac{\mu_0 I^2}{8\pi^2 a^2}$$

With $I = 1$ MA and $a = 1$ mm:

$$P_{magnetic} \approx 40 \text{ GPa} \approx 400,000 \text{ atm}$$

### 7.3 Energy Balance

The magnetic energy per unit length:

$$U_B = \frac{1}{2} L' I^2$$

where $L' \approx \frac{\mu_0}{2\pi} \ln\left(\frac{b}{a}\right)$ is the inductance per unit length.

---

## 8. Practical Calculations

### Example 1: Magnetic Field Calculation

**Problem**: Calculate the magnetic field at the surface of a 2 mm radius plasma column carrying 500 kA.

**Solution**:

$$B_\theta = \frac{\mu_0 I}{2\pi a} = \frac{4\pi \times 10^{-7} \times 5 \times 10^5}{2\pi \times 0.002} = 50 \text{ T}$$

### Example 2: Magnetic Pressure

**Problem**: Calculate the magnetic pressure for the above case.

**Solution**:

$$P_B = \frac{B^2}{2\mu_0} = \frac{50^2}{2 \times 4\pi \times 10^{-7}} = 1.0 \times 10^9 \text{ Pa} = 1 \text{ GPa}$$

### Example 3: Equilibrium Temperature

**Problem**: If the plasma density is $n = 10^{25}$ m⁻³ and fully ionized deuterium ($Z=1$), what temperature is needed for equilibrium?

**Solution**:

$$P = 2 n k_B T = P_B$$

$$T = \frac{P_B}{2 n k_B} = \frac{10^9}{2 \times 10^{25} \times 1.38 \times 10^{-23}} = 3.6 \times 10^6 \text{ K} \approx 310 \text{ eV}$$

---

## 9. Summary

Key points to remember:

1. **Magnetic pressure** $P_B = B^2/(2\mu_0)$ drives plasma compression
2. The **z-pinch** geometry creates an inward radial force
3. **Equilibrium** requires balance between magnetic and kinetic pressure
4. **Instabilities** limit compression but can enhance fusion yield
5. DPF achieves extreme pressures (GPa) through current focusing

---

## Mathematical Reference

### Vector Identities

$$\vec{J} \times \vec{B} = \frac{1}{\mu_0}(\vec{B} \cdot \nabla)\vec{B} - \nabla\left(\frac{B^2}{2\mu_0}\right)$$

### Useful Formulas

| Quantity | Formula |
|----------|---------|
| Magnetic pressure | $P_B = B^2/(2\mu_0)$ |
| Surface field | $B_\theta = \mu_0 I/(2\pi a)$ |
| Alfvén velocity | $v_A = B/\sqrt{\mu_0\rho}$ |
| Alfvén time | $\tau_A = a/v_A$ |

---

## Further Reading

1. Freidberg, J.P. "Ideal MHD" (Chapter 5)
2. Ryutov, D.D. "Characterizing the Pinch" (Review of Modern Physics)
3. Liberman, M.A. "Physics of High-Density Z-Pinch Plasmas"

---

## Exercises

1. Calculate the magnetic pressure at the surface of the pinch for various currents (100 kA to 2 MA) and plot the result.
2. Derive the expression for the inductance per unit length of a coaxial pinch.
3. Estimate the Alfvén transit time for a DPF pinch with $a=0.5$ mm, $n=10^{25}$ m⁻³, $B=100$ T.

---

*Previous: [Ionization Primer](ionization_primer.md) | Next: [Bennett Relation](bennett_relation.md)*
