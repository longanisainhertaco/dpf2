# Quiz: Plasma Fundamentals

This quiz assesses your understanding of plasma physics fundamentals as they apply to Dense Plasma Focus (DPF) devices.

---

## Instructions

- Answer all questions to the best of your ability
- Show your work for calculation problems
- For multiple choice, select the best answer
- Refer to the educational materials as needed for review

---

## Section A: Ionization Concepts (25 points)

### Question A1 (5 points)

**What is ionization and why is it essential for plasma formation?**

*Write a brief explanation (2-3 sentences).*

<details>
<summary>Answer</summary>

Ionization is the process of removing electrons from neutral atoms or molecules, creating positively charged ions and free electrons. It is essential for plasma formation because plasma is defined as a quasi-neutral gas of charged particles that exhibits collective behavior. Without ionization, the gas remains electrically neutral with no free charge carriers.

</details>

---

### Question A2 (5 points)

**The Saha equation describes ionization equilibrium. Match each term in the equation with its physical meaning:**

$$\frac{n_{i+1} n_e}{n_i} = \frac{2}{\Lambda^3} \frac{g_{i+1}}{g_i} \exp\left(-\frac{E_{ion}}{k_B T}\right)$$

| Term | Options |
|------|---------|
| $n_e$ | a) Ionization energy |
| $E_{ion}$ | b) Statistical weights |
| $g_i, g_{i+1}$ | c) Electron density |
| $\Lambda$ | d) Thermal de Broglie wavelength |
| $T$ | e) Temperature |

<details>
<summary>Answer</summary>

- $n_e$ → c) Electron density
- $E_{ion}$ → a) Ionization energy
- $g_i, g_{i+1}$ → b) Statistical weights
- $\Lambda$ → d) Thermal de Broglie wavelength
- $T$ → e) Temperature

</details>

---

### Question A3 (5 points)

**Multiple Choice: At what temperature is hydrogen approximately 50% ionized at a density of $10^{21}$ m⁻³?**

- [ ] A) 0.1 eV
- [ ] B) 0.8 eV
- [ ] C) 5.0 eV
- [ ] D) 13.6 eV

<details>
<summary>Answer</summary>

**B) 0.8 eV**

At 50% ionization, $\alpha = 0.5$, and the Saha equation gives approximately $T \approx 0.8$ eV for $n = 10^{21}$ m⁻³. This is much lower than the ionization energy (13.6 eV) because thermal ionization occurs in the tail of the Maxwellian distribution, and at this density, relatively few electrons are needed to achieve 50% ionization.

</details>

---

### Question A4 (5 points)

**Calculation: Calculate the ionization fraction of deuterium plasma at:**
- Temperature: T = 3 eV
- Total density: n = 10²³ m⁻³
- Ionization energy: E_i = 13.6 eV

*Hint: Use the simplified Saha function and solve for α.*

<details>
<summary>Answer</summary>

Using the Saha equation in the form:
$$\frac{\alpha^2}{1-\alpha} = \frac{S(T)}{n_{total}}$$

where $S(T) = 2.4 \times 10^{21} \cdot T^{3/2} \cdot \exp(-E_i/k_B T)$

With T = 3 eV and $k_B T$ = 3 eV:
- $T^{3/2} = (3 \times 11605)^{3/2} \approx 2.1 \times 10^7$ K$^{3/2}$
- $\exp(-13.6/3) = \exp(-4.53) \approx 0.011$

$S(T) = 2.4 \times 10^{21} \times 2.1 \times 10^7 \times 0.011 \approx 5.5 \times 10^{26}$ m⁻³

$\frac{\alpha^2}{1-\alpha} = \frac{5.5 \times 10^{26}}{10^{23}} = 5500$

Solving: $\alpha^2 = 5500(1-\alpha)$
$\alpha^2 + 5500\alpha - 5500 = 0$

$\alpha \approx 0.999$ or approximately **99.9% ionized**

</details>

---

### Question A5 (5 points)

**In DPF operation, list the three main ionization mechanisms and briefly describe when each is most important.**

<details>
<summary>Answer</summary>

1. **Electron Impact Ionization**: Most important during the breakdown and rundown phases when energetic electrons collide with neutral gas atoms. This is the primary ionization mechanism that converts the fill gas into plasma.

2. **Photoionization**: Becomes important in the dense pinch column where intense X-ray emission can ionize nearby neutral gas or lower ionization states. Most significant during the pinch phase.

3. **Field Ionization**: Important during the initial breakdown when strong electric fields at electrode surfaces lower the ionization barrier. Also relevant in regions of high electric field gradients in the plasma.

</details>

---

## Section B: Magnetic Pressure (25 points)

### Question B1 (5 points)

**Write the formula for magnetic pressure and explain its physical origin.**

<details>
<summary>Answer</summary>

$$P_B = \frac{B^2}{2\mu_0}$$

Physical origin: Magnetic pressure arises from the Lorentz force acting on current-carrying plasma. When current flows through the plasma, it generates a magnetic field. The interaction between the current density $\vec{J}$ and the magnetic field $\vec{B}$ produces a force $\vec{F} = \vec{J} \times \vec{B}$. In a z-pinch geometry, this force is directed radially inward, compressing the plasma. The magnetic pressure represents the energy density of the magnetic field that drives this compression.

</details>

---

### Question B2 (5 points)

**Calculation: A plasma column carries 300 kA of current and has a radius of 2 mm. Calculate:**

a) The magnetic field at the surface of the column
b) The magnetic pressure

<details>
<summary>Answer</summary>

a) Magnetic field at surface:
$$B_\theta = \frac{\mu_0 I}{2\pi a} = \frac{4\pi \times 10^{-7} \times 3 \times 10^5}{2\pi \times 0.002}$$
$$B_\theta = \frac{4 \times 10^{-7} \times 3 \times 10^5}{2 \times 0.002} = \frac{0.12}{0.004} = 30 \text{ T}$$

b) Magnetic pressure:
$$P_B = \frac{B^2}{2\mu_0} = \frac{30^2}{2 \times 4\pi \times 10^{-7}} = \frac{900}{8\pi \times 10^{-7}}$$
$$P_B = \frac{900}{2.51 \times 10^{-6}} \approx 3.6 \times 10^8 \text{ Pa} = 360 \text{ MPa}$$

This is about 3,600 atmospheres!

</details>

---

### Question B3 (5 points)

**Multiple Choice: In a z-pinch, what is the direction of the magnetic force on the plasma?**

- [ ] A) Axial (along the current flow)
- [ ] B) Azimuthal (around the axis)
- [ ] C) Radially inward (toward the axis)
- [ ] D) Radially outward (away from the axis)

<details>
<summary>Answer</summary>

**C) Radially inward (toward the axis)**

In a z-pinch, the current flows axially ($\vec{J}$ in the z-direction) and creates an azimuthal magnetic field ($\vec{B}$ in the θ-direction). The Lorentz force $\vec{F} = \vec{J} \times \vec{B}$ is therefore in the radial direction, pointing inward. This inward force compresses the plasma toward the axis—the pinch effect.

</details>

---

### Question B4 (5 points)

**The Alfvén velocity is given by $v_A = B/\sqrt{\mu_0 \rho}$. Explain why this velocity is important for pinch dynamics.**

<details>
<summary>Answer</summary>

The Alfvén velocity is the characteristic speed at which magnetic disturbances propagate through a magnetized plasma. It is important for pinch dynamics because:

1. **Compression timescale**: The Alfvén transit time $\tau_A = a/v_A$ (where $a$ is the pinch radius) sets the characteristic timescale for radial compression and equilibration.

2. **Instability growth**: MHD instabilities (sausage, kink) grow on timescales related to the Alfvén time. The growth rate $\gamma \sim k v_A$ determines how quickly instabilities develop.

3. **Information propagation**: Magnetic signals travel at $v_A$, so the plasma cannot respond to perturbations faster than this speed.

4. **Energy coupling**: The efficiency of magnetic energy transfer to kinetic energy depends on how the plasma velocity compares to $v_A$.

</details>

---

### Question B5 (5 points)

**True or False with explanation:**

"In a DPF pinch with 1 MA of current compressed to 1 mm radius, the magnetic pressure exceeds 1 GPa."

<details>
<summary>Answer</summary>

**TRUE**

Calculation:
$$B = \frac{\mu_0 I}{2\pi a} = \frac{4\pi \times 10^{-7} \times 10^6}{2\pi \times 10^{-3}} = \frac{4 \times 10^{-1}}{2 \times 10^{-3}} = 200 \text{ T}$$

$$P_B = \frac{B^2}{2\mu_0} = \frac{200^2}{2 \times 4\pi \times 10^{-7}} = \frac{4 \times 10^4}{2.51 \times 10^{-6}} \approx 1.6 \times 10^{10} \text{ Pa} = 16 \text{ GPa}$$

This is actually much larger than 1 GPa, demonstrating the extreme pressures achievable in DPF devices.

</details>

---

## Section C: Bennett Relation (25 points)

### Question C1 (5 points)

**State the Bennett relation and identify what each symbol represents.**

<details>
<summary>Answer</summary>

The Bennett relation is:

$$I^2 = \frac{8\pi k_B T}{\mu_0} N_l (1+Z)$$

Where:
- $I$ = total current through the plasma column (A)
- $k_B$ = Boltzmann constant (1.38 × 10⁻²³ J/K)
- $T$ = plasma temperature (K)
- $\mu_0$ = permeability of free space (4π × 10⁻⁷ H/m)
- $N_l$ = line density (particles per unit length, m⁻¹)
- $Z$ = average ionization state
- $(1+Z)$ = accounts for both ions and electrons

</details>

---

### Question C2 (5 points)

**What physical equilibrium does the Bennett relation describe?**

<details>
<summary>Answer</summary>

The Bennett relation describes **magnetohydrostatic equilibrium** in a z-pinch plasma column. Specifically, it represents the balance between:

1. **Inward magnetic pressure**: The $\vec{J} \times \vec{B}$ force that compresses the plasma

2. **Outward thermal pressure**: The kinetic pressure $P = nk_BT(1+Z)$ that resists compression

When these forces are balanced, the plasma column is in equilibrium—it neither expands nor contracts. The Bennett relation gives the specific relationship between current, temperature, and density required for this equilibrium state.

Note: This is an idealized steady-state condition. Real DPF plasmas are highly dynamic and may not achieve true Bennett equilibrium.

</details>

---

### Question C3 (10 points)

**Calculation: A DPF achieves a peak current of 600 kA with a line density of $4 \times 10^{18}$ particles/m. Assuming fully ionized deuterium (Z=1), calculate:**

a) The Bennett equilibrium temperature (in eV)
b) The Bennett equilibrium temperature (in Kelvin)
c) Is this temperature sufficient for significant D-D fusion?

<details>
<summary>Answer</summary>

a) Rearranging the Bennett relation:
$$T = \frac{\mu_0 I^2}{8\pi k_B N_l (1+Z)}$$

$$T = \frac{4\pi \times 10^{-7} \times (6 \times 10^5)^2}{8\pi \times 1.38 \times 10^{-23} \times 4 \times 10^{18} \times 2}$$

$$T = \frac{4\pi \times 10^{-7} \times 3.6 \times 10^{11}}{8\pi \times 1.38 \times 10^{-23} \times 8 \times 10^{18}}$$

$$T = \frac{4 \times 3.6 \times 10^4}{8 \times 1.38 \times 8 \times 10^{-5}}$$

$$T = \frac{1.44 \times 10^5}{8.83 \times 10^{-4}} \approx 1.63 \times 10^8 \text{ K}$$

b) Converting to eV:
$$T_{eV} = \frac{T_K}{11605} = \frac{1.63 \times 10^8}{11605} \approx 14,000 \text{ eV} = 14 \text{ keV}$$

c) **Yes**, this temperature is sufficient for significant D-D fusion. The D-D fusion cross-section becomes significant above about 5-10 keV and peaks around 100 keV. At 14 keV, the reactivity $\langle\sigma v\rangle$ is substantial (order of 10⁻²⁴ m³/s), enabling measurable fusion reactions.

</details>

---

### Question C4 (5 points)

**The neutron yield from DPF devices often scales as $Y_n \propto I^4$. Derive this scaling law using the Bennett relation.**

*Hint: Start with fusion rate ∝ n²⟨σv⟩ and use Bennett to relate n and T to I.*

<details>
<summary>Answer</summary>

Starting assumptions:
- Bennett: $nT \propto I^2/a^2$ (where $a$ is pinch radius)
- Fusion rate: $R \propto n^2 \langle\sigma v\rangle$
- For D-D below ~20 keV: $\langle\sigma v\rangle \propto T^2$ approximately

Step 1: From Bennett, $n \propto I^2/(a^2 T)$

Step 2: Assuming pinch dynamics keep $nT$ proportional to $I^2/a^2$:
$$n^2 \propto \frac{I^4}{a^4 T^2}$$

Step 3: The fusion rate:
$$R \propto n^2 \langle\sigma v\rangle \propto \frac{I^4}{a^4 T^2} \cdot T^2 = \frac{I^4}{a^4}$$

Step 4: The pinch volume scales as $V \propto a^2 \cdot l$ where $l$ is length

Step 5: Total yield:
$$Y \propto R \cdot V \propto \frac{I^4}{a^4} \cdot a^2 = \frac{I^4}{a^2}$$

Step 6: If $a$ scales weakly with $I$ (or is approximately constant for a given device geometry):
$$Y_n \propto I^4$$

This explains the strong dependence of neutron yield on current observed in DPF experiments.

</details>

---

## Section D: Safety (25 points)

### Question D1 (5 points)

**List three types of electrical hazards specific to DPF experiments.**

<details>
<summary>Answer</summary>

1. **Electrocution from high voltage**: DPF systems operate at 15-50 kV, far exceeding the lethal threshold. Contact with charged components can cause immediate death.

2. **Capacitor bank hazards**: Capacitors store kilojoules of energy and can deliver lethal shocks even after power is disconnected. Dielectric absorption can cause voltage recovery after discharge.

3. **Arc flash**: Fault conditions can produce intense arc flashes with temperatures exceeding 10,000°C, causing severe burns, UV exposure, and blast injuries.

Other valid answers include: step/touch potential, induced voltages, electromagnetic interference, fire from cable heating.

</details>

---

### Question D2 (5 points)

**What is LOTO and why is it critical for DPF safety?**

<details>
<summary>Answer</summary>

**LOTO** stands for **Lockout/Tagout**, a safety procedure required by OSHA (29 CFR 1910.147) to control hazardous energy.

It is critical for DPF safety because:

1. **Multiple energy sources**: DPF systems have multiple isolation points (power supply, trigger system, capacitor bank) that must all be secured.

2. **Stored energy**: Even with power off, capacitors retain lethal charge. LOTO ensures systematic verification and grounding.

3. **Personnel protection**: Individual locks prevent accidental re-energization while workers are in contact with equipment.

4. **Clear communication**: Tags indicate who is working on the system and why, preventing premature restoration of power.

5. **Legal requirement**: LOTO is mandatory under OSHA regulations for work on equipment with potential for hazardous energy release.

</details>

---

### Question D3 (5 points)

**Multiple Choice: What radiation types are produced by DPF devices? (Select all that apply)**

- [ ] A) X-rays from bremsstrahlung
- [ ] B) Neutrons from D-D fusion
- [ ] C) Alpha particles from uranium decay
- [ ] D) Gamma rays from neutron activation
- [ ] E) Electron beams from runaway electrons

<details>
<summary>Answer</summary>

**A, B, D, and E are correct**

- ✓ **A) X-rays from bremsstrahlung**: Electrons decelerating in the plasma produce continuous X-ray spectra
- ✓ **B) Neutrons from D-D fusion**: 2.45 MeV neutrons are produced when deuterium fuses
- ✗ **C) Alpha particles from uranium decay**: This is not relevant to DPF operation (no uranium present)
- ✓ **D) Gamma rays from neutron activation**: Fast neutrons can activate materials, which then emit gamma rays as they decay
- ✓ **E) Electron beams from runaway electrons**: High-energy electron beams are generated and produce hard X-rays when striking electrodes

</details>

---

### Question D4 (5 points)

**Describe the multi-layer approach to neutron shielding and explain why each layer is necessary.**

<details>
<summary>Answer</summary>

Neutron shielding requires three layers:

**Layer 1: Moderator** (e.g., polyethylene, paraffin, water)
- Purpose: Slow down fast neutrons through elastic collisions with hydrogen nuclei
- Why needed: Fast neutrons (2.45 MeV) have low capture cross-sections; they must be slowed to thermal energies (~0.025 eV) for effective capture
- Typical thickness: 10-30 cm

**Layer 2: Absorber** (e.g., borated polyethylene, cadmium, B₄C)
- Purpose: Capture thermal neutrons
- Why needed: Thermal neutrons have high capture cross-sections for certain isotopes (¹⁰B, ¹¹³Cd); capture prevents transmission through shield
- Typical thickness: 1-5 mm cadmium or mixed with moderator

**Layer 3: Gamma Shield** (e.g., lead, steel)
- Purpose: Absorb gamma rays produced during neutron capture
- Why needed: When neutrons are captured, the nucleus emits high-energy gamma rays (capture gammas) that must be attenuated
- Typical thickness: 2-5 cm lead

</details>

---

### Question D5 (5 points)

**A student is preparing for a DPF experiment. List five items that should be on their pre-shot safety checklist.**

<details>
<summary>Answer</summary>

Any five of the following are acceptable:

1. All personnel accounted for and in designated safe zone
2. Experimental area clear and all access doors closed
3. Door interlocks verified functional
4. All personnel wearing appropriate dosimeters
5. Area radiation monitors operational and reading background
6. Shielding properly positioned and secured
7. Fire extinguisher accessible and charged
8. Emergency stop reachable and functional
9. Countdown announced on PA system
10. Hearing protection worn by all observers
11. Shot logged with date, time, and operator name
12. Neutron yield estimate within permitted limits
13. Accumulated dose for all personnel below regulatory limits
14. Emergency procedures accessible and reviewed

</details>

---

## Scoring Guide

| Section | Points Available |
|---------|-----------------|
| A: Ionization Concepts | 25 |
| B: Magnetic Pressure | 25 |
| C: Bennett Relation | 25 |
| D: Safety | 25 |
| **Total** | **100** |

### Grade Scale

| Score | Grade |
|-------|-------|
| 90-100 | A |
| 80-89 | B |
| 70-79 | C |
| 60-69 | D |
| <60 | F |

---

## Further Study

If you found any section challenging, review the corresponding educational materials:

- **Ionization**: [Ionization Primer](../plasma_fundamentals/ionization_primer.md)
- **Magnetic Pressure**: [Magnetic Pressure](../plasma_fundamentals/magnetic_pressure.md)
- **Bennett Relation**: [Bennett Relation](../plasma_fundamentals/bennett_relation.md)
- **Safety**: [Electrical Hazards](../safety/electrical_hazards.md) and [Radiation Awareness](../safety/radiation_awareness.md)
