# Radiation Safety for DPF Experiments

This module covers radiation safety awareness for Dense Plasma Focus (DPF) experiments, focusing on X-ray and neutron hazards.

## ⚠️ IMPORTANT SAFETY NOTICE

**DPF devices produce ionizing radiation. Work with DPF equipment only under qualified supervision and with proper radiation safety training.**

---

## Learning Objectives

After completing this module, you will be able to:

1. Identify radiation sources in DPF experiments
2. Understand X-ray production mechanisms and hazards
3. Recognize neutron radiation sources and shielding requirements
4. Apply basic radiation protection principles
5. Use dosimetry equipment properly

---

## 1. Overview of DPF Radiation

### 1.1 Radiation Types Produced

DPF devices produce several types of ionizing radiation:

```
┌─────────────────────────────────────────────────────────────────┐
│                    DPF RADIATION SOURCES                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  X-RAYS (Photons)                                               │
│  ├── Bremsstrahlung: Electron deceleration in plasma            │
│  ├── Characteristic: K-shell emission from target materials     │
│  └── Energy range: 1 keV - 1 MeV                                │
│                                                                  │
│  NEUTRONS                                                        │
│  ├── D-D fusion: 2.45 MeV neutrons                              │
│  ├── D-T fusion: 14.1 MeV neutrons (if tritium present)         │
│  └── Beam-target: Directional emission                          │
│                                                                  │
│  OTHER RADIATION                                                 │
│  ├── Gamma rays: From neutron activation                        │
│  ├── Electron beams: Runaway electrons, ~100 keV - 1 MeV        │
│  └── Ion beams: Deuterium ions, ~100 keV - 1 MeV                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Radiation Quantities

| Quantity | Symbol | Unit | Meaning |
|----------|--------|------|---------|
| Activity | A | Becquerel (Bq) | Disintegrations per second |
| Absorbed dose | D | Gray (Gy) | Energy deposited per kg |
| Equivalent dose | H | Sievert (Sv) | Biological effect |
| Exposure | X | Roentgen (R) | Air ionization |
| Fluence | Φ | n/cm² | Particles per area |

### 1.3 Typical DPF Radiation Output

| DPF Class | Neutron Yield | X-ray Dose/Shot | Primary Hazard |
|-----------|---------------|-----------------|----------------|
| Small (<1 kJ) | 10⁴ - 10⁶ | <1 mSv at 1 m | X-rays |
| Medium (1-10 kJ) | 10⁷ - 10⁹ | 1-10 mSv at 1 m | X-rays + Neutrons |
| Large (>100 kJ) | 10¹⁰ - 10¹² | >10 mSv at 1 m | Neutrons |

---

## 2. X-Ray Production Mechanisms

### 2.1 Bremsstrahlung Radiation

**Bremsstrahlung** ("braking radiation") occurs when electrons decelerate in the plasma:

$$P_{brem} = 1.69 \times 10^{-32} n_e n_i Z^2 T_e^{1/2} \quad \text{W/m}^3$$

where:
- $n_e$ = electron density (m⁻³)
- $n_i$ = ion density (m⁻³)
- $Z$ = atomic number
- $T_e$ = electron temperature (eV)

The spectrum is continuous with:
$$\frac{dP}{dE} \propto \exp\left(-\frac{E}{k_B T_e}\right)$$

### 2.2 Characteristic X-Rays

When electron beams strike electrode materials, characteristic X-rays are emitted:

| Material | Kα Energy | Kβ Energy |
|----------|-----------|-----------|
| Copper | 8.0 keV | 8.9 keV |
| Tungsten | 59.3 keV | 67.2 keV |
| Iron | 6.4 keV | 7.1 keV |

### 2.3 Hard X-Ray Production

DPF produces hard X-rays (>10 keV) through:

1. **Electron beams**: Accelerated by induced electric fields
2. **Runaway electrons**: Escape thermal population
3. **Anode interaction**: Beam strikes electrode

Hard X-ray energy can exceed 100 keV, requiring heavy shielding.

### 2.4 X-Ray Temporal Structure

```
    Intensity
        │
        │      ┌── Soft X-rays (pinch)
        │     ╱│
        │    ╱ │╲   Hard X-rays (beam)
        │   ╱  │ ╲
        │  ╱   │  ╲
        │ ╱    │   ╲
        ├╱─────┴────╲───────────► Time
        0    50   100   150 ns
            Pinch time
```

---

## 3. Neutron Production

### 3.1 Fusion Reactions

DPF neutrons are produced by deuterium fusion:

**D-D Fusion** (50% probability each branch):
$$D + D \rightarrow He^3 + n \quad (E_n = 2.45 \text{ MeV})$$
$$D + D \rightarrow T + p \quad (E_p = 3.0 \text{ MeV})$$

**D-T Fusion** (if tritium present):
$$D + T \rightarrow He^4 + n \quad (E_n = 14.1 \text{ MeV})$$

### 3.2 Neutron Emission Mechanisms

DPF neutron production has two components:

1. **Thermonuclear**: From hot, dense plasma
   - Isotropic emission
   - Spectrum peaked at 2.45 MeV

2. **Beam-target**: From accelerated ions hitting neutral gas
   - Anisotropic (forward-peaked)
   - Spectrum Doppler-shifted

### 3.3 Neutron Yield Scaling

The neutron yield scales with current as:

$$Y_n \propto I^4$$

| Current | Typical Yield |
|---------|---------------|
| 100 kA | 10⁵ - 10⁶ |
| 500 kA | 10⁸ - 10⁹ |
| 1 MA | 10¹⁰ - 10¹¹ |
| 2 MA | 10¹¹ - 10¹² |

### 3.4 Neutron Energy Spectrum

```
    Counts
        │
        │     ┌─── D-D peak (2.45 MeV)
        │    ╱│╲
        │   ╱ │ ╲
        │  ╱  │  ╲
        │ ╱   │   ╲    D-T peak (14.1 MeV)
        │╱    │    ╲         │
        ├─────┴─────╲────────┼───────► Energy
        0     2.45       10   14.1 MeV
```

---

## 4. Radiation Shielding

### 4.1 ALARA Principle

**A**s **L**ow **A**s **R**easonably **A**chievable

Apply the three principles of radiation protection:

1. **Time**: Minimize exposure duration
2. **Distance**: Maximize distance from source (inverse square law)
3. **Shielding**: Use appropriate materials

### 4.2 X-Ray Shielding

X-rays are attenuated exponentially:

$$I = I_0 \exp(-\mu x)$$

where $\mu$ is the linear attenuation coefficient.

**Half-Value Layer (HVL)**: Thickness to reduce intensity by 50%

| Material | HVL (100 keV) | HVL (500 keV) |
|----------|---------------|---------------|
| Lead | 0.1 mm | 4 mm |
| Concrete | 15 mm | 34 mm |
| Steel | 2.5 mm | 10 mm |
| Aluminum | 15 mm | 25 mm |

**Typical DPF X-ray shielding**: 2-5 mm lead equivalent

### 4.3 Neutron Shielding

Neutrons require multi-layer shielding:

```
┌─────────────────────────────────────────────────────────────────┐
│                    NEUTRON SHIELDING DESIGN                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  LAYER 1: MODERATOR (slow neutrons)                             │
│  └── Polyethylene, paraffin, water (hydrogen-rich)              │
│      Thickness: 10-30 cm                                         │
│                                                                  │
│  LAYER 2: ABSORBER (capture thermal neutrons)                   │
│  └── Borated polyethylene, cadmium, boron carbide               │
│      Thickness: 1-5 mm (Cd) or mixed with moderator             │
│                                                                  │
│  LAYER 3: GAMMA SHIELD (capture gamma from n-capture)           │
│  └── Lead, steel                                                │
│      Thickness: 2-5 cm                                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 Shielding Materials Summary

| Material | X-ray Effectiveness | Neutron Effectiveness |
|----------|--------------------|-----------------------|
| Lead | Excellent | Poor (activates) |
| Polyethylene | Poor | Excellent (moderator) |
| Borated poly | Poor | Excellent (mod + absorb) |
| Concrete | Good | Good (both) |
| Water | Poor | Good (moderator) |

### 4.5 Practical Shielding Example

For a 10⁸ neutron yield DPF:

**Unshielded dose at 1 m**: ~100 mSv/shot (dangerous!)

**Required shielding for 10 μSv/shot**:
- 25 cm borated polyethylene
- Plus 3 cm lead (for capture gammas)

**Or distance**: Move to 30 m (inverse square)

---

## 5. Dosimetry

### 5.1 Dose Limits

| Category | Annual Limit | Per-Shot Target |
|----------|--------------|-----------------|
| Occupational | 50 mSv (whole body) | <10 μSv |
| Extremities | 500 mSv | <100 μSv |
| Lens of eye | 150 mSv | <50 μSv |
| Public | 1 mSv | <1 μSv |

### 5.2 Dosimetry Equipment

| Device | Type | Range | Response Time |
|--------|------|-------|---------------|
| TLD badge | Passive | 0.1 mSv - 10 Sv | Post-exposure |
| Film badge | Passive | 0.1 mSv - 10 Sv | Post-exposure |
| Electronic dosimeter | Active | 0.1 μSv - 10 Sv | Real-time |
| Survey meter | Active | 0.1 μSv/h - 10 Sv/h | Real-time |
| Neutron rem meter | Active | 0.1 μSv/h - 100 mSv/h | Real-time |

### 5.3 Dosimeter Placement

```
     Front view:          Side view:
     
        ┌─┐                   │
       ╱───╲                 ╱│╲
      │  ●  │ ← Eye level   │ ● │ ← Chest badge
      │     │               │   │
      │  ◉  │ ← Chest       │   │
      │     │               │   │
     ╱│     │╲              │   │
    ╱ │     │ ╲             │   │
   ◉  │     │  ◉            │   │
      │     │               │   │
      │     │               │   │

   ● = Primary dosimeter (chest)
   ◉ = Secondary/extremity dosimeters
```

### 5.4 Dosimetry Procedures

1. **Before entry**: Check dosimeter battery/status
2. **During work**: Monitor real-time dose rate
3. **After exit**: Record cumulative dose
4. **Weekly**: Review dose logs
5. **Monthly**: Submit dosimeters for reading
6. **Annually**: Review total exposure

---

## 6. Radiation Monitoring

### 6.1 Area Monitoring

Fixed monitoring points around DPF:

```
                    ┌───────────┐
                    │  Control  │
                    │   Room    │
                    │   [M1]    │
                    └─────┬─────┘
                          │
    ┌─────┐         ┌─────┴─────┐         ┌─────┐
    │[M2] │─────────│    DPF    │─────────│[M3] │
    │     │         │  Chamber  │         │     │
    └─────┘         └─────┬─────┘         └─────┘
                          │
                    ┌─────┴─────┐
                    │  Forward  │
                    │   Zone    │
                    │   [M4]    │
                    └───────────┘

    [M1-M4]: Radiation monitoring stations
```

### 6.2 Survey Requirements

**Before shots**:
- Background radiation level
- Verify all monitors operational
- Check shielding integrity

**After shots**:
- Measure residual activation
- Survey beam dump area
- Check for contamination (if tritium used)

### 6.3 Alarm Levels

| Level | Action |
|-------|--------|
| Background (<1 μSv/h) | Normal operations |
| Elevated (1-10 μSv/h) | Investigate, reduce time |
| High (10-100 μSv/h) | Limit access, additional shielding |
| Very High (>100 μSv/h) | Evacuate, no entry |

---

## 7. Neutron Activation

### 7.1 Activation Concerns

Fast neutrons can activate materials, creating radioactive isotopes:

| Material | Reaction | Half-life | Gamma Energy |
|----------|----------|-----------|--------------|
| Copper | ⁶³Cu(n,γ)⁶⁴Cu | 12.7 h | 511 keV |
| Steel | ⁵⁶Fe(n,p)⁵⁶Mn | 2.6 h | 847 keV |
| Aluminum | ²⁷Al(n,γ)²⁸Al | 2.2 min | 1779 keV |
| Air (Ar) | ⁴⁰Ar(n,γ)⁴¹Ar | 1.8 h | 1294 keV |

### 7.2 Activation Precautions

1. **Wait time**: Allow decay before approach (typically 10+ minutes)
2. **Survey**: Check activation levels before handling components
3. **Material selection**: Avoid high-activation materials near chamber
4. **Ventilation**: Exhaust activated air from experimental area

### 7.3 Long-Term Activation

For high-yield DPF (>10¹⁰ n/shot):
- Structural materials may accumulate activity
- Periodic surveys required
- May require controlled disposal procedures

---

## 8. Emergency Procedures

### 8.1 Radiation Emergency Response

```
┌─────────────────────────────────────────────────────────────────┐
│              RADIATION EMERGENCY RESPONSE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. EVACUATE the area immediately                                │
│     ↓                                                            │
│  2. NOTIFY radiation safety officer                              │
│     ↓                                                            │
│  3. SECURE the area (prevent re-entry)                          │
│     ↓                                                            │
│  4. ASSESS exposure (read dosimeters)                           │
│     ↓                                                            │
│  5. DOCUMENT all personnel present and doses                    │
│     ↓                                                            │
│  6. SEEK MEDICAL attention if dose >100 mSv                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Contamination Response (Tritium)

If tritium is used in DPF experiments:

1. **Isolate**: Seal the area
2. **Ventilate**: Activate exhaust systems
3. **Survey**: Use tritium-specific detectors
4. **Decontaminate**: Follow facility procedures
5. **Bioassay**: Urinalysis for exposed personnel

---

## 9. Pre-Shot Radiation Safety Checklist

Before every DPF shot:

```
□ All personnel wearing appropriate dosimeters
□ Area radiation monitors operational and reading background
□ Shielding properly positioned and secured
□ Exclusion zone established and marked
□ All personnel in safe zone or behind shielding
□ Ventilation system operating (for activation)
□ Neutron yield estimate within permitted range
□ Accumulated dose for personnel below limits
□ Radiation safety officer notified (if required)
□ Emergency procedures reviewed and accessible
```

---

## 10. Summary

Key radiation safety principles:

1. **DPF produces ionizing radiation**: Both X-rays and neutrons
2. **X-rays dominate at low energies**: Shield with high-Z materials
3. **Neutrons require layered shielding**: Moderator + absorber + gamma shield
4. **Apply ALARA**: Time, distance, shielding
5. **Monitor exposure**: Personal dosimetry for all personnel
6. **Be aware of activation**: Wait before approaching after shots
7. **Know emergency procedures**: Practice regularly

---

## Regulatory References

- 10 CFR 20 - Standards for Protection Against Radiation (NRC)
- OSHA 29 CFR 1910.1096 - Ionizing Radiation
- ICRP Publication 103 - Recommendations on Radiological Protection
- NCRP Report 147 - Structural Shielding for Medical X-Ray Facilities

---

## Further Reading

1. Martin, J.E. "Physics for Radiation Protection" (3rd ed.)
2. IAEA Safety Standards Series - Radiation Protection
3. Knoll, G.F. "Radiation Detection and Measurement"

---

*Previous: [Electrical Hazards](electrical_hazards.md)*
