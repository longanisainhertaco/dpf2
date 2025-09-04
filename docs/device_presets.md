# Device Presets

DPF2 includes a small library of Dense Plasma Focus (DPF) geometries located
in the `device_profiles/` directory.  These presets provide starting points for
common educational and industrial machines and may be selected from the CLI or
GUI.  All dimensions below are given in centimetres unless noted otherwise.

## EDU1K – 1 kJ Educational DPF

| Parameter | Value |
|-----------|-------|
| Energy | 1 kJ |
| Anode radius | 0.5 cm |
| Cathode radius | 1.5 cm |
| Anode length | 7.0 cm |

Scaling laws:

- Voltage follows \(V = \sqrt{2E/C}\) for capacitor energy \(E\).
- To maintain similarity, electrode radii scale with \(\sqrt{E}\); lengths
  should scale roughly three times the anode radius.

## IND20K – 20 kJ Industrial DPF

| Parameter | Value |
|-----------|-------|
| Energy | 20 kJ |
| Anode radius | 1.5 cm |
| Cathode radius | 4.5 cm |
| Anode length | 20.0 cm |

Scaling laws:

- Pinch current obeys \(I \propto \sqrt{C}\,V\).
- Geometric dimensions scale approximately with \(\sqrt{E}\) to preserve
  current density.

## PF1000 – MJ-scale Reference

| Parameter | Value |
|-----------|-------|
| Energy | 500 kJ |
| Anode radius | 2.5 cm |
| Cathode radius | 6.0 cm |
| Anode length | 16.0 cm |

Scaling laws:

- Maintaining aspect ratio \(L \approx 6.4 r_a\) preserves rundown dynamics.
- Neutron yield scales roughly with \(I^4\) in this regime.

These profiles can be extended or customised by editing the JSON files under
`device_profiles/`.
