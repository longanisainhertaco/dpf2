# Experimental Shot High-Resolution Trace

The file `experimental_shot_highres.json` records a 1 µs discharge of a series
RLC circuit sampled every 10 ns. The trace was produced with the analytic
RLC solver in `dpf2` using a 1 µH inductor, 2 mΩ resistor, 0.5 µF capacitor
and an initial capacitor voltage of 1 kV. In addition to time, current and
voltage, synthetic pressure and temperature diagnostics are included together
with the integrated neutron yield. This dataset acts as a surrogate for a
high-speed oscilloscope capture and is used for regression testing.
