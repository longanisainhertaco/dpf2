# PF1000 Instability Validation

Two lightweight comparison scripts validate simplified instability models
against PF1000 current and voltage traces (`Reference/PF1000/shot001.csv`).

* `benchmarks/PF1000/m0_instability_compare.py`
* `benchmarks/PF1000/lower_hybrid_drift_compare.py`

Each script predicts voltage spikes from the respective model, plots the
predicted waveform against the measured trace, and prints any time points with
more than 20% deviation.  The resulting plots are saved alongside the scripts
for quick inspection.
