# Lab-mode guided walkthrough

This tutorial links the lab-mode UI controls to the underlying plasma physics
they influence. It pairs each visual element with a short experiment you can
repeat shot-to-shot while recording manifests for reproducibility.

## Jitter-aware shot series

1. Open the Lab Mode panel and enable **Jitter**.
2. Adjust **Switch Jitter (ns)** to introduce trigger timing variation. Observe
   how earlier triggers increase the current ramp rate and how the pinch time
   shifts in the KPI plots.
3. Increase **Pressure Jitter (%)** to broaden the fill gas distribution. Runs
   with higher fill tend to quench current rise sooner, lowering peak current
   and yield.
4. Export the manifest bundle after 10–20 shots. Each `run_manifest.json`
   contains the random seeds and device state so you can replay the exact shot
   series on another cluster.

## Voltage/pressure sweeps with project management

1. Use the **Project Manager** to import or define two configuration sets
   (e.g. different anode radii).
2. Launch the **Pareto (yield vs. pressure/voltage)** search from the CLI:
   ```bash
   dpf2 pareto-opt --config config.json --pressure-bounds 2.5:6.0 --voltage-bounds 10e3:18e3 --output pareto_lab
   ```
3. In the UI, select both projects. The comparison cards highlight the best
   yield and wall-plug efficiency found on the Pareto front, alongside your
   notebook notes for diagnostics alignment.

## Diagnostics overlays

* The **Voltage/Pressure sliders** update the sheath-beam overlay in real time.
  Link this to the **Regime Dashboard** to see when you enter unstable MHD
  regimes at high voltage/pressure corners.
* The **Instability Visualiser** shows mode growth as you increase jitter.
  Use it to correlate shot-to-shot variations with the recorded jitter values
  in each manifest.

## Tips for reproducibility

* Always enable **Lab Mode** when running sweeps or Pareto searches so every
  shot writes a manifest. These include the container hash and dataset hashes
  when `--datasets` is provided.
* Store generated scaling plots (`strong.png`, `weak.png`, `hdf5_io.png`) with
  their DOIs in `run_manifest.json` when publishing HPC images.
