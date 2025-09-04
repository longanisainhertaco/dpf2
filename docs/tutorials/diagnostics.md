# Diagnostics Tutorial

Engineers can extract meaningful signals from simulations using these steps.

## 1. Collect detailed diagnostics

Run a simulation with extended logging enabled:

```bash
python -m dpf2.cli.main simulate -c config.json -o run --diagnostics --verbose
```

The `run` directory will contain JSON histories and waveform files for each shot.

## 2. Examine current and voltage traces

Plot electrical signals to verify circuit behavior:

```bash
python -m dpf2.cli.main plot-run --history run/history.json --output iv.png
```

Look for ringing or phase shifts that indicate wiring issues.

## 3. Inspect density and temperature profiles

Export profiles at peak compression:

```bash
python -m dpf2.cli.main diagnostics -i run -o analysis/ --profiles
```

Visualizing these profiles helps determine whether the pinch meets design goals.

## 4. Generate a concise report

Summarize metrics for review meetings:

```bash
python -m dpf2.cli.main report -i run -o report.pdf
```

The PDF combines plots, peak values, and configuration parameters for quick interpretation by the engineering team.
