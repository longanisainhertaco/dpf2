# Advanced Physics Tutorial

Explore high-fidelity plasma models and optional physics packages.

## 1. Enable extended models

Activate the Hall MHD solver and radiation transport by installing the
appropriate extras:

```bash
pip install .[radiation]
```

Update your configuration to switch on the modules:

```yaml
physics:
  hall_mhd: true
  radiation: true
```

## 2. Run with detailed diagnostics

The additional physics can slow the solver, so enable only the
instrumentation you need:

```bash
dpf2 simulate -c advanced.yml -o run --diagnostics
```

Current and voltage traces now include the effects of the activated
models. Inspect the history file or use the plotting helpers to compare
runs.
