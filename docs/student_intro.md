# Student-Friendly Introduction

Welcome to the dense plasma focus (DPF) project! This guide gives students a
conceptual overview of how a DPF compresses plasma. The animation below shows a
particle moving toward the pinch region as magnetic pressure increases.

![Plasma pinch animation](images/field_animation.svg)

The simulation follows a small plasma blob as it accelerates radially. When it
reaches the center, energy density rises and fusion reactions may occur.

Key ideas:

- Electric current in the capacitor bank creates an azimuthal magnetic field.
- The field pushes plasma inward, increasing temperature and density.
- Proper timing of diagnostics helps capture peak conditions.

## Interactive Sandbox Walk-Through

The new sandbox lets you experiment with a simplified model directly in your
browser or a Jupyter notebook.

1. Launch the sandbox with:
   ```bash
   python -m dpf2.web.sandbox
   ```
2. Select the `student_sandbox.yaml` preset to load nominal device settings.
3. Adjust fill pressure and capacitor voltage using the sliders and watch the
   pinch animation update in real time.
4. Download the run history to compare with classroom calculations.

After exploring the sandbox, continue learning by running the
[tutorials](tutorials/quickstart.md) or reading the [user guide](user_guide.md).

## Classroom Walk-Through

1. Complete the [Quickstart tutorial](tutorials/quickstart.md) to run your
   first simulated shot.
2. Try the [`run_simulation.py`](../examples/run_simulation.py) example to see
   how configuration files drive a full run.
3. Explore the [optimization](tutorials/optimization.md) and
   [diagnostics](tutorials/diagnostics.md) tutorials for deeper engineering
   insight.
