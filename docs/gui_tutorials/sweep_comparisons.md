# Comparing Sweeps Across Projects

The Qt sweep tool supports multiple projects.  Enter a project name before
running a sweep to group results.  The overlay panel at the bottom of the window
shows yield curves from every project, enabling quick comparison of
"Yield vs Pressure" or other KPIs.

```python
from dpf2.gui.qt_sweep import launch
launch()
```

Sweeps from different projects are colour coded and can be exported to CSV for
further analysis.
