# Comparing Sweeps Across Projects

The Qt sweep tool supports multiple projects.  Enter a project name before
running a sweep to group results.  The new sweep panel at the bottom of the
window overlays curves from every project, enabling quick comparison of "Yield
vs Pressure" or any stored KPI.

```python
from dpf2.gui.qt_sweep import launch
launch()
```

Sweeps from different projects are colour coded and can be exported to CSV for
further analysis.  The panel accepts an arbitrary metric name making it easy to
compare efficiency or pinch time as well as yield.
