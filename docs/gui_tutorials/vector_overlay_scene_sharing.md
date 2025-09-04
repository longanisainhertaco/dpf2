# Vector Overlays and Scene Sharing

The interactive GUI can display synthetic sheath vector fields and save or load
simulation scenes.

```python
from dpf2.gui import interactive
interactive.launch()
```

Adjust the voltage and pressure sliders to update the J×B quiver overlay.  Use
**Save Scene** to export the current metrics to a JSON file and **Load Scene** to
restore a previously exported file, enabling simple sharing of simulation
setups.
