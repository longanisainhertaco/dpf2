# Editing Geometries in the GUI

The interactive and Qt interfaces can now import STEP, STL or VTK geometry files.
Use the *Upload Geometry* control to load a mesh and visualise the resulting
anode/cathode shapes.  Translation controls apply simple offsets for quick
position tweaks.

```python
from dpf2.gui import interactive
interactive.launch()
```

After uploading a geometry file you may translate it and view the updated mesh
in real time.  Electrode outlines are rendered using Plotly allowing basic
inspection without external tools.
