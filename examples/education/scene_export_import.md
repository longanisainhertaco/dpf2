# Scene Export and Import

`ProjectManager` can persist its state to a JSON file and restore it later.

```python
from dpf2.gui import ProjectManager
pm = ProjectManager(project="demo")
pm.metrics["run"] = {0.5: {"yield": 1.0}}
pm.params["run"] = "initial_pressure"
pm.export_scene("demo_scene.json")

pm2 = ProjectManager()
pm2.import_scene("demo_scene.json")
print(pm2.metrics)
```
