# Scalability Studies

Synthetic scripts in `scripts/scaling_tests.py` provide helper functions to
estimate ideal weak and strong scaling.  Generated data can be written to a
markdown report using `document_results`.

Example:

```python
from scripts.scaling_tests import weak_scaling, strong_scaling, document_results
weak = weak_scaling([1, 2, 4])
strong = strong_scaling([1, 2, 4])
document_results("scaling_report.md", weak, strong)
```
