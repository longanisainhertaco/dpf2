import sys
from pathlib import Path

# Provide a minimal ``numpy`` implementation for environments where the
# dependency is unavailable.  The stub registers itself under ``sys.modules``
# as ``numpy``.
import runpy

# Execute the ``numpy`` stub to provide a minimal implementation when the
# real dependency is unavailable.  The stub registers itself in ``sys.modules``.
runpy.run_path(Path(__file__).with_name("numpy_stub.py"))

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
