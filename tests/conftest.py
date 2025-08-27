import sys
from pathlib import Path

# Provide a minimal ``numpy`` implementation for environments where the
# dependency is unavailable.  If the real ``numpy`` package is installed the
# stub is skipped and the genuine implementation is used.
try:  # pragma: no cover - prefer real numpy
    import numpy  # noqa: F401
except Exception:  # pragma: no cover - fallback to stub
    import runpy
    runpy.run_path(Path(__file__).with_name("numpy_stub.py"))

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
