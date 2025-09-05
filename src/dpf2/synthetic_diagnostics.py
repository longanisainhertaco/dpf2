"""Legacy synthetic diagnostics module.

This module re-exports the public API from
``dpf2.synthetic_diagnostics.core`` to maintain backwards compatibility
with older imports that expected ``dpf2.synthetic_diagnostics`` to be a
standalone module rather than a package.
"""

from .synthetic_diagnostics.core import *  # noqa: F401,F403
from .synthetic_diagnostics.core import __all__  # noqa: F401

