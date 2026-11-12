"""whyTrace.ai compatibility alias module."""
from whytrace import *  # noqa: F401, F403
import whytrace as _orig

__all__ = _orig.__all__ if hasattr(_orig, "__all__") else [k for k in dir(_orig) if not k.startswith("_")]
__version__ = getattr(_orig, "__version__", "0.1.0")

