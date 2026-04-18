"""C++ symbol demangling utility.

Provides a cached, robust wrapper around c++filt for demangling C++ symbols.
Handles missing binary, timeouts, and other subprocess failures gracefully
by returning the mangled name as a fallback.
"""

import logging
import subprocess
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4096)
def demangle(mangled: str) -> str:
    """Demangle a C++ symbol name using c++filt.

    Results are cached via LRU cache to avoid redundant subprocess calls
    for repeated symbols (common in loop-unrolled GPU kernels).

    Args:
        mangled: Mangled C++ name (e.g., _ZN6Kokkos...).

    Returns:
        Demangled name, or original if demangling fails.
    """
    if not mangled.startswith("_Z"):
        return mangled
    try:
        result = subprocess.run(
            ["c++filt", mangled],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.debug("c++filt demangling failed for %s: %s", mangled, e)
    return mangled
