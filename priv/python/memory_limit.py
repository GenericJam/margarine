"""
Memory limiting utilities for Python processes.

Prevents Python from consuming all available system memory by setting hard limits.
This is especially important when running multiple tests or model instances.
"""

import resource
import sys


def set_memory_limit_gb(limit_gb=16):
    """
    Set a hard memory limit for the Python process.

    Args:
        limit_gb: Maximum memory in gigabytes (default: 16GB)

    Note:
        - On macOS: DISABLED - RLIMIT_AS doesn't work with embedded Python (Pythonx)
        - On Linux: Uses RLIMIT_DATA (data segment) for heap memory limiting
        - Limit applies to heap allocations, not total address space

    Example:
        >>> set_memory_limit_gb(16)  # Cap at 16GB heap on Linux
        >>> # On macOS, this is a no-op (relies on Elixir-side memory checks)

    Why not RLIMIT_AS?
        - RLIMIT_AS limits virtual address space (includes mmap, libraries, stack)
        - Pythonx runs Python in-process via NIFs (shares BEAM's address space)
        - BEAM's address space is already > 16GB typically
        - Setting RLIMIT_AS fails with "current limit exceeds maximum"
        - Use RLIMIT_DATA instead to limit heap allocations only
    """
    import platform

    # Skip memory limiting on macOS - Pythonx runs in-process via NIFs
    # and RLIMIT_AS doesn't work with embedded Python
    if platform.system() == "Darwin":
        print(
            "[Memory Limit] Skipping on macOS (Pythonx embedded Python). "
            "Relying on Elixir-side memory checks.",
            file=sys.stderr
        )
        return

    try:
        # Convert GB to bytes
        limit_bytes = limit_gb * 1024 * 1024 * 1024

        # On Linux, use RLIMIT_DATA (heap memory) instead of RLIMIT_AS (address space)
        # This limits heap allocations without affecting mmap'd model weights
        if hasattr(resource, 'RLIMIT_DATA'):
            resource.setrlimit(resource.RLIMIT_DATA, (limit_bytes, limit_bytes))
            print(
                f"[Memory Limit] Set RLIMIT_DATA to {limit_gb}GB ({limit_bytes:,} bytes)",
                file=sys.stderr
            )
        else:
            print(
                "[Memory Limit] RLIMIT_DATA not available on this platform",
                file=sys.stderr
            )

    except (ValueError, OSError) as e:
        # May fail on some systems or if limit is too low
        print(f"[Memory Limit] Warning: Could not set limit: {e}", file=sys.stderr)


def get_current_memory_usage_mb():
    """
    Get current memory usage of this process in MB.

    Returns:
        float: Memory usage in megabytes
    """
    import psutil
    import os

    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def check_memory_available_gb():
    """
    Check available system memory in GB.

    Returns:
        float: Available memory in gigabytes
    """
    import psutil

    return psutil.virtual_memory().available / (1024**3)


# Auto-apply memory limit when this module is imported
# Can be overridden by calling set_memory_limit_gb() again
# Note: On macOS, this is a no-op (see set_memory_limit_gb docstring)
if __name__ != "__main__":
    # Only set limit when imported (not when run directly)
    # On macOS with Pythonx: skipped (embedded Python shares BEAM address space)
    # On Linux: sets RLIMIT_DATA to limit heap allocations
    set_memory_limit_gb(16)
