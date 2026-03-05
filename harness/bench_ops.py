"""Backward-compatible shim for bench operations.

The implementation has been moved under ``harness/ops`` for modularization.
Importing ``bench_ops`` remains supported for existing command entrypoints.
"""

from ops import *  # noqa: F401,F403
