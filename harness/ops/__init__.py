"""Ops package public surface.

Single-export layer over ``legacy_ops`` to keep imports stable while reducing
indirection depth.
"""

from .legacy_ops import *  # noqa: F401,F403
