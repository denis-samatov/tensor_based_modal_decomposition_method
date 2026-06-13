"""
TBMD Package - Tensor-Based Modal Decomposition Method

Main package for tensor-based modal decomposition algorithms with geometry awareness.
"""

# Re-export geometry modules for backward compatibility
# This allows "from TBMD.geometry import ..." to work
from .core import geometry

__all__ = ["geometry"]

__version__ = "2.0.0"
