"""Visualization modules for fields, sensors, and metrics."""

from .tensor import visualize_tensor
from .generic import plot_two_matrices, plot_original_reconstructed_diff
from .utils import normalize_for_rgb_display
from .wells import visualize_wells_placement

__all__ = [
    'visualize_tensor',
    'plot_two_matrices',
    'normalize_for_rgb_display',
    'plot_original_reconstructed_diff',
    'visualize_wells_placement'
]
