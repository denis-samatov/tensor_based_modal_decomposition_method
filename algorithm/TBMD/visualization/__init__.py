"""
Visualization Module

Модули для визуализации полей, сенсоров, метрик
"""

from .plots import (
    visualize_tensor,
    plot_two_matrices,
    normalize_for_rgb_display,
    plot_original_reconstructed_diff,
    visualize_wells_placement
)

__all__ = [
    'visualize_tensor',
    'plot_two_matrices',
    'normalize_for_rgb_display',
    'plot_original_reconstructed_diff',
    'visualize_wells_placement'
]
