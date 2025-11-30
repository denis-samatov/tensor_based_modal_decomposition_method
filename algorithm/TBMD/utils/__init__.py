"""
TBMD Utilities Module

Includes standard utilities and geometry-aware extensions.
"""

from .tbmd_utils import (
    to_torch_tensor,
    get_torch_device,
    extract_step_number,
    auto_select_mode,
    reconstruct_tensor,
    build_Y_matrices,
    build_wells_matrix,
    set_seed,
    set_torch_printoptions,
    compute_reconstruction_metrics
)

from .tnavigator_export import save_pressure_for_tnavigator

# Re-export geometry classes for backward compatibility
def MeshGeometry(*args, **kwargs):
    from ..geometry import MeshGeometry as _MeshGeometry
    _deprecated_import('MeshGeometry')
    return _MeshGeometry(*args, **kwargs)

def TorchMeshGeometry(*args, **kwargs):
    from ..geometry import TorchMeshGeometry as _TorchMeshGeometry
    _deprecated_import('TorchMeshGeometry')
    return _TorchMeshGeometry(*args, **kwargs)

def MeshGraphBuilder(*args, **kwargs):
    from ..geometry import MeshGraphBuilder as _MeshGraphBuilder
    _deprecated_import('MeshGraphBuilder')
    return _MeshGraphBuilder(*args, **kwargs)

def GeometricWeightComputer(*args, **kwargs):
    from ..geometry import GeometricWeightComputer as _GeometricWeightComputer
    _deprecated_import('GeometricWeightComputer')
    return _GeometricWeightComputer(*args, **kwargs)

def estimate_characteristic_length(*args, **kwargs):
    from ..geometry import estimate_characteristic_length as _estimate_characteristic_length
    _deprecated_import('estimate_characteristic_length')
    return _estimate_characteristic_length(*args, **kwargs)

# Deprecated: visualization functions moved to visualization module
import warnings

def _deprecated_import(name):
    """Helper to show deprecation warning"""
    warnings.warn(
        f"Importing '{name}' from algorithm.TBMD.utils is deprecated. "
        f"Use 'from algorithm.TBMD.visualization import {name}' instead.",
        DeprecationWarning,
        stacklevel=3
    )

# Re-export visualization functions for backward compatibility
try:
    from ..visualization.plots import (
        visualize_tensor as _visualize_tensor,
        plot_two_matrices as _plot_two_matrices,
        normalize_for_rgb_display as _normalize_for_rgb_display,
        plot_original_reconstructed_diff as _plot_original_reconstructed_diff,
        visualize_wells_placement as _visualize_wells_placement
    )
    
    def visualize_tensor(*args, **kwargs):
        _deprecated_import('visualize_tensor')
        return _visualize_tensor(*args, **kwargs)
    
    def plot_two_matrices(*args, **kwargs):
        _deprecated_import('plot_two_matrices')
        return _plot_two_matrices(*args, **kwargs)
    
    def normalize_for_rgb_display(*args, **kwargs):
        _deprecated_import('normalize_for_rgb_display')
        return _normalize_for_rgb_display(*args, **kwargs)
    
    def plot_original_reconstructed_diff(*args, **kwargs):
        _deprecated_import('plot_original_reconstructed_diff')
        return _plot_original_reconstructed_diff(*args, **kwargs)
    
    def visualize_wells_placement(*args, **kwargs):
        _deprecated_import('visualize_wells_placement')
        return _visualize_wells_placement(*args, **kwargs)
    
except ImportError:
    # If visualization module not available, skip
    pass

__all__ = [
    # Standard utils
    'to_torch_tensor',
    'get_torch_device',
    'extract_step_number',
    'auto_select_mode',
    'reconstruct_tensor',
    'build_Y_matrices',
    'build_wells_matrix',
    'set_seed',
    'set_torch_printoptions',
    'compute_reconstruction_metrics',
    # tNavigator export
    'save_pressure_for_tnavigator',
    # Geometry utilities
    'MeshGeometry',
    'TorchMeshGeometry',
    'MeshGraphBuilder',
    'GeometricWeightComputer',
    'estimate_characteristic_length',
    # Deprecated visualization (for backward compatibility)
    'visualize_tensor',
    'plot_two_matrices',
    'normalize_for_rgb_display',
    'plot_original_reconstructed_diff',
    'visualize_wells_placement',
    # Deprecated data logic (for backward compatibility)
    'DataLoader',
    'process_data',
    'split_data_in_memory',
    'split_data_in_memory_ordered'
]

# Re-export data logic for backward compatibility
try:
    from ..data_utils import (
        DataLoader as _DataLoader,
        process_data as _process_data,
        split_data_in_memory as _split_data_in_memory,
        split_data_in_memory_ordered as _split_data_in_memory_ordered
    )

    def DataLoader(*args, **kwargs):
        # DataLoader is a class, so we return the class, but we can't wrap it easily in a function 
        # if it's used as a class. 
        # However, the original utils.DataLoader was a class with static methods.
        # If we just assign it, it works.
        _deprecated_import('DataLoader')
        return _DataLoader(*args, **kwargs)
    
    # Better approach for class: just assign it and maybe warn on access if possible, 
    # but for now let's just assign it directly to avoid issues with class instantiation vs static methods.
    # The warning in _deprecated_import is for functions.
    
    # Let's just re-assign for now without wrapper to ensure 100% compat for static methods
    DataLoader = _DataLoader
    
    def process_data(*args, **kwargs):
        _deprecated_import('process_data')
        return _process_data(*args, **kwargs)

    def split_data_in_memory(*args, **kwargs):
        _deprecated_import('split_data_in_memory')
        return _split_data_in_memory(*args, **kwargs)

    def split_data_in_memory_ordered(*args, **kwargs):
        _deprecated_import('split_data_in_memory_ordered')
        return _split_data_in_memory_ordered(*args, **kwargs)

except ImportError as e:
    print(f"Error importing data modules in utils/__init__.py: {e}")
    pass


