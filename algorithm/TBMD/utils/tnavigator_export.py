# tnavigator_export.py
# Python 3.10+
"""
Module for exporting pressure fields to tNavigator-compatible CSV format.

The output CSV files can be imported into tNavigator for visualization 
and comparison of original vs reconstructed pressure distributions.
"""
from __future__ import annotations
import os
from typing import Optional, Tuple
import numpy as np

try:
    import torch
except ImportError:
    torch = None  # allows working with numpy only


def _to_numpy_2d(
    x, 
    expect_shape: Optional[Tuple[int, int]] = (139, 48)
) -> np.ndarray:
    """
    Convert input to numpy.float64 2D array and validate shape.
    
    Parameters
    ----------
    x : array-like or torch.Tensor
        Input 2D array (pressure field slice).
    expect_shape : tuple of int, optional
        Expected shape (NY, NX). If None, shape validation is skipped.
        Default is (139, 48) for Brugge model.
        
    Returns
    -------
    np.ndarray
        2D numpy array of float64.
        
    Raises
    ------
    ValueError
        If input is not 2D or doesn't match expected shape.
    """
    if torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}.")
    if expect_shape and tuple(x.shape) != tuple(expect_shape):
        raise ValueError(f"Expected shape {expect_shape}, got {x.shape}.")
    return x


def _maybe_transpose(arr: np.ndarray, ij_order: str) -> np.ndarray:
    """
    Handle axis ordering for tNavigator export.
    
    Parameters
    ----------
    arr : np.ndarray
        Input 2D array.
    ij_order : str
        Axis ordering:
        - 'JI' (default): arr.shape == (J, I) -> J=row 1..NY, I=column 1..NX
        - 'IJ': arr.shape == (I, J), transpose for export
        
    Returns
    -------
    np.ndarray
        Properly oriented array for export.
        
    Raises
    ------
    ValueError
        If ij_order is not 'JI' or 'IJ'.
    """
    ij_order = ij_order.upper()
    if ij_order not in ("JI", "IJ"):
        raise ValueError("ij_order must be 'JI' or 'IJ'.")
    return arr.T if ij_order == "IJ" else arr


def _write_ijk_csv(
    arr: np.ndarray, 
    path: str, 
    value_header: str = "Pressure, psi"
) -> None:
    """
    Write CSV in I,J,K,Value format for tNavigator import.
    
    Parameters
    ----------
    arr : np.ndarray
        2D array with shape (NY, NX), where rows=J, columns=I.
    path : str
        Output file path.
    value_header : str, optional
        Header name for value column (unused in current implementation,
        but kept for potential future use).
    """
    ny, nx = arr.shape  # rows=J, columns=I
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("I,J,K,Value\n")
        for j in range(ny):        # J: 1..NY
            for i in range(nx):    # I: 1..NX
                val = arr[j, i]
                f.write(f"{i+1},{j+1},1,{val:.6f}\n")


def save_pressure_for_tnavigator(
    original_X,
    original_X_reconstructed,
    out_dir: str = "export",
    ij_order: str = "JI",
    expect_shape: Optional[Tuple[int, int]] = (139, 48),
) -> dict:
    """
    Save three CSV files for tNavigator import.
    
    Exports original pressure, reconstructed pressure, and their difference
    to CSV files that can be imported into tNavigator for visualization.
    
    Parameters
    ----------
    original_X : array-like or torch.Tensor
        Original 2D pressure field slice.
    original_X_reconstructed : array-like or torch.Tensor
        Reconstructed 2D pressure field slice.
    out_dir : str, optional
        Output directory path. Default is "export".
    ij_order : str, optional
        Axis ordering of input arrays:
        - 'JI': arrays have shape (NY, NX) = (139, 48) - default
        - 'IJ': arrays have shape (NX, NY), will be transposed
    expect_shape : tuple of int, optional
        Expected shape of input arrays. Default is (139, 48) for Brugge model.
        Set to None to skip shape validation.
        
    Returns
    -------
    dict
        Dictionary with keys 'original', 'reconstructed', 'difference'
        mapping to the respective output file paths.
        
    Examples
    --------
    >>> # Basic usage with 2D slice from 3D tensor
    >>> files = save_pressure_for_tnavigator(
    ...     original_X[:, :, 0], 
    ...     original_X_reconstructed[:, :, 0],
    ...     out_dir="export_4d_wells"
    ... )
    >>> print(files['original'])
    export_4d_wells/pressure_original_psi.csv
    
    >>> # Custom shape for different grid
    >>> files = save_pressure_for_tnavigator(
    ...     original, reconstructed,
    ...     expect_shape=(100, 50),
    ...     out_dir="custom_export"
    ... )
    
    Notes
    -----
    The difference is computed as (original - reconstructed):
    - Positive values indicate reconstruction underestimates pressure
    - Negative values indicate reconstruction overestimates pressure
    """
    X = _maybe_transpose(_to_numpy_2d(original_X, expect_shape), ij_order)
    Xrec = _maybe_transpose(_to_numpy_2d(original_X_reconstructed, expect_shape), ij_order)
    diff = X - Xrec  # sign matters: positive = reconstruction underestimates

    paths = {
        "original": os.path.join(out_dir, "pressure_original_psi.csv"),
        "reconstructed": os.path.join(out_dir, "pressure_reconstructed_psi.csv"),
        "difference": os.path.join(out_dir, "pressure_difference_psi.csv"),
    }
    _write_ijk_csv(X, paths["original"])
    _write_ijk_csv(Xrec, paths["reconstructed"])
    _write_ijk_csv(diff, paths["difference"])
    return paths


__all__ = [
    'save_pressure_for_tnavigator',
]
