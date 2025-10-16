"""
metrics.py  ·  TBMD utilities
=============================

Complete, self-contained implementation of the four quality metrics that
accompany the TBMD reconstruction experiments:

    • Normalised Frobenius error  (eq. 40 in the paper)
    • Mean-Squared Error (MSE)
    • Structural Similarity Index (SSIM, eq. 41 with C₁ = 0.012, C₂ = 0.032)
    • Peak-Signal-to-Noise-Ratio (PSNR)

The code supports **NumPy** arrays and **PyTorch** tensors, arbitrary spatial
dimensions (2-D, 3-D, …) and optional foreground masks.
"""

import math
import warnings
from typing import Optional, Tuple, Union

import numpy as np
import torch
from skimage.metrics import structural_similarity as _ssim


ArrayLike = Union[np.ndarray, torch.Tensor]

def _to_numpy(a: ArrayLike) -> np.ndarray:
    """Detach/clone to CPU if *a* is a torch tensor, else np.asarray.

    Parameters
    ----------
    a : ArrayLike
        The array-like object to convert to a NumPy array.

    Returns
    -------
    np.ndarray
        The converted NumPy array.
    """
    if torch.is_tensor(a):
        a = a.detach().cpu()
    return np.asarray(a)

def _supports_mask() -> bool:
    """Return True if the installed skimage.ssims supports the *mask* keyword.

    Returns
    -------
    bool
        True if the mask keyword is supported, False otherwise.
    """
    from inspect import signature

    return "mask" in signature(_ssim).parameters

def compute_metrics(
    A_rec: ArrayLike,
    A_ref: ArrayLike,
    *,
    background_value: float | None = None,
    mask: Optional[np.ndarray] = None,
    max_val: float | None = None,
) -> Tuple[float, float, float, float]:
    """
    Parameters
    ----------
    A_rec, A_ref
        Reconstructed and reference volumes (any ndim ≥ 2).  NumPy or PyTorch.
    background_value
        Intensity value that represents background.  Voxels equal to this
        value are excluded **unless** an explicit *mask* is supplied.
    mask
        Boolean array selecting the *foreground*; overrides *background_value*.
    max_val
        Maximum possible pixel / voxel value used for PSNR.  If ``None``,
        defaults to ``A_ref.max() – A_ref.min()`` (data range).

    Returns
    -------
    err_norm  : float
        Normalised Frobenius error (eq. 40, foreground only).
    mse       : float
        Mean-squared error on the foreground.
    ssim_val  : float
        Structural similarity index (mean across channels if any).
    psnr      : float
        PSNR [dB] on the foreground region.
    """
    # -- convert & validate -------------------------------------------------
    A_rec = _to_numpy(A_rec)
    A_ref = _to_numpy(A_ref)

    if A_rec.shape != A_ref.shape:
        raise ValueError("A_rec and A_ref must have identical shapes")

    if mask is None:
        if background_value is None:
            mask = np.ones_like(A_ref, dtype=bool)
        else:
            mask = A_ref != background_value
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != A_ref.shape:
            raise ValueError("mask.shape must match the input volumes")

    if not mask.any():
        raise ValueError("Foreground mask is empty – nothing to evaluate")

    # -- normalised Frobenius error & MSE -----------------------------------
    diff = A_rec.astype(np.float64) - A_ref
    diff_fg = diff[mask]
    ref_fg = A_ref[mask].astype(np.float64)

    mse = float(np.mean(diff_fg**2))
    denom = float(np.sum(ref_fg**2))
    err_norm = math.inf if denom == 0 else float(np.sqrt(np.sum(diff_fg**2)) / np.sqrt(denom))

    # -- SSIM ---------------------------------------------------------------
    C1_paper, C2_paper = 0.012, 0.032          # (K₁L)² and (K₂L)² in eq. 41
    data_range = float(A_ref.max() - A_ref.min())
    if data_range < 1e-12:
        ssim_val = 1.0 if np.allclose(A_rec, A_ref) else 0.0
    else:
        K1 = math.sqrt(C1_paper) / data_range
        K2 = math.sqrt(C2_paper) / data_range

        if A_ref.ndim == 2:                        # single-channel
            ssim_val = _ssim(
                A_ref,
                A_rec,
                data_range=data_range,
                K1=K1,
                K2=K2,
                gaussian_weights=True,
                channel_axis=None,
                mask=mask if _supports_mask() else None,  # falls back gracefully
            )
            if not _supports_mask() and mask is not None:
                warnings.warn("SSIM mask ignored: upgrade scikit-image ≥ 0.20 for masked SSIM")
        else:                                      # channel-last ≥ 3-D
            ssim_vals = []
            for c in range(A_ref.shape[-1]):
                this_mask = mask[..., c] if mask.ndim == A_ref.ndim else mask
                ssim_vals.append(
                    _ssim(
                        A_ref[..., c],
                        A_rec[..., c],
                        data_range=data_range,
                        K1=K1,
                        K2=K2,
                        gaussian_weights=True,
                        channel_axis=None,
                        mask=this_mask if _supports_mask() else None,
                    )
                )
            ssim_val = float(np.mean(ssim_vals))

    # -- PSNR ---------------------------------------------------------------
    if mse == 0:
        psnr = math.inf
    else:
        max_I = float(max_val) if max_val is not None else data_range
        if max_I < 1e-12:
            psnr = 0.0
        else:
            psnr = float(20 * math.log10(max_I / math.sqrt(mse)))

    return err_norm, mse, ssim_val, psnr



# import numpy as np
# import torch
# from skimage.metrics import structural_similarity
# from typing import Union, Tuple, Optional




# def calculate_error_and_ssim(
#     A_re: Union[np.ndarray, torch.Tensor],
#     A: Union[np.ndarray, torch.Tensor],
#     background_value: float | None = None,
#     mask: Optional[np.ndarray] = None,
# ) -> Tuple[float, float, float, float]:
#     """
#     Calculates reconstruction error, MSE, SSIM, and PSNR, optionally limited to the foreground.

#     Parameters
#     ----------
#     A_re, A : np.ndarray | torch.Tensor
#         Reconstructed and reference volumes. Must have the same dimensions.
#     background_value : float, optional
#         Scalar intensity value representing the background (e.g., -1000 for CT, 0 for MRI).
#         Voxels equal to this value are excluded from both metrics when building the mask.
#     mask : bool ndarray, optional
#         Explicit foreground mask (True = foreground). Overrides `background_value`.

#     Returns
#     -------
#     error : float
#         Normalized Frobenius norm *only for foreground voxels*.
#     mse_value : float
#         Mean Squared Error on the foreground voxels.
#     ssim_value : float
#         Structural Similarity Index, averaged across channels (if applicable).
#         SSIM is calculated for the entire image/channel due to limitations in skimage.metrics.structural_similarity
#         for directly applying an arbitrary mask to the SSIM computation process.
#         Constants K1, K2 are chosen so that (K1*L)^2 and (K2*L)^2 correspond to C1=0.012 and C2=0.032 from the paper.
#     psnr_value : float
#         Peak Signal-to-Noise Ratio in decibels (dB), calculated only for foreground voxels.
#     """
#     # Convert torch.Tensor to numpy.ndarray
#     if torch.is_tensor(A_re):
#         A_re = A_re.detach().cpu().numpy()
#     if torch.is_tensor(A):
#         A = A.detach().cpu().numpy()

#     A_re, A = np.asarray(A_re), np.asarray(A)

#     # Check dimensions
#     if A_re.shape != A.shape:
#         raise ValueError("Dimensions of A_re and A must match.")

#     # --- Building / validating mask ---
#     if mask is None:
#         if background_value is None:
#             mask = np.ones_like(A, dtype=bool)   # The entire image is foreground
#         else:
#             mask = A != background_value
#     else:
#         mask = np.asarray(mask, dtype=bool)
#         if mask.shape != A.shape:
#             raise ValueError("Mask size must match input arrays.")

#     if not mask.any():
#         # If the mask is empty but we want to avoid an error, we could return (np.nan, np.nan)
#         # or (np.inf, 0.0) depending on the desired behavior.
#         # The paper doesn't specify this case. To match previous code, we'll raise an error.
#         raise ValueError("Foreground mask is empty - nothing to compare.")

#     # --- Frobenius error on the foreground (according to Equation 40 of the paper) ---
#     # error_i = ||A_re_i - A_i||_F / ||A_i||_F
#     # Applied to elements specified by the mask.
    
#     # Numerator: norm of the difference on the mask
#     diff_on_mask = A_re[mask] - A[mask]
#     numerator_error_sq_sum = np.sum(np.square(diff_on_mask))

#     # Denominator: norm of the original on the mask
#     A_on_mask_sq_sum = np.sum(np.square(A[mask]))

#     if A_on_mask_sq_sum == 0:
#         # If the norm of the original on the mask is zero:
#         # - If the reconstructed image on the mask is also zero (or identical), error is 0.
#         # - Otherwise (original is 0, reconstructed is not 0), error tends to infinity.
#         error = 0.0 if numerator_error_sq_sum == 0 else np.inf
#     else:
#         error = np.sqrt(numerator_error_sq_sum) / np.sqrt(A_on_mask_sq_sum)

#     # --- SSIM ---
#     # Constants C1 and C2 from the paper (p. 12)
#     # C1_paper = 0.01**2
#     # C2_paper = 0.03**2
#     C1_paper = 0.01
#     C2_paper = 0.03

#     # data_range (L) is used to calculate K1, K2, so that (K_i * L)^2 = C_i_paper
#     # In skimage.metrics.structural_similarity, C1=(K1*L)^2, C2=(K2*L)^2.
#     # We calculate data_range across the entire reference image A.
#     # If SSIM were strictly calculated on the mask, data_range could be A[mask].max() - A[mask].min().
#     # However, since skimage.ssim doesn't directly accept a mask for the window process,
#     # we use the global data_range.
#     min_A, max_A = A.min(), A.max()
#     data_range = max_A - min_A

#     # Handle the case when data_range is very small (practically 0)
#     if data_range < 1e-6: # Threshold to prevent floating point issues
#         # If there are no variations in the reference image:
#         # SSIM = 1.0 if images are identical, otherwise can be considered 0.0 (or another value).
#         # np.allclose is used to compare floating point arrays.
#         if np.allclose(A_re, A):
#             ssim_value = 1.0
#         else:
#             # If images are not identical, but data_range=0, SSIM is not very meaningful.
#             # skimage.ssim might return NaN or another value. It's safer to set 0.0.
#             ssim_value = 0.0
#     else:
#         # Calculate K1 and K2 for skimage.ssim to match C1_paper and C2_paper
#         K1 = np.sqrt(C1_paper) / data_range
#         K2 = np.sqrt(C2_paper) / data_range
        
#         # Make sure K1 and K2 are not too large if data_range is very small but not zero.
#         # This can happen if C1_paper, C2_paper are not intended for such scaling.
#         # However, we follow the formula.

#         # Define win_size. Must be odd and <= min(image dimensions).
#         # skimage uses 7 by default. If the image size is smaller, it adapts.
#         # Explicitly set to None so skimage uses its default logic (7 or less).
#         win_size = None # Allows skimage to choose an appropriate win_size (usually 7 or less)

#         if A.ndim == 2:
#             # IMPORTANT: skimage.metrics.structural_similarity does NOT accept a 'mask' argument.
#             # The comment in the original code "skimage >= 0.19 supports a 'mask' keyword" for ssim is incorrect.
#             # Therefore, SSIM is calculated for the entire image (or channel).
#             ssim_value = structural_similarity(
#                 A, A_re, # Order: Reference, Reconstructed
#                 win_size=win_size,
#                 data_range=data_range,
#                 K1=K1,
#                 K2=K2,
#                 channel_axis=None, # For 2D image, explicitly indicate no channel axis
#                 gaussian_weights=True # Standard behavior
#             )
#         elif A.ndim >= 3: # Assume the last axis is channels
#             ssim_values_per_channel = []
#             for c in range(A.shape[-1]):
#                 # Define win_size for each 2D slice of the channel
#                 current_win_size = None # skimage will determine for the slice A[...,c]
#                 if A.shape[0] < 7 or A.shape[1] < 7 : # If slice dimensions are less than 7
#                      current_win_size = min(A.shape[0], A.shape[1])
#                      if current_win_size % 2 == 0: # win_size must be odd
#                          current_win_size = max(1, current_win_size -1)


#                 channel_ssim = structural_similarity(
#                     A[..., c], A_re[..., c], # Order: Reference, Reconstructed
#                     win_size=current_win_size,
#                     data_range=data_range, # data_range is usually global for all channels
#                     K1=K1,
#                     K2=K2,
#                     channel_axis=None, # Process each channel as a 2D image
#                     gaussian_weights=True
#                 )
#                 ssim_values_per_channel.append(channel_ssim)
#             ssim_value = np.mean(ssim_values_per_channel)
#         else: # Unexpected number of dimensions
#             raise ValueError(f"Unsupported number of dimensions for SSIM: {A.ndim}")

#     # Handle NaN in ssim_value, which could occur (although previous logic tries to prevent this)
#     if np.isnan(ssim_value):
#         # This can happen if, for example, K1/K2 became NaN due to data_range=0,
#         # but this case should already be handled.
#         # If still NaN, can set to 0 as a safe value.
#         ssim_value = 0.0

#     # --- PSNR calculation (Peak Signal-to-Noise Ratio) ---
#     # PSNR = 20 * log10(MAX_I / sqrt(MSE))
#     # where MAX_I is the maximum possible pixel value and MSE is Mean Squared Error
#     # We calculate PSNR only for foreground voxels (using the mask)
    
#     # Calculate MSE on the masked region (foreground only)
#     mse_on_mask = float(np.mean(np.square(diff_on_mask)))
    
#     if mse_on_mask == 0:
#         # Perfect reconstruction - PSNR is infinite
#         psnr_value = np.inf
#     else:
#         # Maximum possible pixel value is determined from the data range of the reference image
#         # We use the global data_range calculated earlier for consistency with SSIM
#         max_pixel_value = data_range
        
#         # Handle edge case where data_range is very small
#         if max_pixel_value < 1e-6:
#             # If there's no variation in the reference image, PSNR is not meaningful
#             # Set to a high value if images are identical, 0 otherwise
#             if np.allclose(A_re[mask], A[mask]):
#                 psnr_value = np.inf
#             else:
#                 psnr_value = 0.0
#         else:
#             # Standard PSNR calculation
#             psnr_value = 20 * np.log10(max_pixel_value / np.sqrt(mse_on_mask))

#     # Handle potential NaN or infinite values
#     if np.isnan(psnr_value):
#         psnr_value = 0.0
#     elif np.isinf(psnr_value) and mse_on_mask > 0:
#         # If PSNR is infinite but MSE > 0, something went wrong
#         psnr_value = 0.0

#     return float(error), float(mse_on_mask), float(ssim_value), float(psnr_value)
