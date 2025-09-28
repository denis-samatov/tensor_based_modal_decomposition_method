import numpy as np
import torch
from skimage.color import rgb2gray
from typing import Optional, Union, Tuple, Dict
from skimage.transform import resize as sk_resize
from tqdm import tqdm

ArrayLike = Union[np.ndarray, torch.Tensor]

def safe_copy(x: ArrayLike) -> ArrayLike:
    """Return a *detached* copy of an array or tensor.

    Parameters
    ----------
    x : ndarray | torch.Tensor
        The object to be copied. If *x* is a ``torch.Tensor`` the function
        returns ``x.clone()``; otherwise ``x.copy()`` is used.

    Returns
    -------
    ndarray | torch.Tensor
        An independent copy/clone that can be modified in-place without
        affecting the original *x*.
    """
    return x.clone() if torch.is_tensor(x) else x.copy()


def foreground_stats(
    arr: np.ndarray,
    mask: Optional[np.ndarray] = None,
    background_value: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    """Compute basic statistics of foreground voxels/pixels.

    The *foreground* comprises all elements either selected by *mask* **or**
    whose value differs from *background_value*.

    Parameters
    ----------
    arr : ndarray
        Input array.
    mask : ndarray[bool], optional
        Boolean mask where *True* denotes foreground. If *None*, the mask is
        built internally from *background_value*.
    background_value : float, optional
        If given, all elements equal to this value are considered background
        and excluded from the statistics.

    Returns
    -------
    tuple of float
        ``(min, max, mean, std)`` of the foreground values. If the foreground
        is empty, the function returns ``(0.0, 0.0, 0.0, 1.0)`` so callers can
        still safely unpack the tuple.
    """
    if mask is None:
        mask = (
            np.ones_like(arr, dtype=bool)
            if background_value is None
            else (arr != background_value)
        )
    if not mask.any():
        return 0.0, 0.0, 0.0, 1.0
    vals = arr[mask]
    return float(vals.min()), float(vals.max()), float(vals.mean()), float(vals.std())

def inverse_normalization(
    normalized_tensor: ArrayLike,
    normalization_method: str = "minmax",
    global_params: Optional[Dict[str, float]] = None,
    background_value: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
    convert_to_grayscale: bool = False,
) -> ArrayLike:
    """Invert a previously applied normalisation.

    Parameters
    ----------
    normalized_tensor : ndarray | torch.Tensor
        The tensor to be denormalised (``[..., T]`` layout is accepted).
    normalization_method : {'minmax', 'zscore'}, default 'minmax'
        Type of normalisation that had been applied.
    global_params : dict[str, float]
        Parameters required to perform the inversion. For *minmax* these are
        ``{'min': float, 'max': float}``; for *zscore* they are ``{'mean': float,
        'std': float}``.
    background_value : float, optional
        Value that marks background voxels/pixels which should **not** be
        modified (useful for sparse medical volumes).
    mask : ndarray[bool], optional
        If given, only positions where ``mask is True`` are denormalised. Takes
        precedence over *background_value*.
    convert_to_grayscale : bool, default False
        If *True*, the function assumes that the input tensor had been
        converted from RGB → grayscale earlier (e.g. via
        ``process_tensor(convert_to_grayscale=True)``) and therefore replicates
        the single grayscale channel back into **three** identical channels
        (RGB). Concretely, a tensor of shape ``(H, W, T)`` becomes
        ``(H, W, 3, T)``. Other shapes are left unchanged.

    Returns
    -------
    ndarray | torch.Tensor
        The tensor in its original value range / distribution.

    Raises
    ------
    ValueError
        If *global_params* is missing or incomplete, or if *normalization_method*
        is unknown.
    """

    if global_params is None:
        raise ValueError("global_params must be provided")

    # Build mask if not supplied
    if mask is None:
        mask = (
            np.ones_like(normalized_tensor, dtype=bool)
            if background_value is None
            else (normalized_tensor != background_value)
        )

    restored: ArrayLike = safe_copy(normalized_tensor)

    if normalization_method == "minmax":
        if not {"min", "max"} <= global_params.keys():
            raise ValueError("global_params needs 'min' and 'max' for minmax inversion.")
        gmin, gmax = global_params["min"], global_params["max"]
        restored[mask] = restored[mask] * (gmax - gmin) + gmin

    elif normalization_method == "zscore":
        if not {"mean", "std"} <= global_params.keys():
            raise ValueError("global_params needs 'mean' and 'std' for zscore inversion.")
        gmean, gstd = global_params["mean"], global_params["std"]
        restored[mask] = restored[mask] * gstd + gmean

    else:
        raise ValueError("Unknown normalization method. Use 'minmax' or 'zscore'.")

    # ------------------------------------------------------------------
    # 4. Optionally replicate grayscale → RGB (3 identical channels)
    # ------------------------------------------------------------------
    if convert_to_grayscale:
        if restored.ndim == 3:  # (H, W, T)
            # Replicate channel dimension → (H, W, 3, T)
            if isinstance(restored, np.ndarray):
                restored = np.repeat(restored[:, :, np.newaxis, :], 3, axis=2)
            else:  # assume torch.Tensor-like
                restored = restored.unsqueeze(2).repeat(1, 1, 3, 1)
        elif restored.ndim == 2:  # (H, W) edge-case
            if isinstance(restored, np.ndarray):
                restored = np.repeat(restored[:, :, np.newaxis], 3, axis=2)
            else:
                restored = restored.unsqueeze(2).repeat(1, 1, 3)

    return restored

def calculate_global_minmax_params(
    data: Dict[str, np.ndarray] | np.ndarray,
    masks: Optional[Dict[str, np.ndarray] | np.ndarray] = None,
    background_value: Optional[float] = None,
) -> Tuple[float, float]:
    """Determine the global *min* and *max* across an entire dataset.

    Parameters
    ----------
    data : dict[str, ndarray] | ndarray
        Either a mapping ``{subject_id: volume}`` or a single volume.
    masks : dict[str, ndarray] | ndarray, optional
        Matching foreground masks. Ignored if *background_value* is provided.
    background_value : float, optional
        If specified, any voxel equal to this value is considered background and
        excluded from the global extrema calculation.

    Returns
    -------
    (float, float)
        ``(global_min, global_max)`` suitable for *min-max* normalisation.
    """
    global_min, global_max = np.inf, -np.inf

    if isinstance(data, dict):
        for sid, tensor in data.items():
            m = None if masks is None else masks[sid]
            m_min, m_max, _, _ = foreground_stats(
                tensor, mask=m, background_value=background_value
            )
            global_min, global_max = min(global_min, m_min), max(global_max, m_max)
    else:
        global_min, global_max, _, _ = foreground_stats(
            data, mask=masks, background_value=background_value
        )

    return global_min, global_max


def calculate_global_zscore_params(
    data: Dict[str, np.ndarray] | np.ndarray,
    masks: Optional[Dict[str, np.ndarray] | np.ndarray] = None,
    background_value: Optional[float] = None,
) -> Tuple[float, float]:
    """Compute the global *mean* and *std* for z-score normalisation.

    The function concatenates all foreground voxels into a 1-D vector and
    calculates statistics on the aggregate.

    Parameters
    ----------
    data : dict[str, ndarray] | ndarray
        Dataset in the same conventions as :pyfunc:`calculate_global_minmax_params`.
    masks : dict[str, ndarray] | ndarray, optional
        Foreground masks corresponding to *data*.
    background_value : float, optional
        Background marker value.

    Returns
    -------
    (float, float)
        ``(global_mean, global_std)`` ready for *z-score* normalisation.
    """

    all_vals: list[np.ndarray] = []

    def _append_vals(arr: np.ndarray, m: Optional[np.ndarray]):
        v = arr if m is None else arr[m]
        all_vals.append(v.flatten())

    if isinstance(data, dict):
        for sid, tensor in data.items():
            m = None if masks is None else masks[sid]
            _append_vals(tensor, m)
    else:
        _append_vals(data, masks)  # type: ignore[arg-type]

    concat = np.concatenate(all_vals, axis=0)
    return float(concat.mean()), float(concat.std())

def process_tensor(
    tensor: np.ndarray,
    resize_shape: Optional[Tuple[int, int]] = None,
    convert_to_grayscale: bool = False,
    normalization_method: Optional[str] = None,
    global_params: Optional[Dict[str, float]] = None,
    background_value: Optional[float] = None,
    verbose: bool = False,
) -> np.ndarray:
    """Pre-process a single multi-slice tensor.

    Steps performed **slice-wise** (i.e. independently for each time-frame):

    1. RGB → grayscale conversion (if *convert_to_grayscale* and *tensor* is 4-D).
    2. Spatial resizing via ``skimage.transform.resize``.
    3. Normalisation (*min-max*, *z-score* or **no** scaling).

    Parameters
    ----------
    tensor : ndarray
        3-D ``(H, W, T)`` or 4-D ``(H, W, 3, T)`` array.
    resize_shape : (int, int), optional
        Target spatial size ``(new_H, new_W)``. If *None*, original resolution
        is preserved.
    convert_to_grayscale : bool, default False
        Whether to collapse RGB channels using ``skimage.color.rgb2gray``.
    normalization_method : {'minmax', 'zscore', None}, default None
        * ``'minmax'`` – scales values to [0, 1].
        * ``'zscore'`` – zero-mean, unit-variance.
        * ``None``      – skip normalisation.
    global_params : dict[str, float], optional
        Statistics pre-computed across the dataset (see functions above). If
        *None*, slice-local statistics are used.
    background_value : float, optional
        Background label to be excluded from normalisation.
    verbose : bool, default False
        Whether to display visual information about processing parameters.

    Returns
    -------
    ndarray
        Processed tensor with the same *T* but potentially different ``H × W``.

    Raises
    ------
    ValueError
        If an unsupported *normalization_method* is specified.
    """
    # Display processing parameters if verbose
    if verbose:
        original_shape = tensor.shape
        print("\n" + "="*60)
        print("TENSOR PROCESSING CONFIGURATION")
        print("="*60)
        print(f"▸ Input tensor shape:    {original_shape}")
        print(f"▸ Resize shape:          {resize_shape if resize_shape else 'No resizing'}")
        print(f"▸ Convert to grayscale:  {convert_to_grayscale}")
        print(f"▸ Normalization method:  {normalization_method if normalization_method else 'No normalization'}")
        if normalization_method and global_params:
            if normalization_method == "minmax" and {"min", "max"} <= global_params.keys():
                print(f"  ↳ Global min/max:      {global_params['min']:.4f} / {global_params['max']:.4f}")
            elif normalization_method == "zscore" and {"mean", "std"} <= global_params.keys():
                print(f"  ↳ Global mean/std:     {global_params['mean']:.4f} / {global_params['std']:.4f}")
            else:
                print(f"  ↳ Using local statistics (per slice)")
        print(f"▸ Background value:      {background_value if background_value is not None else 'None'}")
        print("="*60 + "\n")
    
    processed: list[np.ndarray] = []
    T = tensor.shape[-1]

    def _apply_minmax(img: np.ndarray, gmin: float, gmax: float) -> np.ndarray:
        if background_value is None:
            return (img - gmin) / (gmax - gmin)
        mask = img != background_value
        if mask.any():
            img = safe_copy(img)
            img[mask] = (img[mask] - gmin) / (gmax - gmin)
        return img

    def _apply_zscore(img: np.ndarray, gmean: float, gstd: float) -> np.ndarray:
        if gstd == 0:
            return img  # avoid division by zero
        if background_value is None:
            return (img - gmean) / gstd
        mask = img != background_value
        if mask.any():
            img = safe_copy(img)
            img[mask] = (img[mask] - gmean) / gstd
        return img

    for t in range(T):
        img: np.ndarray = tensor[..., t]

        # 1. Colour → grayscale
        if tensor.ndim == 4 and convert_to_grayscale:
            img = rgb2gray(img)

        # 2. Resize
        if resize_shape is not None:
            img = sk_resize(img, resize_shape, anti_aliasing=True)

        # 3. Normalisation
        if normalization_method == "minmax":
            if global_params and {"min", "max"} <= global_params.keys():
                gmin, gmax = global_params["min"], global_params["max"]
            else:
                gmin, gmax, _, _ = foreground_stats(img, background_value=background_value)
            if gmax > gmin:
                img = _apply_minmax(img, gmin, gmax)

        elif normalization_method == "zscore":
            if global_params and {"mean", "std"} <= global_params.keys():
                gmean, gstd = global_params["mean"], global_params["std"]
            else:
                _, _, gmean, gstd = foreground_stats(img, background_value=background_value)
            img = _apply_zscore(img, gmean, gstd)

        elif normalization_method is not None:
            raise ValueError("Unknown normalization method. Use 'minmax', 'zscore', or None.")

        processed.append(img)

    return np.stack(processed, axis=-1)

def process_data(
    data: Dict[str, np.ndarray],
    resize_shape: Optional[Tuple[int, int]] = None,
    convert_to_grayscale: bool = False,
    normalization_method: Optional[str] = None,
    global_params: Optional[Dict[str, float]] = None,
    background_value: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Apply :pyfunc:`process_tensor` to every entry in a dataset.

    A **tqdm** progress bar is displayed for convenience, and shape information
    for each subject is printed to stdout.

    Parameters
    ----------
    data : dict[str, ndarray]
        Mapping ``{subject_id: volume}``.
    resize_shape : (int, int), optional
        Target spatial size ``(new_H, new_W)``. If *None*, original resolution
        is preserved.
    convert_to_grayscale : bool, default False
        Whether to collapse RGB channels using ``skimage.color.rgb2gray``.
    normalization_method : {'minmax', 'zscore', None}, default None
        * ``'minmax'`` – scales values to [0, 1].
        * ``'zscore'`` – zero-mean, unit-variance.
        * ``None``      – skip normalisation.
    global_params : dict[str, float], optional
        Statistics pre-computed across the dataset.
    background_value : float, optional
        Background label to be excluded from normalisation.
    verbose : bool, default True
        Whether to display visual information about processing parameters.

    Returns
    -------
    dict[str, ndarray]
        The processed dataset keyed by the same *subject_id*s.
    """
    processed: Dict[str, np.ndarray] = {}
    
    # Display processing parameters if verbose
    if verbose:
        print("\n" + "="*60)
        print("DATA PROCESSING CONFIGURATION")
        print("="*60)
        print(f"▸ Resize shape:          {resize_shape if resize_shape else 'No resizing'}")
        print(f"▸ Convert to grayscale:  {convert_to_grayscale}")
        print(f"▸ Normalization method:  {normalization_method if normalization_method else 'No normalization'}")
        if normalization_method and global_params:
            if normalization_method == "minmax" and {"min", "max"} <= global_params.keys():
                print(f"  ↳ Global min/max:      {global_params['min']:.4f} / {global_params['max']:.4f}")
            elif normalization_method == "zscore" and {"mean", "std"} <= global_params.keys():
                print(f"  ↳ Global mean/std:     {global_params['mean']:.4f} / {global_params['std']:.4f}")
            else:
                print(f"  ↳ Using local statistics (per slice)")
        print(f"▸ Background value:      {background_value if background_value is not None else 'None'}")
        print("="*60 + "\n")
    
    for sid, tensor in tqdm(data.items(), desc="Processing subjects"):
        try:
            # Pass verbose=False to avoid repeated parameter display for each subject
            processed[sid] = process_tensor(
                tensor,
                resize_shape,
                convert_to_grayscale,
                normalization_method,
                global_params,
                background_value,
                verbose=False,  # Avoid repeated display for each subject
            )
            print(f"{sid}: {processed[sid].shape}")
        except Exception as exc:
            print(f"Error on {sid}: {exc}")
    return processed
