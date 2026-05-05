import numpy as np
import torch
from skimage.color import rgb2gray
from skimage.transform import resize as sk_resize
from typing import Optional, Union, Tuple, Dict, List
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, torch.Tensor]

class DataProcessor:
    """
    Базовый класс для обработки данных
    
    Examples:
        >>> processor = DataProcessor()
        >>> processed = processor.process(tensor)
    """
    
    def __init__(self, device: str = 'cpu', dtype: torch.dtype = torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype
    
    def process(self, data: torch.Tensor) -> torch.Tensor:
        """
        Обработать данные (должен быть переопределен в подклассах)
        
        Args:
            data: Входные данные
            
        Returns:
            Обработанные данные
        """
        return data


class Normalizer(DataProcessor):
    """
    Нормализация данных
    
    Поддерживает разные типы нормализации:
    - minmax: к диапазону [0, 1]
    - zscore: стандартизация (mean=0, std=1)
    - maxabs: к диапазону [-1, 1] по максимуму абсолютного значения
    
    Examples:
        >>> normalizer = Normalizer(method='minmax')
        >>> normalized = normalizer.normalize(tensor)
        >>> original = normalizer.denormalize(normalized)
    """
    
    def __init__(
        self,
        method: str = 'minmax',
        feature_range: Tuple[float, float] = (0, 1),
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32
    ):
        """
        Args:
            method: Метод нормализации ('minmax', 'zscore', 'maxabs')
            feature_range: Целевой диапазон для minmax
            device: Torch device
            dtype: Torch dtype
        """
        super().__init__(device, dtype)
        self.method = method
        self.feature_range = feature_range
        
        # Параметры для денормализации
        self.params = {}
    
    def normalize(
        self,
        data: torch.Tensor,
        dim: Optional[int] = None
    ) -> torch.Tensor:
        """
        Нормализовать данные
        
        Args:
            data: Входные данные
            dim: Размерность для вычисления статистик (None = все)
            
        Returns:
            Нормализованные данные
        """
        data = data.to(device=self.device, dtype=self.dtype)
        
        if self.method == 'minmax':
            return self._normalize_minmax(data, dim)
        elif self.method == 'zscore':
            return self._normalize_zscore(data, dim)
        elif self.method == 'maxabs':
            return self._normalize_maxabs(data, dim)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _normalize_minmax(
        self,
        data: torch.Tensor,
        dim: Optional[int] = None
    ) -> torch.Tensor:
        """Min-max нормализация"""
        if dim is None:
            min_val = data.min()
            max_val = data.max()
        else:
            min_val = data.min(dim=dim, keepdim=True)[0]
            max_val = data.max(dim=dim, keepdim=True)[0]
        
        # Сохранить параметры
        self.params['min'] = min_val
        self.params['max'] = max_val
        
        # Нормализовать
        data_norm = (data - min_val) / (max_val - min_val + 1e-8)
        
        # Масштабировать к feature_range
        feat_min, feat_max = self.feature_range
        data_norm = data_norm * (feat_max - feat_min) + feat_min
        
        return data_norm
    
    def _normalize_zscore(
        self,
        data: torch.Tensor,
        dim: Optional[int] = None
    ) -> torch.Tensor:
        """Z-score стандартизация"""
        if dim is None:
            mean = data.mean()
            std = data.std()
        else:
            mean = data.mean(dim=dim, keepdim=True)
            std = data.std(dim=dim, keepdim=True)
        
        # Сохранить параметры
        self.params['mean'] = mean
        self.params['std'] = std
        
        # Стандартизовать
        data_norm = (data - mean) / (std + 1e-8)
        
        return data_norm
    
    def _normalize_maxabs(
        self,
        data: torch.Tensor,
        dim: Optional[int] = None
    ) -> torch.Tensor:
        """Max-abs нормализация"""
        if dim is None:
            max_abs = torch.abs(data).max()
        else:
            max_abs = torch.abs(data).max(dim=dim, keepdim=True)[0]
        
        # Сохранить параметры
        self.params['max_abs'] = max_abs
        
        # Нормализовать
        data_norm = data / (max_abs + 1e-8)
        
        return data_norm
    
    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Денормализовать данные
        
        Args:
            data: Нормализованные данные
            
        Returns:
            Оригинальные данные
        """
        if not self.params:
            logger.warning("Параметры нормализации не найдены, возвращаю данные как есть")
            return data
        
        if self.method == 'minmax':
            feat_min, feat_max = self.feature_range
            data = (data - feat_min) / (feat_max - feat_min)
            data = data * (self.params['max'] - self.params['min']) + self.params['min']
        
        elif self.method == 'zscore':
            data = data * self.params['std'] + self.params['mean']
        
        elif self.method == 'maxabs':
            data = data * self.params['max_abs']
        
        return data
    
    def fit(self, data: torch.Tensor, dim: Optional[int] = None):
        """
        Вычислить параметры нормализации (без применения)
        
        Args:
            data: Входные данные
            dim: Размерность для статистик
        """
        data = data.to(device=self.device, dtype=self.dtype)
        
        if self.method == 'minmax':
            if dim is None:
                self.params['min'] = data.min()
                self.params['max'] = data.max()
            else:
                self.params['min'] = data.min(dim=dim, keepdim=True)[0]
                self.params['max'] = data.max(dim=dim, keepdim=True)[0]
        
        elif self.method == 'zscore':
            if dim is None:
                self.params['mean'] = data.mean()
                self.params['std'] = data.std()
            else:
                self.params['mean'] = data.mean(dim=dim, keepdim=True)
                self.params['std'] = data.std(dim=dim, keepdim=True)
        
        elif self.method == 'maxabs':
            if dim is None:
                self.params['max_abs'] = torch.abs(data).max()
            else:
                self.params['max_abs'] = torch.abs(data).max(dim=dim, keepdim=True)[0]
    
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Применить ранее вычисленные параметры
        
        Args:
            data: Входные данные
            
        Returns:
            Нормализованные данные
        """
        if not self.params:
            raise ValueError("Сначала вызовите fit() или normalize()")
        
        data = data.to(device=self.device, dtype=self.dtype)
        
        if self.method == 'minmax':
            data = (data - self.params['min']) / (self.params['max'] - self.params['min'] + 1e-8)
            feat_min, feat_max = self.feature_range
            data = data * (feat_max - feat_min) + feat_min
        
        elif self.method == 'zscore':
            data = (data - self.params['mean']) / (self.params['std'] + 1e-8)
        
        elif self.method == 'maxabs':
            data = data / (self.params['max_abs'] + 1e-8)
        
        return data

# =================================================================================================
# MIGRATED FUNCTIONS FROM utils/process_data.py
# =================================================================================================

def safe_copy(x: ArrayLike) -> ArrayLike:
    """Return a *detached* copy of an array or tensor."""
    return x.clone() if torch.is_tensor(x) else x.copy()


def foreground_stats(
    arr: np.ndarray,
    mask: Optional[np.ndarray] = None,
    background_value: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    """Computes basic statistics of foreground voxels/pixels.

    The foreground is defined as all elements selected by `mask` or those whose
    values differ from `background_value`.

    Args:
        arr (np.ndarray): The input array.
        mask (Optional[np.ndarray]): A boolean mask where `True` denotes the
            foreground. If `None`, the mask is created from `background_value`.
            Defaults to None.
        background_value (Optional[float]): A value indicating background
            elements, which are excluded from statistics. Defaults to None.

    Returns:
        Tuple[float, float, float, float]: A tuple containing the min, max,
        mean, and std of the foreground values. If the foreground is empty,
        returns (0.0, 0.0, 0.0, 1.0).
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
    """Inverts a previously applied normalization.

    Args:
        normalized_tensor (ArrayLike): The tensor to be denormalized.
        normalization_method (str): The type of normalization that was
            applied. Can be 'minmax' or 'zscore'. Defaults to "minmax".
        global_params (Optional[Dict[str, float]]): The parameters required for
            inversion (e.g., {'min': float, 'max': float} for 'minmax').
        background_value (Optional[float]): A value marking background pixels
            that should not be modified.
        mask (Optional[np.ndarray]): A mask where `True` indicates positions
            to denormalize. Overrides `background_value`.
        convert_to_grayscale (bool): If `True`, replicates a single grayscale
            channel into three identical RGB channels.

    Returns:
        ArrayLike: The tensor in its original value range.

    Raises:
        ValueError: If `global_params` is missing or incomplete, or if
            `normalization_method` is unknown.
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

    if convert_to_grayscale:
        if restored.ndim == 3:  # (H, W, T)
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
    """Determines the global min and max across an entire dataset.

    Args:
        data (Dict[str, np.ndarray] | np.ndarray): A dictionary mapping subject
            IDs to volumes, or a single volume.
        masks (Optional[Dict[str, np.ndarray] | np.ndarray]): Matching
            foreground masks. Ignored if `background_value` is provided.
        background_value (Optional[float]): A value indicating background
            voxels, which are excluded from the calculation.

    Returns:
        Tuple[float, float]: A tuple containing the global min and max values.
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
    """Computes the global mean and std for z-score normalization.

    This function concatenates all foreground voxels into a 1D vector and
    calculates statistics on the aggregate.

    Args:
        data (Dict[str, np.ndarray] | np.ndarray): A dataset, either as a
            dictionary of volumes or a single volume.
        masks (Optional[Dict[str, np.ndarray] | np.ndarray]): Foreground masks
            corresponding to the data.
        background_value (Optional[float]): A value indicating background
            voxels.

    Returns:
        Tuple[float, float]: A tuple containing the global mean and standard
        deviation.
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
    """Pre-processes a single multi-slice tensor.

    This function applies a series of processing steps to each time-frame in
    the tensor, including grayscale conversion, resizing, and normalization.

    Args:
        tensor (np.ndarray): A 3D (H, W, T) or 4D (H, W, 3, T) array.
        resize_shape (Optional[Tuple[int, int]]): The target spatial size
            (new_H, new_W). If `None`, no resizing is performed.
        convert_to_grayscale (bool): If `True`, collapses RGB channels to
            grayscale.
        normalization_method (Optional[str]): The normalization method to use
            ('minmax', 'zscore', or `None`).
        global_params (Optional[Dict[str, float]]): Pre-computed statistics
            for normalization. If `None`, statistics are computed per-slice.
        background_value (Optional[float]): A value to be excluded from
            normalization.
        verbose (bool): If `True`, displays information about the processing
            parameters.

    Returns:
        np.ndarray: The processed tensor.

    Raises:
        ValueError: If an unsupported `normalization_method` is specified.
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
    """Applies `process_tensor` to every entry in a dataset.

    This function iterates through a dictionary of tensors, applying the same
    processing steps to each one. A progress bar is displayed, and shape
    information for each subject is printed.

    Args:
        data (Dict[str, np.ndarray]): A dictionary mapping subject IDs to
            volumes.
        resize_shape (Optional[Tuple[int, int]]): The target spatial size
            (new_H, new_W).
        convert_to_grayscale (bool): If `True`, collapses RGB channels to
            grayscale.
        normalization_method (Optional[str]): The normalization method to use.
        global_params (Optional[Dict[str, float]]): Pre-computed statistics
            for normalization.
        background_value (Optional[float]): A value to be excluded from
            normalization.
        verbose (bool): If `True`, displays information about the processing
            parameters.

    Returns:
        Dict[str, np.ndarray]: The processed dataset, with the same subject IDs.
    """
    processed: Dict[str, np.ndarray] = {}
    
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
            processed[sid] = process_tensor(
                tensor,
                resize_shape,
                convert_to_grayscale,
                normalization_method,
                global_params,
                background_value,
                verbose=False,
            )
            print(f"{sid}: {processed[sid].shape}")
        except Exception as exc:
            print(f"Error on {sid}: {exc}")
    return processed
