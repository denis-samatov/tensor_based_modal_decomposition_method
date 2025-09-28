import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
import re
import torch

from pathlib import Path
from typing import Union, Optional, Dict
from collections import defaultdict
from tqdm import tqdm




def extract_step_number(filename: str) -> int:
    """
    Extract the step number from a filename of the form 'PRESSURE_STEP_XXX.png'.
    Returns 0 if no step number is found (as a fallback).
    """
    match = re.search(r"_(\d+)", filename)
    if match:
        return int(match.group(1))
    return 0

def auto_select_mode(tensor: Union[np.ndarray, 'tl.tensor', torch.Tensor],
                     x_hat: Union[np.ndarray, 'tl.tensor', torch.Tensor]) -> int:
    """
    Automatically selects the mode (dimension) of the tensor that matches the size of
    the given vector x_hat. Accepts:
      - NumPy arrays
      - PyTorch tensors
      - TensorLy tensors (any backend recognized by TensorLy).

    Parameters
    ----------
    tensor : np.ndarray or tensorly-recognized tensor or torch.Tensor
        Input tensor of any shape.
    x_hat : np.ndarray or tensorly-recognized tensor or torch.Tensor
        1D vector whose size must match one of the tensor dimensions.

    Returns
    -------
    int
        Index of the mode (dimension) in the tensor that matches the size of x_hat.

    Raises
    ------
    TypeError
        If 'tensor' is not a NumPy array, a PyTorch tensor, or recognized by TensorLy.
    ValueError
        If 'x_hat' is not 1D or if its size does not match any dimension of 'tensor'.

    Examples
    --------
    >>> import numpy as np
    >>> tensor = np.random.rand(4, 5, 6)
    >>> x_hat = np.random.rand(5)
    >>> mode = auto_select_mode(tensor, x_hat)
    >>> print(mode)  # Output: 1 (because x_hat matches tensor dimension 5)
    """

    is_numpy_tensor = isinstance(tensor, np.ndarray)
    is_torch_tensor = torch.is_tensor(tensor)
    is_tl_tensor = tl.backend.is_tensor(tensor)

    if not (is_numpy_tensor or is_torch_tensor or is_tl_tensor):
        raise TypeError("The input 'tensor' must be a NumPy array, a PyTorch tensor, or a TensorLy-recognized tensor.")

    x_hat_is_numpy = isinstance(x_hat, np.ndarray)
    x_hat_is_torch = torch.is_tensor(x_hat)
    x_hat_is_tl = tl.backend.is_tensor(x_hat)

    if not (x_hat_is_numpy or x_hat_is_torch or x_hat_is_tl):
        raise TypeError("x_hat must be a NumPy array, a PyTorch tensor, or a TensorLy-recognized tensor.")

    if x_hat_is_numpy:
        if x_hat.ndim != 1:
            raise ValueError("x_hat must be a 1D array (vector).")
        x_hat_size = x_hat.size
    else:
        if len(x_hat.shape) != 1:
            raise ValueError("x_hat must be a 1D tensor (vector).")
        x_hat_size = x_hat.shape[0]

    tensor_shape = tensor.shape

    for mode, dim in enumerate(tensor_shape):
        if dim == x_hat_size:
            return mode

    raise ValueError(
        f"Cannot apply x_hat of size {x_hat_size} to tensor with shape {tensor_shape}. "
        "Ensure x_hat matches one of the tensor dimensions."
    )

def generate_noisy_datasets(
    data,
    noise_level: float = 0.1,
    num_noisy_datasets: int = 5,
    output_dir: str = None,
    experiment_id: str = None
) -> dict:
    """
    Generates multiple datasets by adding Gaussian noise to the input data.
    Supports torch.Tensor, numpy.ndarray, or a dictionary (defaultdict) of them.
    
    The returned dict has keys 'noisy_dataset_{index}', where index 1 is the original data.
    
    Parameters:
        data (torch.Tensor, numpy.ndarray, or dict): The original data.
        noise_level (float): The level of noise to add.
        num_noisy_datasets (int): The number of noisy datasets to generate.
        output_dir (str, optional): The base directory where datasets will be saved.
        experiment_id (str, optional): An identifier for the experiment.
        
    Returns:
        dict: A dictionary with keys 'noisy_dataset_{index}' and corresponding tensor/array values.
    """
    # Initialize the datasets dictionary using defaultdict
    datasets = defaultdict(lambda: None)
    
    # Extract the tensor from the input if data is a dict (use the first key's value)
    if isinstance(data, dict):
        key = next(iter(data))
        tensor = data[key]
    else:
        tensor = data
    
    # Add the original data as the first dataset
    datasets["noisy_dataset_1"] = tensor
    
    # Helper function to add noise based on the data type
    def add_noise(item):
        if isinstance(item, torch.Tensor):
            noise = noise_level * torch.randn_like(item)
            return item + noise
        elif isinstance(item, np.ndarray):
            noise = noise_level * np.random.randn(*item.shape).astype(item.dtype)
            return item + noise
        else:
            raise TypeError("Unsupported data type. Expected torch.Tensor or numpy.ndarray.")
    
    # Generate additional noisy datasets
    for idx in range(2, num_noisy_datasets + 1):
        datasets[f"noisy_dataset_{idx}"] = add_noise(tensor)
    
    # Save datasets to disk if an output directory is provided
    if output_dir is not None:
        base_path = Path(output_dir)
        noisy_datasets_folder = base_path / "noisy_datasets"
        if experiment_id:
            noisy_datasets_folder = noisy_datasets_folder / f"experiment_{experiment_id}"
        noisy_datasets_folder.mkdir(parents=True, exist_ok=True)
        
        # Save each dataset
        for key, dataset in tqdm(datasets.items(), desc="Saving datasets", total=len(datasets)):
            dataset_folder = noisy_datasets_folder / key
            dataset_folder.mkdir(parents=True, exist_ok=True)
            file_path = dataset_folder / "data.pt"
            torch.save(dataset, file_path)
    
    return datasets


def reconstruct_tensor(
    A_tensor: Union[torch.Tensor, np.ndarray],
    x_hat:   Union[torch.Tensor, np.ndarray],
    zero_threshold: float = 1e-4,
    decimals: int = 3
) -> Optional[torch.Tensor | np.ndarray]:
    """
    Reconstruct A · x_hat and round the result to the requested precision.

    Parameters
    ----------
    A_tensor : torch.Tensor | np.ndarray
        Basis tensor A with shape (*spatial_dims, W).
    x_hat    : torch.Tensor | np.ndarray
        Coefficient vector of shape (W,) or (W, 1).
    zero_threshold : float, default 1e-4
        Entries with |value| < threshold are set to zero before rounding.
    decimals : int, default 4
        Number of decimal places to keep in the final tensor.

    Returns
    -------
    Reconstructed tensor (same backend as the inputs) or None on failure.
    """
    try:
        # Convert NumPy inputs to torch for uniform handling
        if isinstance(A_tensor, np.ndarray):
            A_tensor = torch.from_numpy(A_tensor)
        if isinstance(x_hat, np.ndarray):
            x_hat = torch.from_numpy(x_hat)

        # Ensure computation on CPU for safety
        if A_tensor.device != x_hat.device:
            print("Warning: tensors on different devices – moving to CPU.")
        A_tensor, x_hat = A_tensor.cpu(), x_hat.cpu()

        # Infer contraction mode and reconstruct
        mode = auto_select_mode(A_tensor, x_hat.squeeze())  # helper defined elsewhere
        X_rec = tl.tenalg.mode_dot(A_tensor, x_hat.squeeze(), mode=mode)

        # Suppress near‑zero noise
        if isinstance(X_rec, torch.Tensor):
            X_rec = X_rec.clone()
            X_rec[torch.abs(X_rec) < zero_threshold] = 0
            # Round to the desired precision
            factor = 10 ** decimals
            X_rec = torch.round(X_rec * factor) / factor
        else:  # NumPy array
            X_rec = X_rec.copy()
            X_rec[np.abs(X_rec) < zero_threshold] = 0
            X_rec = np.round(X_rec, decimals=decimals)

        print(f"Reconstructed tensor shape: {X_rec.shape}, mode used: {mode}")
        return X_rec

    except (ValueError, TypeError) as err:
        print(f"Reconstruction error: {err}")
        return None

def to_torch_tensor(arr: Union[np.ndarray, torch.Tensor], device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Convert a NumPy array or PyTorch tensor to a TensorLy tensor on a specified device.
    
    If the input is already a torch.Tensor, it is moved to the desired device with the given dtype.
    If it's a NumPy array, it is converted using TensorLy.
    """
    if isinstance(arr, torch.Tensor):
        return arr.to(device=device, dtype=dtype)
    elif isinstance(arr, np.ndarray):
        return torch.from_numpy(arr).to(dtype=dtype, device=device)
    else:
        try:
            return tl.tensor(arr, dtype=dtype, device=device)
        except Exception as e:
            raise TypeError("Input must be a NumPy array or a PyTorch tensor.") from e

def get_torch_device(device: str = 'cpu') -> torch.device:
    """
    Convert a device string into a torch.device.
    
    Parameters:
    -----------
    device : str
        A string indicating the device type. Options are 'cpu', 'cuda', or 'mps'.
    Returns:
    --------
    torch.device
        The corresponding torch device.
    
    Raises:
    -------
    ValueError
        If 'mps' is requested but not available on the system.
    """
    # Decide on the device
    device = device.lower()
    if device == 'cuda':
        # Only works if you installed PyTorch with CUDA support + have an Nvidia GPU
        tl.set_backend('pytorch')
        device = torch.device('cuda')
    elif device == 'mps':
        # Only valid if you have a Mac with Apple Silicon + PyTorch 1.12+ with MPS support
        if not torch.backends.mps.is_available():
            raise ValueError("MPS not available on this system or with current PyTorch.")
        tl.set_backend('pytorch')
        device = torch.device('mps')
    else:
        # CPU fallback
        # You can also do tl.set_backend('numpy') if you prefer to revert to the classic backend
        tl.set_backend('pytorch')  # or 'numpy'
        device = torch.device('cpu')

    return device

def build_Y_matrices(tensors: Dict[str, Union[np.ndarray, torch.Tensor]],
                     P: Union[np.ndarray, torch.Tensor, Dict[str, torch.Tensor]],
                     device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Apply sensor mask(s) *P* to all test tensors and return Y matrices.

    Parameters
    ----------
    tensors : Dict[str, array | tensor]
        Mapping *subject → data tensor*.
    P : array | tensor | Dict[str, tensor]
        • Single mask (shared by all subjects) – ndarray / torch.Tensor.
        • Individual masks – dict ``{subject: P_subj}``.
    device : str, default 'cpu'
        Target device.
    """
    # Helper: convert mask to torch once
    def _to_mask(mask):
        return to_torch_tensor(mask, device=device, dtype=torch.int32)

    multiple_masks = isinstance(P, dict)

    if not multiple_masks:
        P_tensor = _to_mask(P)
    
    Y_matrices = {}
    for subject, tensor in tensors.items():
        tensor_torch = to_torch_tensor(tensor, device=device)
        # Choose mask
        if multiple_masks:
            if subject not in P:
                raise KeyError(f"No mask provided for subject '{subject}'.")
            P_tensor = _to_mask(P[subject])
        # Shape check
        if P_tensor.shape != tensor_torch.shape[:-1]:
            raise ValueError(
                f"Sensor mask shape {P_tensor.shape} does not match spatial "
                f"dimensions {tensor_torch.shape[:-1]} for subject '{subject}'."
            )
        # Broadcast along slice axis
        Y = tensor_torch * P_tensor.unsqueeze(-1)
        Y_matrices[subject] = Y
    return Y_matrices

def build_wells_matrix(wells_dict, tensor_shape, device='cpu'):
    """Create a binary sensor-mask matrix *per subject* based on wells coordinates.

    Parameters
    ----------
    wells_dict : Dict[str, List[List[int]]]
        Mapping *subject → list([[i, j], ...])* with well positions.
    tensor_shape : Tuple[int, int]
        Spatial shape of data / basis tensors ``(H, W)``.
    device : str | torch.device, default 'cpu'
        Target device for created tensors.

    Returns
    -------
    Dict[str, torch.Tensor]
        ``{subject: P}``, where each *P* has shape ``(H, W)`` and contains 1 at
        well positions, 0 elsewhere.  Coordinates that fall outside the spatial
        extent are silently ignored (filtered).
    """
    H, W = tensor_shape[:2]
    wells_matrices = {}

    for subject, coords_list in wells_dict.items():
        P = torch.zeros(H, W, device=device)
        for i, j in coords_list:
            if 0 <= i < H and 0 <= j < W:
                P[i, j] = 1
        wells_matrices[subject] = P

    return wells_matrices
