# import numpy as np
# import tensorly as tl
# from typing import Dict, Union, List

# def compute_modal_tensor(core: np.ndarray, factors: List[np.ndarray]) -> np.ndarray:
#     """
#     Compute the time-insensitive modal tensor for a single subject from a Tucker decomposition.
#     Handles both 3D and 4D tensors.

#     Parameters
#     ----------
#     core : np.ndarray
#         Core tensor from Tucker decomposition.
#     factors : List[np.ndarray]
#         Factor matrices from Tucker decomposition.

#     Returns
#     -------
#     np.ndarray
#         Resulting modal tensor.
#     """
#     ndim = core.ndim
#     # Check that number of factor matrices matches core tensor dimensions
#     if len(factors) != ndim:
#         raise ValueError(f"Number of factor matrices ({len(factors)}) does not match core tensor dimensions ({ndim}).")
    
#     # Instead of looping over each time slice, apply multi_mode_dot
#     # to multiply core with all factors except the last one.
#     # This vectorizes the operation over the entire core.
#     modal_tensor = tl.tenalg.multi_mode_dot(core, factors[:-1], modes=list(range(len(factors)-1)))
    
#     # Convert the resulting tensor to a NumPy array (if needed)
#     modal_tensor = tl.to_numpy(modal_tensor)
#     return modal_tensor


# def process_all_subjects(
#     cores: Union[Dict[str, np.ndarray], np.ndarray],
#     factors: Union[Dict[str, List[np.ndarray]], List[np.ndarray]],
# ) -> Dict[str, np.ndarray]:
#     """
#     Compute modal tensors for all subjects or a single subject.

#     Parameters
#     ----------
#     cores : Dict or np.ndarray
#         Core tensors for all subjects or a single subject.
#     factors : Dict or List
#         Factor matrices for all subjects or a single subject.

#     Returns
#     -------
#     Dict[str, np.ndarray]
#         Dictionary of computed modal tensors.
#     """
#     M_tensors = {}

#     if isinstance(cores, dict) and isinstance(factors, dict):
#         for subject in cores:
#             if subject not in factors:
#                 raise KeyError(f"Subject '{subject}' is missing in factors dictionary.")
#             modal_tensor = compute_modal_tensor(cores[subject], factors[subject])
#             M_tensors[subject] = modal_tensor
#             print(f"Processed subject '{subject}': Modal tensor shape {modal_tensor.shape}")
#     elif isinstance(cores, np.ndarray) and isinstance(factors, list):
#         modal_tensor = compute_modal_tensor(cores, factors)
#         M_tensors["single_subject"] = modal_tensor
#         print(f"Processed single subject: Modal tensor shape {modal_tensor.shape}")
#     else:
#         raise TypeError("Invalid input types: cores and factors must both be dict or both be a single subject.")

#     return M_tensors


# def stack_all_modes(M_tensors: Dict[str, np.ndarray]) -> tl.tensor:
#     """
#     Stack all modal tensors from multiple subjects along the time dimension.

#     Parameters
#     ----------
#     M_tensors : Dict[str, np.ndarray]
#         Dictionary of modal tensors from multiple subjects.

#     Returns
#     -------
#     tl.tensor
#         Final stacked tensor.
#     """
#     if not M_tensors:
#         raise ValueError("M_tensors is empty. No data to stack.")

#     # Ensure all modal tensors have the same dimensions
#     first_tensor = next(iter(M_tensors.values()))
#     ndim = first_tensor.ndim
#     time_dim = first_tensor.shape[-1]

#     # Use a list comprehension to extract all slices from each subject
#     all_modes = [modal_tensor[..., t] 
#                  for modal_tensor in M_tensors.values() 
#                  for t in range(modal_tensor.shape[-1])]

#     # Stack all slices along a new last dimension
#     A_tensor = tl.tensor(np.stack(all_modes, axis=-1))
#     print(f"Final stacked tensor shape: {A_tensor.shape}")
#     return A_tensor


# import time
# import warnings
# from dataclasses import dataclass
# from typing import Union, Tuple, List, Optional, Dict, Any

# import numpy as np
# import torch
# import tensorly as tl

# from TBMD.utils.utils import get_torch_device, to_torch_tensor



# class TensorCompressiveSensing:
#     """
#     Implements the tensor-based compressive sensing algorithm (Algorithm 3)
#     for 3D or 4D tensors using PyTorch, without mutating the original inputs.
#     """
    
#     def __init__(
#         self,
#         A: Union[np.ndarray, torch.Tensor],
#         P: Union[np.ndarray, torch.Tensor],
#         Y: Union[np.ndarray, torch.Tensor],
#         max_iter: int,
#         epsilon: float,
#         lambd: float,
#         delta_0: float,
#         delta_max: float,
#         solver_method: str = "triangular",
#         device: str = "cpu",
#         dtype: torch.dtype = torch.float32
#     ):
#         self.device = get_torch_device(device)
#         self.dtype = dtype
        
#         # Сохраняем оригинальные типы данных
#         self.orig_P_dtype = None
#         if isinstance(P, torch.Tensor):
#             self.orig_P_dtype = P.dtype
#         elif isinstance(P, np.ndarray):
#             self.orig_P_dtype = torch.from_numpy(np.empty(1, dtype=P.dtype)).dtype
#         else:
#             self.orig_P_dtype = torch.float32  # Значение по умолчанию

#         # Определяем, являются ли P и других маски целочисленными
#         self.P_is_int = self.orig_P_dtype in [torch.int32, torch.int64, torch.uint8, torch.int8, torch.int16]
        
#         # Преобразуем входные данные в тензоры и делаем независимые копии
#         self.A = to_torch_tensor(A, device="cpu", dtype=self.dtype).detach().clone()
        
#         # Для P сохраняем оригинальный тип, если это int
#         if self.P_is_int:
#             self.P = to_torch_tensor(P, device="cpu", dtype=self.orig_P_dtype).detach().clone()
#             # Для операций преобразуем в рабочий тип, но сохраняем исходный
#             self._P_float = self.P.to(dtype=torch.float32)
#         else:
#             self.P = to_torch_tensor(P, device="cpu", dtype=self.dtype).detach().clone()
#             self._P_float = self.P

#         self.Y_measured = to_torch_tensor(Y, device="cpu", dtype=self.dtype).detach().clone()
        
#         # Перемещаем все на нужное устройство после конвертации
#         self.A = self.A.to(device=self.device)
#         self.P = self.P.to(device=self.device)
#         self._P_float = self._P_float.to(device=self.device)
#         self.Y_measured = self.Y_measured.to(device=self.device)

#         # Остальные параметры алгоритма
#         self.max_iter = max_iter
#         self.epsilon = epsilon
#         self.lambd = lambd
#         self.delta_0 = delta_0
#         self.delta_max = delta_max
#         self.solver_method = solver_method

#         # Валидация размерностей
#         if self.A.ndim < 2:
#             raise ValueError("A must have at least 2 dims: (*spatial_dims, W).")
#         *self.spatial_dims, self.W = self.A.shape
#         self.num_spatial = int(np.prod(self.spatial_dims))

#         if tuple(self.P.shape) != tuple(self.spatial_dims):
#             raise ValueError(f"P shape {self.P.shape} must match A's spatial dims {self.spatial_dims}.")
#         if tuple(self.Y_measured.shape) != tuple(self.spatial_dims):
#             raise ValueError(f"Y shape {self.Y_measured.shape} must match A's spatial dims {self.spatial_dims}.")

#         # Инициализация переменных ADMM
#         self.x_n = torch.zeros((self.W, 1), dtype=self.dtype, device=self.device)
#         self.d_n = torch.zeros_like(self.x_n)
#         self.p_n = torch.zeros_like(self.x_n)
#         self.delta_curr = self.delta_0

#         # Создаем маскированные и сплющенные A и Y
#         self.A_sensors, self.Y_sensors = self._process_inputs()

#         # Предварительно вычисляем постоянные термины
#         self.A_T_A = self.A_sensors.T @ self.A_sensors
#         self.A_T_Y = self.A_sensors.T @ self.Y_sensors
#         self.I_W = torch.eye(self.W, dtype=self.dtype, device=self.device)

#     def _process_inputs(self):
#         # Используем копию с правильным типом для операций
#         P_exp = self._P_float.clone().unsqueeze(-1)  
#         A_masked = self.A.clone() * P_exp     # Используем копию A
#         Y_masked = self.Y_measured.clone() * self._P_float.clone()  # Используем копии Y и P_float

#         # Сплющиваем пространственные размерности
#         A_flat = A_masked.reshape(self.num_spatial, self.W)
#         Y_flat = Y_masked.reshape(self.num_spatial, 1)

#         # Для выборки используем целочисленную копию P, если это нужно
#         if self.P_is_int:
#             P_flat = self.P.clone().reshape(self.num_spatial)  # Целочисленный тип
#         else:
#             P_flat = self._P_float.clone().reshape(self.num_spatial)
            
#         # Используем пороговое значение 0.5 для float, чтобы избежать погрешностей вычислений
#         if not self.P_is_int:
#             sensor_idx = torch.nonzero(P_flat > 0.5, as_tuple=False).view(-1)
#         else:
#             sensor_idx = torch.nonzero(P_flat, as_tuple=False).view(-1)

#         A_s = A_flat.index_select(0, sensor_idx)
#         Y_s = Y_flat.index_select(0, sensor_idx)
#         return A_s, Y_s

#     def _solve_linear_system(self, LHS, RHS):
#         if self.solver_method == "triangular":
#             try:
#                 L = torch.linalg.cholesky(LHS)
#                 y = torch.linalg.solve_triangular(L, RHS, upper=False)
#                 x = torch.linalg.solve_triangular(L.T, y, upper=True)
#                 return x
#             except torch.linalg.LinAlgError:
#                 return torch.linalg.solve(LHS, RHS)
#         elif self.solver_method == "direct":
#             return torch.linalg.solve(LHS, RHS)
#         else:
#             raise ValueError("solver_method must be 'triangular' or 'direct'")

#     def _solve_linear_system(self, LHS, RHS):
#         # Add small regularization to improve conditioning
#         reg_LHS = LHS + 1e-8 * self.I_W
        
#         if self.solver_method == "triangular":
#             try:
#                 L = torch.linalg.cholesky(reg_LHS)
#                 y = torch.linalg.solve_triangular(L, RHS, upper=False)
#                 x = torch.linalg.solve_triangular(L.T, y, upper=True)
#                 return x
#             except torch.linalg.LinAlgError:
#                 # Fall back to direct solve with regularization
#                 return torch.linalg.solve(reg_LHS, RHS)
#         elif self.solver_method == "direct":
#             return torch.linalg.solve(reg_LHS, RHS)
#         else:
#             raise ValueError("solver_method must be 'triangular' or 'direct'")

#     def _shrinkage_thresholding(self, x_hat, p_n, thresh):
#         """
#         Implement the d-update step according to equation (33) in the paper:
#         d_{(n)} ← max(0, x̂ + p_{(n-1)} - ε/δ_{(n-1)}) - max(0, -x̂ - p_{(n-1)} - ε/δ_{(n-1)})
#         """
#         pos_part = torch.clamp(x_hat + p_n - thresh, min=0.0)
#         neg_part = torch.clamp(-x_hat - p_n - thresh, min=0.0)
#         return pos_part - neg_part

#     def solve(self) -> torch.Tensor:
#         """
#         Runs Algorithm 3 (Tensor-based Compressive Sensing Algorithm) and returns x_hat (shape: W,).
#         """
#         x_hat = torch.zeros_like(self.x_n)

#         for _ in range(self.max_iter):
#             # 1) x-update: x_{(n)} ← (A^T·A_dot·A + δ_{(n-1)}·eye(w))^(-1)·(A^T×_3 Y + δ_{(n-1)}·(d_{(n-1)} - p_{(n-1)}))
#             rhs = self.A_T_Y + self.delta_curr * (self.d_n - self.p_n)
#             lhs = self.A_T_A + self.delta_curr * self.I_W
#             self.x_n = self._solve_linear_system(lhs, rhs)

#             # 2) x_hat: x̂ ← λ·x_{(n)} + (1-λ)·d_{(n-1)}
#             d_prev = self.d_n.clone()
#             x_hat = self.lambd * self.x_n + (1.0 - self.lambd) * d_prev

#             # 3) d-update via shrinkage-thresholding according to equation (33)
#             # Ensure delta is never zero to avoid division by zero
#             delta_safe = max(self.delta_curr, 1e-10)
#             thresh = self.epsilon / delta_safe
#             self.d_n = self._shrinkage_thresholding(x_hat, self.p_n, thresh)

#             # 4) p-update: p_{(n)} ← p_{(n-1)} + x̂ - d_{(n)}
#             self.p_n = self.p_n + (x_hat - self.d_n)

#             # 5) delta-update: δ_{(n)} ← min(δ_{(n-1)}, δ_max)
#             self.delta_curr = min(self.delta_curr, self.delta_max)
            
#             # Check for NaN values and handle them
#             if torch.isnan(x_hat).any():
#                 print("Warning: NaN values detected in x_hat. Terminating early.")
#                 # Replace NaNs with zeros or previous valid values
#                 x_hat = torch.where(torch.isnan(x_hat), torch.zeros_like(x_hat), x_hat)
#                 break

#         return x_hat.view(-1)
    
# max_iter = 1000
# epsilon = 1e-2
# lambd = 0.95
# delta_0 = 1.0
# delta_max = 1.0
# solver_method = "triangular"    

# cs_solver = TensorCompressiveSensing(
#     A=A_tensor,
#     P=P,
#     Y=Y,
#     max_iter=max_iter,
#     epsilon=epsilon,
#     lambd=lambd,
#     delta_0=delta_0,
#     delta_max=delta_max,
#     solver_method=solver_method,
#     device="cpu"   # 'cpu' or 'cuda' or 'mps'
# )
# x_hat = cs_solver.solve()


# import numpy as np
# import tensorly as tl
# import matplotlib.pyplot as plt
# from typing import Union, Optional, Tuple

# class TensorTubeQRDecomposition:
#     """
#     Implements a tensor-based QR factorization with 'tube pivoting' for a 3D or 4D tensor.
#     The tensor is assumed to have shape (s1, s2, ..., s_n, k) where the last axis (k)
#     corresponds to the tube dimension and the remaining axes define the sensor domain.
#     """

#     def __init__(self,
#                  tensor: Union[np.ndarray, tl.tensor],
#                  N: int,
#                  rejection_domain: Optional[np.ndarray] = None,
#                  random_state: Optional[int] = None,
#                  check_orthogonality: bool = False):
#         """
#         Initialize the object and set up the domain mask.
#         """
#         if random_state is not None:
#             np.random.seed(random_state)
        
#         # Convert any tensorly tensor to a NumPy array if needed.
#         if tl.is_tensor(tensor):
#             tensor = tl.to_numpy(tensor)
            
#         if tensor.ndim < 3:
#             raise ValueError("Input tensor must have at least 3 dimensions.")
            
#         # Define spatial dimensions and tube dimension.
#         self.spatial_shape = tensor.shape[:-1]
#         self.k = tensor.shape[-1]
        
#         # For consistency with the original code, you may wish to require that the spatial
#         # domain is 2D or 3D. For now we allow any spatial shape.
#         if not (1 <= N <= self.k):
#             raise ValueError(f"N must be in [1, {self.k}]. Got N={N}.")

#         self.tensor = tensor.astype(np.float32)
#         self.N = N

#         # Set up the available sensor positions mask.
#         if rejection_domain is None:
#             self.available = np.ones(self.spatial_shape, dtype=bool)
#         else:
#             if rejection_domain.shape != self.spatial_shape:
#                 raise ValueError(
#                     "rejection_domain shape must match the spatial dimensions "
#                     f"{self.spatial_shape}. Got {rejection_domain.shape}."
#                 )
#             self.available = rejection_domain.copy()
        
#         # Outputs: sensor (pivot) matrix, orthonormal factor, and updated tensor.
#         # P will have the same shape as the sensor domain.
#         self.P: np.ndarray = None  
#         self.Q: np.ndarray = None  
#         self.R: np.ndarray = None  
#         self.check_orthogonality = check_orthogonality

#     def _compute_householder_vector(self, v: np.ndarray) -> np.ndarray:
#         """
#         Compute the Householder vector 'u' for a 1D vector v.
#         """
#         sigma = np.linalg.norm(v)
#         if sigma < 1e-12:
#             return np.zeros_like(v)
        
#         v1 = v[0]
#         sign_v1 = np.sign(v1) if v1 != 0 else 1
#         e1 = np.zeros_like(v)
#         e1[0] = 1
        
#         u = v + sign_v1 * sigma * e1
#         denom = np.sqrt(2 * sigma * (sigma + abs(v1)))
#         if denom < 1e-12:
#             return np.zeros_like(v)
#         return u / denom

#     def _get_pivot_position(self, R: np.ndarray, d: int) -> Tuple:
#         """
#         Vectorized pivot selection based on the maximum L1 norm of tubes along the last axis,
#         starting from index d.
#         Returns a tuple of indices corresponding to the sensor domain.
#         """
#         # Compute the L1 norm along the tube (last axis) from d onward.
#         norms = np.sum(np.abs(R[..., d:]), axis=-1)
#         # Set unavailable positions to -infinity to avoid their selection.
#         norms[~self.available] = -np.inf
#         flat_index = np.argmax(norms)
#         return np.unravel_index(flat_index, self.available.shape)

#     def factorize(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#         """
#         Perform tensor-based QR factorization with tube pivoting.
#         """
#         R = self.tensor.copy()  # Working copy of the tensor.
#         Q = np.eye(self.k, dtype=np.float32)  # Q remains a square matrix on the tube axis.
#         # P marks the sensor positions where a pivot has been chosen.
#         self.P = np.zeros(self.available.shape, dtype=np.int32)
        
#         # Iterate over pivot indices up to min(N, k).
#         for d in range(min(self.N, self.k)):
#             # 1) Find the pivot in the sensor domain.
#             pivot_idx = self._get_pivot_position(R, d)
            
#             # 2) Get the pivot tube starting at index d.
#             pivot_tube = R[pivot_idx + (slice(d, None),)]
#             if np.sum(np.abs(pivot_tube)) < 1e-12:
#                 break

#             # 3) Mark the pivot in the sensor matrix and update availability.
#             self.P[pivot_idx] = 1
#             self.available[pivot_idx] = False

#             # 4) Compute the Householder vector for the pivot tube.
#             u = self._compute_householder_vector(pivot_tube)
#             if np.linalg.norm(u) < 1e-12:
#                 continue

#             # 5) Apply the Householder reflection to R for indices d onward.
#             # R[..., d:] has shape: spatial_shape + (k-d,)
#             sub_R = R[..., d:]
#             # Flatten the sensor domain into a 2D array where each row is a tube.
#             sub_R_2d = sub_R.reshape(-1, sub_R.shape[-1])
#             # Compute the reflection factor for each tube.
#             alpha = sub_R_2d @ u
#             # Update all tubes in place.
#             sub_R_2d -= 2.0 * alpha[:, None] * u
            
#             # 6) Update the orthonormal matrix Q using the same reflector.
#             Q_block = Q[:, d:]
#             alpha_q = Q_block @ u.reshape(-1, 1)
#             Q[:, d:] = Q_block - 2.0 * alpha_q @ u.reshape(1, -1)
            
#             # Optionally check orthogonality of Q.
#             if self.check_orthogonality and not np.allclose(Q.T @ Q, np.eye(self.k), atol=1e-5):
#                 print(f"Warning: Q lost orthogonality at pivot step {d}")

#         self.Q = Q
#         self.R = R
#         return self.P, self.Q, self.R

#     def check_factorization(self, tol: float = 1e-6) -> Tuple[bool, float]:
#         """
#         Check the orthogonality of Q and compute the relative reconstruction error.
#         """
#         if any(x is None for x in [self.P, self.Q, self.R]):
#             raise ValueError("Run factorize() first.")
        
#         is_ortho = np.allclose(self.Q.T @ self.Q, np.eye(self.k), atol=tol)
#         # Contract the last axis of R with the first axis of Q.T.
#         RQ_T = np.tensordot(self.R, self.Q.T, axes=([-1], [0]))
#         diff = self.tensor - RQ_T
#         rel_error = np.linalg.norm(diff) / np.linalg.norm(self.tensor)
#         return is_ortho, rel_error

#     def visualize_sensor_placement(self) -> None:
#         """
#         Visualize sensor placements (where P==1) on the sensor grid.
#         For a 2D sensor domain, a single image is shown.
#         For a higher-dimensional sensor domain, a projection to 2D is used.
#         """
#         if self.P is None:
#             raise ValueError("Run factorize() first.")
        
#         # If sensor domain is 2D, visualize directly.
#         if len(self.P.shape) == 2:
#             sensor_map = self.P
#             fig, ax = plt.subplots(figsize=(self.P.shape[1] / 10, self.P.shape[0] / 10))
#         else:
#             # For sensor domains with >2 dimensions, project to 2D.
#             # For example, take the maximum over the extra dimensions.
#             sensor_map = np.max(self.P, axis=tuple(range(2, len(self.P.shape))))
#             fig, ax = plt.subplots(figsize=(sensor_map.shape[1] / 10, sensor_map.shape[0] / 10))
        
#         ax.set_facecolor("black")
#         ax.imshow(np.zeros(sensor_map.shape), cmap="gray", origin="upper")
#         sensor_positions = np.argwhere(sensor_map == 1)
#         if sensor_positions.size > 0:
#             ax.scatter(sensor_positions[:, 1], sensor_positions[:, 0],
#                        c="red", s=20, label="Sensors")
#         ax.set_title(f"Tube-Pivot QR: Sensor Placement (N={self.N})", color="white")
#         ax.axis("off")
#         plt.show()


# import numpy as np
# import torch
# import tensorly as tl
# import matplotlib.pyplot as plt
# from typing import Union, Optional, Tuple

# # Tell TensorLy to use the PyTorch backend globally
# tl.set_backend('pytorch')


# def _to_device_tensor(
#     arr: Union[np.ndarray, torch.Tensor],
#     device: torch.device,
#     dtype: torch.dtype = torch.float32
# ) -> torch.Tensor:
#     """
#     Convert a NumPy array (or PyTorch tensor) to a TensorLy (PyTorch) tensor on the specified device/dtype.
#     """
#     if isinstance(arr, np.ndarray):
#         return tl.tensor(arr, dtype=dtype, device=device)
#     elif isinstance(arr, torch.Tensor):
#         return arr.to(device=device, dtype=dtype)
#     else:
#         raise TypeError("Input must be a NumPy array or PyTorch tensor.")


# class TensorTubeQRDecomposition:
#     """
#     Implements a tensor-based QR factorization with 'tube pivoting' for a 3D or 4D (or higher) tensor.
#     The tensor is assumed to have shape (s1, s2, ..., sN, k) where the last axis (k)
#     corresponds to the 'tube' dimension and the remaining axes define the sensor domain.
#     """

#     def __init__(
#         self,
#         tensor: Union[np.ndarray, torch.Tensor],
#         N: int,
#         rejection_domain: Optional[Union[np.ndarray, torch.Tensor]] = None,
#         random_state: Optional[int] = None,
#         check_orthogonality: bool = False,
#         device: str = "cpu"
#     ):
#         """
#         Parameters
#         ----------
#         tensor : Union[np.ndarray, torch.Tensor]
#             The input tensor of shape (..., k).
#         N : int
#             Maximum number of pivot tubes (sensors) to select (1 <= N <= k).
#         rejection_domain : Optional[Union[np.ndarray, torch.Tensor]]
#             Boolean mask of shape (...), same as the spatial domain of 'tensor'
#             (excluding the last dimension), specifying which locations are valid.
#         random_state : Optional[int]
#             Seed for reproducibility.
#         check_orthogonality : bool
#             Whether to check Q's orthogonality at each step (slower).
#         device : str
#             'cpu', 'cuda', or 'mps'.
#         """
#         # Set random seed for reproducibility
#         if random_state is not None:
#             np.random.seed(random_state)

#         # Choose a PyTorch device
#         device = device.lower()
#         if device == 'cuda':
#             if not torch.cuda.is_available():
#                 raise ValueError("CUDA is not available on this system/PyTorch build.")
#             self.device = torch.device('cuda')
#         elif device == 'mps':
#             if not torch.backends.mps.is_available():
#                 raise ValueError("MPS not available on this system or in current PyTorch.")
#             self.device = torch.device('mps')
#         else:
#             self.device = torch.device('cpu')

#         # Convert 'tensor' to a PyTorch tensor on the chosen device
#         self.tensor = _to_device_tensor(tensor, self.device)
#         if self.tensor.ndim < 3:
#             raise ValueError("Input tensor must have at least 3 dimensions (…, k).")

#         # Spatial shape and the 'tube' dimension
#         self.spatial_shape = self.tensor.shape[:-1]
#         self.k = self.tensor.shape[-1]
#         if not (1 <= N <= self.k):
#             raise ValueError(f"N must be in [1, {self.k}]. Got N={N}.")
#         self.N = N

#         # Build or validate the rejection domain
#         if rejection_domain is None:
#             # Everything is available
#             self.available = torch.ones(self.spatial_shape, dtype=torch.bool, device=self.device)
#         else:
#             rej_dom_torch = _to_device_tensor(rejection_domain, self.device, dtype=torch.bool)
#             if rej_dom_torch.shape != self.spatial_shape:
#                 raise ValueError(
#                     "rejection_domain shape must match the spatial dimensions "
#                     f"{self.spatial_shape}. Got {rej_dom_torch.shape}."
#                 )
#             self.available = rej_dom_torch

#         # Initialize placeholders
#         self.check_orthogonality = check_orthogonality
#         self.P: Optional[torch.Tensor] = None  # sensor (pivot) array, shape = spatial_shape
#         self.Q: Optional[torch.Tensor] = None  # orthonormal factor, shape = (k, k)
#         self.R: Optional[torch.Tensor] = None  # updated tensor, same shape as self.tensor

#     def _compute_householder_vector(self, v: torch.Tensor) -> torch.Tensor:
#         """
#         Compute the 1D Householder vector 'u' for a tube 'v'.
#         """
#         sigma = torch.norm(v)
#         if sigma < 1e-12:
#             return torch.zeros_like(v)

#         v1 = v[0]
#         sign_v1 = torch.sign(v1) if v1 != 0 else torch.tensor(1.0, device=self.device, dtype=v.dtype)
#         e1 = torch.zeros_like(v)
#         e1[0] = 1.0

#         u = v + sign_v1 * sigma * e1
#         denom = torch.sqrt(2 * sigma * (sigma + torch.abs(v1)))
#         if denom < 1e-12:
#             return torch.zeros_like(v)
#         return u / denom

#     def _get_pivot_position(self, R: torch.Tensor, d: int) -> Tuple[int, ...]:
#         """
#         Vectorized pivot selection:
#          - Compute L1 norm of each tube from index d onward along the last dimension
#          - Exclude positions marked as unavailable
#          - Return the flattened argmax, then unravel to get a multi-dimensional index
#         """
#         # norms shape = spatial_shape
#         norms = torch.sum(torch.abs(R[..., d:]), dim=-1)
#         # mask out unavailable positions
#         norms = torch.where(self.available, norms, torch.tensor(float('-inf'), device=self.device))
#         flat_index = torch.argmax(norms).item()  # get Python int
#         # Use NumPy unravel_index on CPU for convenience
#         return np.unravel_index(flat_index, self.spatial_shape)

#     def factorize(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Perform tensor-based QR factorization with tube pivoting.
        
#         Returns
#         -------
#         P : (…)-shaped int Tensor
#             Pivot indicator array (sensor domain). '1' indicates a chosen sensor.
#         Q : (k, k)-shaped float Tensor
#             Orthonormal factor for the tube dimension.
#         R : same shape as input Tensor
#             Post-factorization result.
#         """
#         # Copy R
#         self.R = self.tensor.clone()
#         # Q is (k x k)
#         self.Q = torch.eye(self.k, device=self.device, dtype=torch.float32)
#         # P marks sensor picks (same shape as domain)
#         self.P = torch.zeros(self.spatial_shape, device=self.device, dtype=torch.int32)

#         # Up to min(N, k) pivot steps
#         for d in range(min(self.N, self.k)):
#             # 1) Pivot selection
#             pivot_idx = self._get_pivot_position(self.R, d)

#             # 2) Check pivot tube
#             # pivot_tube shape = (k-d,)
#             # We pick R[pivot_idx..., d:]
#             # pivot_idx is a Python tuple (i1, i2, ...)
#             # We need to assemble slicing
#             # E.g. R[pivot_idx + (slice(d, None), )]
#             pivot_tube = self.R[pivot_idx + (slice(d, None),)]
#             if torch.sum(torch.abs(pivot_tube)) < 1e-12:
#                 # Tube is negligible => stop
#                 break

#             # 3) Mark pivot
#             self.P[pivot_idx] = 1
#             self.available[pivot_idx] = False

#             # 4) Householder vector
#             u = self._compute_householder_vector(pivot_tube)
#             if torch.norm(u) < 1e-12:
#                 continue

#             # 5) Apply reflection to R[..., d:]
#             sub_R = self.R[..., d:]  # shape = (*spatial_shape, k-d)
#             # Flatten the spatial domain so each row is a tube
#             sub_R_2d = sub_R.view(-1, sub_R.shape[-1])  # shape = (prod(spatial_shape), k-d)
#             alpha = sub_R_2d @ u  # shape = (prod(spatial_shape),)
#             sub_R_2d -= 2.0 * alpha.unsqueeze(1) * u

#             # 6) Update Q similarly
#             Q_block = self.Q[:, d:]         # shape = (k, k-d)
#             alpha_q = Q_block @ u.unsqueeze(-1)  # shape = (k, 1)
#             Q_block -= 2.0 * alpha_q * u.unsqueeze(0)

#             if self.check_orthogonality:
#                 # Check if Q remains orthonormal
#                 # NOTE: This can be expensive for large k
#                 test_ortho = torch.allclose(
#                     self.Q.T @ self.Q,
#                     torch.eye(self.k, device=self.device),
#                     atol=1e-5
#                 )
#                 if not test_ortho:
#                     print(f"Warning: Q lost orthogonality at pivot step {d}")

#         return self.P, self.Q, self.R

#     def check_factorization(self, tol: float = 1e-6) -> Tuple[bool, float]:
#         """
#         Check Q's orthogonality and compute the relative reconstruction error.
        
#         Returns
#         -------
#         is_ortho : bool
#             Whether Q is orthonormal within tolerance.
#         rel_error : float
#             Relative Frobenius norm error of (tensor - R @ Q^T).
#         """
#         if any(x is None for x in [self.P, self.Q, self.R]):
#             raise ValueError("Run factorize() first.")

#         # 1) Orthogonality check
#         ident = torch.eye(self.k, device=self.device)
#         is_ortho = torch.allclose(self.Q.T @ self.Q, ident, atol=tol)

#         # 2) Reconstruction error: we want (R x Q^T) along the last dimension
#         # We can do tensordot with last axis of R and first axis of Q^T
#         # i.e. R shape = (..., k), Q^T shape = (k, k)
#         # => RQ^T shape = (..., k)
#         RQ_T = torch.tensordot(self.R, self.Q.T, dims=([self.R.ndim - 1], [0]))
#         # RQ_T will have the same shape as self.tensor (since the last dimension is k).
#         diff = self.tensor - RQ_T
#         rel_error = torch.norm(diff) / torch.norm(self.tensor)
#         return (bool(is_ortho), float(rel_error.item()))

#     def visualize_sensor_placement(self) -> None:
#         """
#         Visualize sensor placements (P==1) on the sensor domain.
        
#         If the sensor domain is 2D, we show it directly.
#         If >2D, we take a maximum projection over dimensions beyond the first 2.
#         """
#         if self.P is None:
#             raise ValueError("Run factorize() first.")

#         # Move P to CPU for plotting
#         p_cpu = self.P.detach().cpu().numpy()

#         if p_cpu.ndim == 2:
#             sensor_map = p_cpu
#         else:
#             # For domain with >2 dims, do a max projection over dims 2..N
#             # (This is arbitrary; you can choose a different projection or slice.)
#             axes_to_proj = tuple(range(2, p_cpu.ndim))
#             sensor_map = np.max(p_cpu, axis=axes_to_proj)

#         fig, ax = plt.subplots(figsize=(sensor_map.shape[1] / 10, sensor_map.shape[0] / 10))
#         ax.set_facecolor("black")
#         ax.imshow(np.zeros(sensor_map.shape), cmap="gray", origin="upper")

#         sensor_positions = np.argwhere(sensor_map == 1)
#         if sensor_positions.size > 0:
#             ax.scatter(sensor_positions[:, 1], sensor_positions[:, 0],
#                        s=20, c="red", label="Sensors")

#         ax.set_title(f"Tube-Pivot QR: Sensor Placement (N={self.N})", color="white")
#         ax.axis("off")
#         plt.show()

# import tensorly as tl
# import numpy as np
# import matplotlib.pyplot as plt
# import concurrent.futures 
# import torch
# from tensorly.decomposition import tucker
# from tensorly.tucker_tensor import tucker_to_tensor
# from typing import Union, List, Dict, Optional
# from TBMD.utils.utils import to_torch_tensor, get_torch_device


# class TuckerDecomposer:
#     """
#     A class to perform Tucker decomposition on tensors or collections of tensors using TensorLy,
#     with options for CPU, CUDA (if available), or MPS on Apple Silicon.
#     """

#     def __init__(self, 
#                 tensors: Union[tl.tensor, np.ndarray, torch.Tensor, Dict[str, tl.tensor], Dict[str, np.ndarray], Dict[str, torch.Tensor]],
#                 ranks: Optional[Union[int, List[int]]] = None,
#                 epsilon: float = 1e-2,
#                 random_state: Optional[int] = None,
#                 device: str = 'cpu',
#                 dtype: torch.dtype = torch.float32):
#         self.epsilon = epsilon
#         self.random_state = random_state
#         self.cores = None
#         self.factors = None 
#         self.reconstructed_tensors = None
#         self.reconstruction_errors = None
#         self.reconstructed_tensor = None
#         self.reconstruction_error = None
#         self.ranks = ranks
        
#         # Decide on the device
#         self.device = get_torch_device(device)
#         self.dtype = dtype

#         # Convert input to the appropriate device
#         if isinstance(tensors, dict):
#             self.is_collection = True
#             self.tensors = {}
#             for key, tensor in tensors.items():
#                 self.tensors[key] = to_torch_tensor(tensor, device=self.device, dtype=self.dtype)
#         elif isinstance(tensors, (np.ndarray, tl.tensor)):
#             self.is_collection = False
#             self.tensors = to_torch_tensor(tensors, device=self.device, dtype=self.dtype)
#         else:
#             raise ValueError("Tensors must be a TensorLy tensor, numpy array, or a dictionary of tensors.")

#     def _determine_ranks(self, tensor_shape: List[int]) -> List[int]:
#         if self.ranks is None:
#             # Default to min dimension
#             min_rank = min(tensor_shape)
#             ranks = [min_rank for _ in tensor_shape]
#         elif isinstance(self.ranks, int):
#             ranks = [self.ranks] * len(tensor_shape)
#         elif isinstance(self.ranks, list):
#             if len(self.ranks) != len(tensor_shape):
#                 raise ValueError("Ranks list must have the same length as tensor modes.")
#             ranks = self.ranks
#         else:
#             raise ValueError("Ranks must be None, int, or list of ints.")
#         return ranks

#     def _decompose_single(self, tensor: tl.tensor, init: Optional[str] = 'svd'):
#         return tucker(tensor, rank=self._determine_ranks(list(tensor.shape)), 
#                       init=init, tol=self.epsilon, random_state=self.random_state)

#     def decompose(self) -> None:
#         if self.is_collection:
#             self.cores = {}
#             self.factors = {}
#             # Check if device is CPU; if so, use multithreading
#             if self.device.type == 'cpu':
#                 with concurrent.futures.ThreadPoolExecutor() as executor:
#                     futures = {
#                         key: executor.submit(self._decompose_single, tensor, 'svd')
#                         for key, tensor in self.tensors.items()
#                     }
#                     for key, future in futures.items():
#                         core, factors = future.result()
#                         self.cores[key] = core
#                         self.factors[key] = factors
#             else:
#                 # On GPU, process sequentially to leverage asynchronous operations
#                 for key, tensor in self.tensors.items():
#                     core, factors = self._decompose_single(tensor, 'svd')
#                     self.cores[key] = core
#                     self.factors[key] = factors
#         else:
#             self.core, self.factors = self._decompose_single(self.tensors, 'svd')

#     def reconstruct(self) -> None:
#         if self.is_collection:
#             self.reconstructed_tensors = {}
#             self.reconstruction_errors = {}
#             if self.device.type == 'cpu':
#                 with concurrent.futures.ThreadPoolExecutor() as executor:
#                     futures = {
#                         key: executor.submit(lambda k: tucker_to_tensor((self.cores[k], self.factors[k])), key)
#                         for key in self.tensors.keys()
#                     }
#                     for key, future in futures.items():
#                         reconstructed = future.result()
#                         error = tl.norm(self.tensors[key] - reconstructed) / tl.norm(self.tensors[key])
#                         self.reconstructed_tensors[key] = reconstructed
#                         self.reconstruction_errors[key] = float(error)
#             else:
#                 for key in self.tensors.keys():
#                     reconstructed = tucker_to_tensor((self.cores[key], self.factors[key]))
#                     error = tl.norm(self.tensors[key] - reconstructed) / tl.norm(self.tensors[key])
#                     self.reconstructed_tensors[key] = reconstructed
#                     self.reconstruction_errors[key] = float(error)
#         else:
#             reconstructed = tucker_to_tensor((self.core, self.factors))
#             error = tl.norm(self.tensors - reconstructed) / tl.norm(self.tensors)
#             self.reconstructed_tensor = reconstructed
#             self.reconstruction_error = float(error)

#     def visualize(self, subjects: Optional[List[str]] = None) -> None:
#         """
#         Visualize the original and reconstructed tensors.
#         """
#         if self.is_collection:
#             if subjects is None:
#                 subjects = list(self.tensors.keys())
#             for subject in subjects:
#                 if len(self.tensors[subject].shape) != 3:
#                     raise ValueError(f"Tensor for subject {subject} is not 3D and cannot be visualized.")
#                 plt.figure(figsize=(12, 6))
#                 plt.subplot(1, 2, 1)
#                 mid_slice = self.tensors[subject][:, :, self.tensors[subject].shape[2] // 2]
#                 plt.imshow(mid_slice, cmap="gray")
#                 plt.title(f"Original (Subject {subject})")
#                 plt.axis("off")
#                 plt.subplot(1, 2, 2)
#                 mid_slice_rec = self.reconstructed_tensors[subject][:, :, self.reconstructed_tensors[subject].shape[2] // 2]
#                 plt.imshow(mid_slice_rec, cmap="gray")
#                 plt.title(f"Reconstructed (Subject {subject})")
#                 plt.axis("off")
#                 plt.tight_layout()
#                 plt.show()
#         else:
#             if len(self.tensors.shape) != 3:
#                 raise ValueError("Tensor is not 3D and cannot be visualized.")
#             plt.figure(figsize=(12, 6))
#             plt.subplot(1, 2, 1)
#             plt.imshow(self.tensors[:, :, self.tensors.shape[2] // 2], cmap="gray")
#             plt.title("Original Tensor")
#             plt.axis("off")
#             plt.subplot(1, 2, 2)
#             plt.imshow(self.reconstructed_tensor[:, :, self.reconstructed_tensor.shape[2] // 2], cmap="gray")
#             plt.title("Reconstructed Tensor")
#             plt.axis("off")
#             plt.tight_layout()
#             plt.show()

#     def get_cores(self):
#         if self.is_collection:
#             if not self.cores:
#                 raise ValueError("Call `decompose()` first.")
#             return self.cores
#         else:
#             if not hasattr(self, 'core') or self.core is None:
#                 raise ValueError("Call `decompose()` first.")
#             return self.core

#     def get_factors(self):
#         if self.is_collection:
#             if not self.factors:
#                 raise ValueError("Call `decompose()` first.")
#             return self.factors
#         else:
#             if not self.factors:
#                 raise ValueError("Call `decompose()` first.")
#             return self.factors

#     def get_reconstruction_errors(self):
#         if not self.is_collection:
#             raise ValueError("Use `get_reconstruction_error` for single tensor.")
#         if not self.reconstruction_errors:
#             raise ValueError("Call `reconstruct()` first.")
#         return self.reconstruction_errors

#     def get_reconstruction_error(self):
#         if self.is_collection:
#             raise ValueError("Use `get_reconstruction_errors` for a collection.")
#         if not hasattr(self, 'reconstruction_error') or self.reconstruction_error is None:
#             raise ValueError("Call `reconstruct()` first.")
#         return self.reconstruction_error

#     def set_ranks(self, ranks: Union[int, List[int], None]) -> None:
#         if not (isinstance(ranks, (int, list)) or ranks is None):
#             raise ValueError("Ranks must be int, list, or None.")
#         self.ranks = ranks