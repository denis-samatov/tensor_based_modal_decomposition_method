import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from typing import Union, Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from TBMD.modules.TensorBasedTubeFiberPivotQRFactorization import TensorTubeQRDecomposition
from TBMD.modules.TensorBasedCompressiveSensing import TensorCompressiveSensing, CompressiveSensingConfig
from TBMD.utils.metrics import compute_metrics
from TBMD.utils.utils import reconstruct_tensor, to_torch_tensor, build_Y_matrices, build_wells_matrix
from TBMD.config import SEED


@dataclass
class ExperimentConfig:
    """A class to configure and manage experiment parameters.

    Attributes:
        solver_method (str): The solver method for compressive sensing.
        seed (int): The random seed for reproducibility.
        device (str): The device to run computations on (e.g., 'cpu', 'cuda').
        max_iter (int): The maximum number of iterations for the compressive sensing solver.
        epsilon (float): The epsilon value for the L1 regularization term in compressive sensing.
        lambd (float): The relaxation lambda parameter for the compressive sensing solver.
        delta_0 (float): The initial delta value for the compressive sensing solver.
        delta_max (float): The maximum delta value for the compressive sensing solver.
        noise_level (float): The level of noise to add to the measurements.
        num_noise_samples (int): The number of noise samples to generate for each measurement.
        noise_threshold (float): The threshold for determining "zero" values when adding noise.
        confidence_level (float): The confidence level for computing confidence intervals.
        convergence_tol (float): The convergence tolerance for the compressive sensing solver.
        subject_axis (bool): Whether to treat the subject axis as a separate dimension.
        valid_mask (Optional[np.ndarray]): A mask of valid locations for sensor placement.
        wells (Optional[Dict[str, List[Tuple[int, int]]]]): A dictionary of well locations for each subject.
        verbose (bool): Whether to print verbose output.
    """
    
    # Core parameters
    solver_method: str = "triangular"
    seed: int = SEED
    device: str = 'cpu'
    
    # Compressive sensing parameters
    max_iter: int = 1000
    epsilon: float = 1e-2
    lambd: float = 0.95
    delta_0: float = 0.1
    delta_max: float = 1.0
    
    # Noise parameters
    noise_level: float = 0.0
    num_noise_samples: int = 0
    noise_threshold: float = 1e-6
    
    # Analysis parameters
    confidence_level: float = 0.95
    convergence_tol: float = 1e-7
    subject_axis: bool = False
    
    # Validation parameters
    valid_mask: Optional[np.ndarray] = None
    wells: Optional[Dict[str, List[Tuple[int, int]]]] = None
    
    # Output parameters
    verbose: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.confidence_level not in [0.90, 0.95, 0.99]:
            print(f"Warning: confidence_level {self.confidence_level} not in [0.90, 0.95, 0.99]. Using 0.95.")
            self.confidence_level = 0.95


class ExperimentRunner:
    """Runs experiments for tensor-based modal decomposition analysis.

    This class provides a standardized way to run experiments for tensor-based
    modal decomposition analysis. It handles QR decomposition, compressive sensing,
    and noise injection, and returns results in a pandas DataFrame for easy
    analysis.
    """
    
    def __init__(self, config: ExperimentConfig = None):
        """Initializes the ExperimentRunner.
        
        Args:
            config (ExperimentConfig, optional): Configuration object. If None,
                uses default configuration.
        """
        self.config = config if config is not None else ExperimentConfig()
        self._setup_confidence_intervals()
    
    def _setup_confidence_intervals(self):
        """Set up z-scores for confidence interval calculations."""
        self.z_scores = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576
        }
    
    def _compute_confidence_intervals(self, means: List[float], stds: List[float], 
                                    num_samples: int) -> Tuple[List[float], List[float]]:
        """Computes confidence intervals for a given set of sample statistics.
        
        Args:
            means (List[float]): The means of the samples.
            stds (List[float]): The standard deviations of the samples.
            num_samples (int): The number of samples.

        Returns:
            Tuple[List[float], List[float]]: The lower and upper bounds for the
            confidence intervals.
        """
        z = self.z_scores.get(self.config.confidence_level, 1.96)
        
        lower, upper = [], []
        for mean_val, std_val in zip(means, stds):
            std_error = std_val / np.sqrt(num_samples)
            margin = z * std_error
            lower.append(mean_val - margin)
            upper.append(mean_val + margin)
        
        return lower, upper
    
    def _perform_qr_decomposition(self, A_tensor: torch.Tensor, 
                                 number_sensors: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs QR decomposition for sensor placement.

        Args:
            A_tensor (torch.Tensor): The basis tensor for decomposition.
            number_sensors (int): The number of sensors to place.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The Q, R, and P
            tensors from the decomposition.
        """
        # Ensure number_sensors is Python int (not numpy int)
        if hasattr(number_sensors, 'item'):
            number_sensors = number_sensors.item()
        number_sensors = int(number_sensors)
        
        qr_decomp = TensorTubeQRDecomposition(
            tensor=A_tensor,
            N=number_sensors,
            rejection_domain=self.config.valid_mask,
            check_orthogonality=False,
            device=self.config.device,
            random_state=self.config.seed
        )
        return qr_decomp.factorize()
    
    def _solve_compressive_sensing(self, A_tensor: torch.Tensor, P: torch.Tensor, 
                                  Y: torch.Tensor) -> torch.Tensor:
        """Solves the compressive sensing problem.

        Args:
            A_tensor (torch.Tensor): The basis tensor.
            P (torch.Tensor): The sensor placement matrix.
            Y (torch.Tensor): The measured data.

        Returns:
            torch.Tensor: The reconstructed sparse coefficients.
        """
        compressive_sensing_config = CompressiveSensingConfig(
            max_iter=self.config.max_iter,
            epsilon_l1=self.config.epsilon,
            relaxation_lambda=self.config.lambd,
            delta_init=self.config.delta_0,
            delta_max=self.config.delta_max,
            convergence_tol=self.config.convergence_tol,
            solver_method=self.config.solver_method,
            device=self.config.device
        )

        cs_solver = TensorCompressiveSensing(
            A=A_tensor,
            P=P,
            Y=Y,
            config=compressive_sensing_config
        )
        return cs_solver.solve()
    
    def _add_noise_to_measurements(self, Y: torch.Tensor) -> torch.Tensor:
        """Adds noise to measurements, skipping zero values.
        
        This is important for reservoir data where 0 values represent the
        absence of fluid/rock and should not be corrupted with noise.

        Args:
            Y (torch.Tensor): The measurement tensor.

        Returns:
            torch.Tensor: The measurement tensor with added noise.
        """
        if self.config.noise_level > 0:
            # Create mask for non-zero values using configurable threshold
            non_zero_mask = torch.abs(Y) > self.config.noise_threshold
            
            # Generate noise with same shape as Y
            noise = torch.randn_like(Y) * self.config.noise_level * torch.max(torch.abs(Y))
            
            # Apply noise only to non-zero values
            noisy_Y = Y.clone()
            noisy_Y[non_zero_mask] = Y[non_zero_mask] + noise[non_zero_mask]
            
            return noisy_Y
        return Y
    
    def run_full_dataset_experiments(self, 
                                   A_tensor: Union[np.ndarray, torch.Tensor],
                                   test_tensors: Dict[str, Union[np.ndarray, torch.Tensor]],
                                   sensor_values: List[int]) -> pd.DataFrame:
        """Runs experiments across the full dataset with all subjects and slices.
        
        Args:
            A_tensor (Union[np.ndarray, torch.Tensor]): Basis tensor for
                decomposition.
            test_tensors (Dict[str, Union[np.ndarray, torch.Tensor]]): Test
                data per subject.
            sensor_values (List[int]): Range of sensor counts to evaluate.
            
        Returns:
            pd.DataFrame: A table of results with metrics and confidence intervals.
        """
        A_tensor = to_torch_tensor(A_tensor, device=self.config.device, dtype=torch.float32)
        
        # Convert all test tensors once at the beginning
        test_tensors_torch = {
            subject: to_torch_tensor(tensor, device=self.config.device, dtype=torch.float32)
            for subject, tensor in test_tensors.items()
        }
        
        results = []
        num_total_samples = 1 + self.config.num_noise_samples
        
        for number_sensors in tqdm(sensor_values, desc="Full dataset experiments"):
            P, Q, R = self._perform_qr_decomposition(A_tensor, number_sensors)
            Y_mats = build_Y_matrices(test_tensors, P, device=self.config.device)
            
            all_errors, all_ssims, all_psnrs = [], [], []
            
            for subject, Y_subject in Y_mats.items():
                test_data = test_tensors_torch[subject]
                num_slices = test_data.shape[-1]
                
                for slice_idx in range(num_slices):
                    X_slice = test_data[..., slice_idx]
                    Y_slice = Y_subject[..., slice_idx]
                    
                    # Baseline (no noise)
                    x_hat = self._solve_compressive_sensing(A_tensor, P, Y_slice)
                    X_reconstructed = reconstruct_tensor(A_tensor=A_tensor, x_hat=x_hat)
                    error_val, _, ssim_val, psnr_val = compute_metrics(X_slice, X_reconstructed)
                    
                    all_errors.append(error_val)
                    all_ssims.append(ssim_val)
                    all_psnrs.append(psnr_val)
                    
                    # Noise samples
                    for _ in range(self.config.num_noise_samples):
                        noisy_Y = self._add_noise_to_measurements(Y_slice)
                        x_hat = self._solve_compressive_sensing(A_tensor, P, noisy_Y)
                        X_reconstructed = reconstruct_tensor(A_tensor=A_tensor, x_hat=x_hat)
                        error_val, _, ssim_val, psnr_val = compute_metrics(X_slice, X_reconstructed)
                        
                        all_errors.append(error_val)
                        all_ssims.append(ssim_val)
                        all_psnrs.append(psnr_val)
            
            # Calculate statistics
            error_tensor = torch.tensor(all_errors)
            ssim_tensor = torch.tensor(all_ssims)
            psnr_tensor = torch.tensor(all_psnrs)
            
            error_mean, error_std = float(torch.mean(error_tensor)), float(torch.std(error_tensor))
            ssim_mean, ssim_std = float(torch.mean(ssim_tensor)), float(torch.std(ssim_tensor))
            psnr_mean, psnr_std = float(torch.mean(psnr_tensor)), float(torch.std(psnr_tensor))
            
            # Confidence intervals
            error_ci_lower, error_ci_upper = self._compute_confidence_intervals([error_mean], [error_std], len(all_errors))
            ssim_ci_lower, ssim_ci_upper = self._compute_confidence_intervals([ssim_mean], [ssim_std], len(all_ssims))
            psnr_ci_lower, psnr_ci_upper = self._compute_confidence_intervals([psnr_mean], [psnr_std], len(all_psnrs))
            
            results.append({
                'sensors': number_sensors,
                'error_mean': error_mean,
                'error_std': error_std,
                'ssim_mean': ssim_mean,
                'ssim_std': ssim_std,
                'psnr_mean': psnr_mean,
                'psnr_std': psnr_std,
                'error_ci_lower': error_ci_lower[0],
                'error_ci_upper': error_ci_upper[0],
                'ssim_ci_lower': ssim_ci_lower[0],
                'ssim_ci_upper': ssim_ci_upper[0],
                'psnr_ci_lower': psnr_ci_lower[0],
                'psnr_ci_upper': psnr_ci_upper[0],
                'num_samples': len(all_errors)
            })
        
        return pd.DataFrame(results)
    
    def run_single_slice_experiments(self,
                                   A_tensor: Union[np.ndarray, torch.Tensor],
                                   test_tensors: Dict[str, Union[np.ndarray, torch.Tensor]],
                                   subject_name: str,
                                   slice_idx: int,
                                   sensor_values: List[int]) -> pd.DataFrame:
        """Runs experiments for a specific slice of a specific subject.
        
        Args:
            A_tensor (Union[np.ndarray, torch.Tensor]): Basis tensor for
                decomposition.
            test_tensors (Dict[str, Union[np.ndarray, torch.Tensor]]): Test
                data per subject.
            subject_name (str): Name of the subject to analyze.
            slice_idx (int): Index of the slice to analyze.
            sensor_values (List[int]): Range of sensor counts to evaluate.
            
        Returns:
            pd.DataFrame: A table of results with statistics and confidence
            intervals.
        """
        A_tensor = to_torch_tensor(A_tensor, device=self.config.device, dtype=torch.float32)
        test_data = to_torch_tensor(test_tensors[subject_name], device=self.config.device, dtype=torch.float32)
        X_slice = test_data[..., slice_idx]
        
        results = []
        num_total_samples = 1 + self.config.num_noise_samples
        
        for number_sensors in tqdm(sensor_values, desc=f"Single slice experiments (slice {slice_idx})"):
            P, Q, R = self._perform_qr_decomposition(A_tensor, number_sensors)
            Y_mats = build_Y_matrices(test_tensors, P, device=self.config.device)
            Y_subject = Y_mats[subject_name]
            Y_slice = Y_subject[..., slice_idx]
            
            slice_errors, slice_ssims, slice_psnrs = [], [], []
            
            # Baseline
            x_hat = self._solve_compressive_sensing(A_tensor, P, Y_slice)
            X_reconstructed = reconstruct_tensor(A_tensor=A_tensor, x_hat=x_hat)
            err_base, _, ssim_base, psnr_base = compute_metrics(X_slice, X_reconstructed)
            
            slice_errors.append(err_base)
            slice_ssims.append(ssim_base)
            slice_psnrs.append(psnr_base)
            
            # Noise samples
            for _ in range(self.config.num_noise_samples):
                noisy_Y_slice = self._add_noise_to_measurements(Y_slice)
                x_hat = self._solve_compressive_sensing(A_tensor, P, noisy_Y_slice)
                X_reconstructed = reconstruct_tensor(A_tensor=A_tensor, x_hat=x_hat)
                err_noisy, _, ssim_noisy, psnr_noisy = compute_metrics(X_slice, X_reconstructed)
                
                slice_errors.append(err_noisy)
                slice_ssims.append(ssim_noisy)
                slice_psnrs.append(psnr_noisy)
            
            # Calculate statistics
            error_mean, error_std = float(torch.mean(torch.tensor(slice_errors))), float(torch.std(torch.tensor(slice_errors)))
            ssim_mean, ssim_std = float(torch.mean(torch.tensor(slice_ssims))), float(torch.std(torch.tensor(slice_ssims)))
            psnr_mean, psnr_std = float(torch.mean(torch.tensor(slice_psnrs))), float(torch.std(torch.tensor(slice_psnrs)))
            
            # Confidence intervals
            error_ci_lower, error_ci_upper = self._compute_confidence_intervals([error_mean], [error_std], len(slice_errors))
            ssim_ci_lower, ssim_ci_upper = self._compute_confidence_intervals([ssim_mean], [ssim_std], len(slice_ssims))
            psnr_ci_lower, psnr_ci_upper = self._compute_confidence_intervals([psnr_mean], [psnr_std], len(slice_psnrs))
            
            results.append({
                'sensors': number_sensors,
                'subject': subject_name,
                'slice_idx': slice_idx,
                'error_mean': error_mean,
                'error_std': error_std,
                'ssim_mean': ssim_mean,
                'ssim_std': ssim_std,
                'psnr_mean': psnr_mean,
                'psnr_std': psnr_std,
                'error_ci_lower': error_ci_lower[0],
                'error_ci_upper': error_ci_upper[0],
                'ssim_ci_lower': ssim_ci_lower[0],
                'ssim_ci_upper': ssim_ci_upper[0],
                'psnr_ci_lower': psnr_ci_lower[0],
                'psnr_ci_upper': psnr_ci_upper[0],
                'num_samples': len(slice_errors)
            })
        
        return pd.DataFrame(results)
    
    def run_single_slice_wells_experiments(self,
                                         A_tensor: Union[np.ndarray, torch.Tensor],
                                         test_tensors: Dict[str, Union[np.ndarray, torch.Tensor]],
                                         subject_name: str,
                                         slice_idx: int,
                                         sensor_values: List[int]) -> pd.DataFrame:
        """Runs wells experiments for a specific slice and subject.
        
        Args:
            A_tensor (Union[np.ndarray, torch.Tensor]): Basis tensor for
                decomposition.
            test_tensors (Dict[str, Union[np.ndarray, torch.Tensor]]): Test
                data per subject.
            subject_name (str): Name of the subject to analyze.
            slice_idx (int): Index of the slice to analyze.
            sensor_values (List[int]): Range of sensor counts to evaluate.
            
        Returns:
            pd.DataFrame: A table of results with statistics and confidence
            intervals.
        """
        if self.config.wells is None:
            raise ValueError("Wells configuration must be provided for wells experiments")
        
        A_tensor = to_torch_tensor(A_tensor, device=self.config.device, dtype=torch.float32)
        test_data = to_torch_tensor(test_tensors[subject_name], device=self.config.device, dtype=torch.float32)
        X_slice = test_data[..., slice_idx]
        
        # Получить wells только для subject_name
        wells_list = self.config.wells.get(subject_name, [])
        # Убрать дубликаты и невалидные координаты
        valid_wells = []
        seen = set()
        for i, j in wells_list:
            if (i, j) not in seen and 0 <= i < A_tensor.shape[0] and 0 <= j < A_tensor.shape[1]:
                valid_wells.append([i, j])
                seen.add((i, j))
        
        results = []
        num_total_samples = 1 + self.config.num_noise_samples
        
        for N in tqdm(sensor_values, desc=f"Single slice wells experiments (slice {slice_idx})"):
            selected_wells = valid_wells[:min(N, len(valid_wells))]
            wells_dict = {subject_name: selected_wells}
            P = build_wells_matrix(wells_dict, A_tensor.shape, device=self.config.device)
            
            # Build Y matrix only for the specific subject
            Y_mats = build_Y_matrices({subject_name: test_tensors[subject_name]}, P, device=self.config.device)
            Y_subject = Y_mats[subject_name]
            Y_slice = Y_subject[..., slice_idx]
            
            slice_errors, slice_ssims, slice_psnrs = [], [], []
            
            # Baseline (no noise)
            x_hat = self._solve_compressive_sensing(A_tensor, P, Y_slice)
            X_reconstructed = reconstruct_tensor(A_tensor=A_tensor, x_hat=x_hat)
            err_base, _, ssim_base, psnr_base = compute_metrics(X_slice, X_reconstructed)
            
            slice_errors.append(err_base)
            slice_ssims.append(ssim_base)
            slice_psnrs.append(psnr_base)
            
            # Noise samples
            for _ in range(self.config.num_noise_samples):
                noisy_Y_slice = self._add_noise_to_measurements(Y_slice)
                x_hat = self._solve_compressive_sensing(A_tensor, P, noisy_Y_slice)
                X_reconstructed = reconstruct_tensor(A_tensor=A_tensor, x_hat=x_hat)
                err_noisy, _, ssim_noisy, psnr_noisy = compute_metrics(X_slice, X_reconstructed)
                
                slice_errors.append(err_noisy)
                slice_ssims.append(ssim_noisy)
                slice_psnrs.append(psnr_noisy)
            
            # Calculate statistics
            error_mean, error_std = float(torch.mean(torch.tensor(slice_errors))), float(torch.std(torch.tensor(slice_errors)))
            ssim_mean, ssim_std = float(torch.mean(torch.tensor(slice_ssims))), float(torch.std(torch.tensor(slice_ssims)))
            psnr_mean, psnr_std = float(torch.mean(torch.tensor(slice_psnrs))), float(torch.std(torch.tensor(slice_psnrs)))
            
            # Confidence intervals
            error_ci_lower, error_ci_upper = self._compute_confidence_intervals([error_mean], [error_std], len(slice_errors))
            ssim_ci_lower, ssim_ci_upper = self._compute_confidence_intervals([ssim_mean], [ssim_std], len(slice_ssims))
            psnr_ci_lower, psnr_ci_upper = self._compute_confidence_intervals([psnr_mean], [psnr_std], len(slice_psnrs))
            
            results.append({
                'sensors': N,
                'subject': subject_name,
                'slice_idx': slice_idx,
                'error_mean': error_mean,
                'error_std': error_std,
                'ssim_mean': ssim_mean,
                'ssim_std': ssim_std,
                'psnr_mean': psnr_mean,
                'psnr_std': psnr_std,
                'error_ci_lower': error_ci_lower[0],
                'error_ci_upper': error_ci_upper[0],
                'ssim_ci_lower': ssim_ci_lower[0],
                'ssim_ci_upper': ssim_ci_upper[0],
                'psnr_ci_lower': psnr_ci_lower[0],
                'psnr_ci_upper': psnr_ci_upper[0],
                'num_samples': len(slice_errors)
            })
        
        return pd.DataFrame(results)

    def run_full_dataset_wells_experiments(self,
                                         A_tensor: Union[np.ndarray, torch.Tensor],
                                         test_tensors: Dict[str, Union[np.ndarray, torch.Tensor]],
                                         sensor_values: List[int]) -> pd.DataFrame:
        """Runs wells experiments across the full dataset.
        
        Args:
            A_tensor (Union[np.ndarray, torch.Tensor]): Basis tensor for
                decomposition.
            test_tensors (Dict[str, Union[np.ndarray, torch.Tensor]]): Test
                data per subject.
            sensor_values (List[int]): Range of sensor counts to evaluate.
            
        Returns:
            pd.DataFrame: A table of results with metrics and confidence intervals.
        """
        from TBMD.utils.utils import build_wells_matrix
        if self.config.wells is None:
            raise ValueError("Wells configuration must be provided for wells experiments")
        
        A_tensor = to_torch_tensor(A_tensor, device=self.config.device, dtype=torch.float32)
        test_tensors_torch = {
            subject: to_torch_tensor(tensor, device=self.config.device, dtype=torch.float32)
            for subject, tensor in test_tensors.items()
        }
        
        # Подготовить валидные wells для каждого subject
        wells_dict_valid = {}
        for subject, wells_list in self.config.wells.items():
            valid_wells = []
            seen = set()
            for i, j in wells_list:
                if (i, j) not in seen and 0 <= i < A_tensor.shape[0] and 0 <= j < A_tensor.shape[1]:
                    valid_wells.append([i, j])
                    seen.add((i, j))
            wells_dict_valid[subject] = valid_wells
        
        results = []
        num_total_samples = 1 + self.config.num_noise_samples
        
        for N in tqdm(sensor_values, desc="Full dataset wells experiments"):
            # Для каждого subject взять только N первых wells
            wells_dict_N = {subject: wells[:min(N, len(wells))] for subject, wells in wells_dict_valid.items()}
            P = build_wells_matrix(wells_dict_N, A_tensor.shape, device=self.config.device)
            
            # Build Y matrices for all subjects
            Y_mats = build_Y_matrices(test_tensors, P, device=self.config.device)
            
            all_errors, all_mses, all_ssims, all_psnrs = [], [], [], []
            
            for subject, Y_subject in Y_mats.items():
                test_data = test_tensors_torch[subject]
                num_slices = test_data.shape[-1]
                
                for slice_idx in range(num_slices):
                    X_slice = test_data[..., slice_idx]
                    Y_slice = Y_subject[..., slice_idx]
                    
                    # Baseline (no noise)
                    x_hat = self._solve_compressive_sensing(A_tensor, P, Y_slice)
                    X_reconstructed = reconstruct_tensor(A_tensor=A_tensor, x_hat=x_hat)
                    error_val, mse_val, ssim_val, psnr_val = compute_metrics(X_slice, X_reconstructed)
                    
                    all_errors.append(error_val)
                    all_mses.append(mse_val)
                    all_ssims.append(ssim_val)
                    all_psnrs.append(psnr_val)
                    
                    # Noise samples
                    for _ in range(self.config.num_noise_samples):
                        noisy_Y = self._add_noise_to_measurements(Y_slice)
                        x_hat = self._solve_compressive_sensing(A_tensor, P, noisy_Y)
                        X_reconstructed = reconstruct_tensor(A_tensor=A_tensor, x_hat=x_hat)
                        error_val, mse_val, ssim_val, psnr_val = compute_metrics(X_slice, X_reconstructed)
                        
                        all_errors.append(error_val)
                        all_mses.append(mse_val)
                        all_ssims.append(ssim_val)
                        all_psnrs.append(psnr_val)
            
            # Calculate statistics
            error_tensor = torch.tensor(all_errors)
            mse_tensor = torch.tensor(all_mses)
            ssim_tensor = torch.tensor(all_ssims)
            psnr_tensor = torch.tensor(all_psnrs)
            
            error_mean, error_std = float(torch.mean(error_tensor)), float(torch.std(error_tensor))
            mse_mean, mse_std = float(torch.mean(mse_tensor)), float(torch.std(mse_tensor))
            ssim_mean, ssim_std = float(torch.mean(ssim_tensor)), float(torch.std(ssim_tensor))
            psnr_mean, psnr_std = float(torch.mean(psnr_tensor)), float(torch.std(psnr_tensor))
            
            # Confidence intervals
            error_ci_lower, error_ci_upper = self._compute_confidence_intervals([error_mean], [error_std], len(all_errors))
            mse_ci_lower, mse_ci_upper = self._compute_confidence_intervals([mse_mean], [mse_std], len(all_mses))
            ssim_ci_lower, ssim_ci_upper = self._compute_confidence_intervals([ssim_mean], [ssim_std], len(all_ssims))
            psnr_ci_lower, psnr_ci_upper = self._compute_confidence_intervals([psnr_mean], [psnr_std], len(all_psnrs))
            
            results.append({
                'sensors': N,
                'error_mean': error_mean,
                'error_std': error_std,
                'mse_mean': mse_mean,
                'mse_std': mse_std,
                'ssim_mean': ssim_mean,
                'ssim_std': ssim_std,
                'psnr_mean': psnr_mean,
                'psnr_std': psnr_std,
                'error_ci_lower': error_ci_lower[0],
                'error_ci_upper': error_ci_upper[0],
                'mse_ci_lower': mse_ci_lower[0],
                'mse_ci_upper': mse_ci_upper[0],
                'ssim_ci_lower': ssim_ci_lower[0],
                'ssim_ci_upper': ssim_ci_upper[0],
                'psnr_ci_lower': psnr_ci_lower[0],
                'psnr_ci_upper': psnr_ci_upper[0],
                'num_samples': len(all_errors)
            })
        
        return pd.DataFrame(results)


# Utility functions
def ensure_sensor_values_are_int(sensor_values: List) -> List[int]:
    """Ensures that sensor values are Python integers.

    This function converts numpy integers or other numeric types to Python int,
    which is useful for avoiding type validation errors.

    Args:
        sensor_values (List): A list of sensor counts, which may contain
            numpy integers.

    Returns:
        List[int]: A list of Python integers.
    """
    result = []
    for val in sensor_values:
        if hasattr(val, 'item'):  # numpy scalar
            result.append(val.item())
        else:
            result.append(int(val))
    return result


# Backward compatibility functions (deprecated)
def compute_confidence_intervals(means, stds, num_samples, confidence_level=0.95):
    """Compute confidence intervals.

    .. deprecated:: 0.1.0
       Use :class:`ExperimentRunner` instead.

    Parameters
    ----------
    means : list
        The means of the samples.
    stds : list
        The standard deviations of the samples.
    num_samples : int
        The number of samples.
    confidence_level : float, optional
        The confidence level for the interval, by default 0.95.

    Returns
    -------
    tuple
        The lower and upper bounds of the confidence intervals.
    """
    print("Warning: This function is deprecated. Use ExperimentRunner class instead.")
    config = ExperimentConfig(confidence_level=confidence_level)
    runner = ExperimentRunner(config)
    return runner._compute_confidence_intervals(means, stds, num_samples)


def run_experiments(*args, **kwargs):
    """Run experiments.

    .. deprecated:: 0.1.0
       Use :meth:`ExperimentRunner.run_full_dataset_experiments` instead.
    """
    print("Warning: This function is deprecated. Use ExperimentRunner.run_full_dataset_experiments() instead.")
    config = ExperimentConfig()
    runner = ExperimentRunner(config)
    # This would need more complex mapping - recommend using the class directly
    raise NotImplementedError("Please use ExperimentRunner class directly")


def run_experiments_single_slice(*args, **kwargs):
    """Run experiments for a single slice.

    .. deprecated:: 0.1.0
       Use :meth:`ExperimentRunner.run_single_slice_experiments` instead.
    """
    print("Warning: This function is deprecated. Use ExperimentRunner.run_single_slice_experiments() instead.")
    raise NotImplementedError("Please use ExperimentRunner class directly")


def run_experiments_df(*args, **kwargs):
    """Run experiments and return a DataFrame.

    .. deprecated:: 0.1.0
       Use :meth:`ExperimentRunner.run_experiments` instead.
    """
    print("Warning: This function is deprecated. Use ExperimentRunner.run_experiments() instead.")
    raise NotImplementedError("Please use ExperimentRunner class directly")


def run_experiments_wells_df(*args, **kwargs):
    """Run experiments with wells and return a DataFrame.

    .. deprecated:: 0.1.0
       Use :meth:`ExperimentRunner.run_wells_experiments` instead.
    """
    print("Warning: This function is deprecated. Use ExperimentRunner.run_wells_experiments() instead.")
    raise NotImplementedError("Please use ExperimentRunner class directly")


def plot_analytics(df: pd.DataFrame, 
                  metrics: List[str] = ['error', 'ssim', 'psnr'],
                  plot_type: str = "individual",
                  title_prefix: str = "Experiment Results",
                  figsize: Tuple[int, int] = (8, 5),
                  save_path: Optional[str] = None,
                  show_plots: bool = True) -> None:
    """Plots analytics results from a DataFrame.

    This function provides comprehensive visualization options and replicates the
    functionality of the original `plot_analytics` function from `plots.py`, but
    is adapted for a DataFrame input format.

    Args:
        df (pd.DataFrame): The results DataFrame from `ExperimentRunner` methods.
        metrics (List[str]): A list of metrics to plot. Defaults to `['error', 'ssim', 'psnr']`.
        plot_type (str): The type of plot to generate. Can be one of 'individual',
            'combined', 'normalized', or 'all'. Defaults to 'individual'.
        title_prefix (str): A prefix for the plot titles. Defaults to "Experiment Results".
        figsize (Tuple[int, int]): The figure size for individual plots. Defaults to (8, 5).
        save_path (Optional[str]): The base path to save plots. Defaults to None.
        show_plots (bool): Whether to display the plots. Defaults to True.
    """
    import numpy as np
    
    # Determine data format (with or without confidence intervals)
    has_ci = any(f'{metric}_ci_lower' in df.columns for metric in metrics)
    has_mean_std = any(f'{metric}_mean' in df.columns for metric in metrics)
    
    # Extract data for plotting
    sensor_values = df['sensors'].values
    plot_data = {}
    
    for metric in metrics:
        if has_mean_std and f'{metric}_mean' in df.columns:
            # Data with confidence intervals
            plot_data[metric] = {
                'means': df[f'{metric}_mean'].values,
                'lower': df[f'{metric}_ci_lower'].values if f'{metric}_ci_lower' in df.columns else df[f'{metric}_mean'].values - df[f'{metric}_std'].values,
                'upper': df[f'{metric}_ci_upper'].values if f'{metric}_ci_upper' in df.columns else df[f'{metric}_mean'].values + df[f'{metric}_std'].values,
                'std': df[f'{metric}_std'].values if f'{metric}_std' in df.columns else None
            }
        elif metric in df.columns:
            # Simple data without confidence intervals
            plot_data[metric] = {
                'means': df[metric].values,
                'lower': df[metric].values,  # No CI, use same values
                'upper': df[metric].values,
                'std': None
            }
        else:
            print(f"Warning: Metric '{metric}' not found in DataFrame")
            continue
    
    if not plot_data:
        print("No valid metrics found in DataFrame")
        return
    
    # Color mapping
    colors = {'error': 'blue', 'ssim': 'green', 'psnr': 'red', 'mse': 'orange'}
    
    def save_plot(suffix=""):
        if save_path:
            path = f"{save_path}_{suffix}.png" if suffix else f"{save_path}.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
    
    # Plot 1: Individual plots for each metric
    if plot_type in ['individual', 'all']:
        for metric in plot_data.keys():
            data = plot_data[metric]
            color = colors.get(metric, 'black')
            
            plt.figure(figsize=figsize)
            plt.plot(sensor_values, data['means'], color=color, label=f'Mean {metric.upper()}')
            plt.scatter(sensor_values, data['means'], color=color, marker='o', s=30)
            
            # Add confidence intervals or std deviation
            if not np.array_equal(data['lower'], data['means']) or not np.array_equal(data['upper'], data['means']):
                plt.fill_between(sensor_values, data['lower'], data['upper'], 
                               color=color, alpha=0.2, label='95% CI')
            
            plt.title(f'{metric.upper()} vs. Sensors')
            plt.xlabel('Number of Sensors (N)')
            
            if metric == 'error':
                plt.ylabel('Error')
            elif metric == 'ssim':
                plt.ylabel('SSIM')
            elif metric == 'psnr':
                plt.ylabel('PSNR (dB)')
            else:
                plt.ylabel(metric.upper())
            
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            if save_path:
                save_plot(f"{metric}")
            
            if show_plots:
                plt.show()
            else:
                plt.close()
    
    # Plot 2: Combined Normalized Plot (Error inverted and SSIM)
    if plot_type in ['normalized', 'all'] and 'error' in plot_data and 'ssim' in plot_data:
        plt.figure(figsize=(10, 5))
        
        error_data = plot_data['error']
        ssim_data = plot_data['ssim']
        
        # Convert to numpy arrays for calculations
        error_means_np = np.array(error_data['means'])
        error_lower_np = np.array(error_data['lower'])
        error_upper_np = np.array(error_data['upper'])
        ssim_means_np = np.array(ssim_data['means'])
        ssim_lower_np = np.array(ssim_data['lower'])
        ssim_upper_np = np.array(ssim_data['upper'])
        
        # Determine global min/max for normalization
        error_min_val = np.min(error_lower_np)
        error_max_val = np.max(error_upper_np)
        ssim_min_val = np.min(ssim_lower_np)
        ssim_max_val = np.max(ssim_upper_np)
        
        error_range = error_max_val - error_min_val if error_max_val > error_min_val else 1.0
        ssim_range = ssim_max_val - ssim_min_val if ssim_max_val > ssim_min_val else 1.0
        
        # Normalize and invert error (so higher is better)
        norm_error_means = (error_means_np - error_min_val) / error_range
        # For inverted error, swap the CI bounds
        norm_error_lower_ci = (error_upper_np - error_min_val) / error_range
        norm_error_upper_ci = (error_lower_np - error_min_val) / error_range
        
        # Normalize SSIM (higher is already better)
        norm_ssim_means = (ssim_means_np - ssim_min_val) / ssim_range
        norm_ssim_lower_ci = (ssim_lower_np - ssim_min_val) / ssim_range
        norm_ssim_upper_ci = (ssim_upper_np - ssim_min_val) / ssim_range
        
        # Plot normalized metrics
        plt.plot(sensor_values, norm_error_means, color='blue', label='Error')
        plt.scatter(sensor_values, norm_error_means, color='blue', marker='o', s=30)
        plt.fill_between(sensor_values, 
                        np.minimum(norm_error_lower_ci, norm_error_upper_ci),
                        np.maximum(norm_error_lower_ci, norm_error_upper_ci),
                        color='blue', alpha=0.2)
        
        plt.plot(sensor_values, norm_ssim_means, color='green', label='SSIM')
        plt.scatter(sensor_values, norm_ssim_means, color='green', marker='o', s=30)
        plt.fill_between(sensor_values, norm_ssim_lower_ci, norm_ssim_upper_ci, 
                        color='green', alpha=0.2)
        
        plt.xlabel('Number of Sensors (N)')
        plt.ylabel('Normalized Quality Metrics')
        plt.title('Performance Metrics vs. Number of Sensors')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1)
        plt.tight_layout()
        
        if save_path:
            save_plot("normalized")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # Plot 3: Combined Non-Normalized Plot (All metrics, only if explicitly requested)
    if plot_type == 'combined':
        plt.figure(figsize=(12, 6))
        
        for i, (metric, data) in enumerate(plot_data.items()):
            color = colors.get(metric, f'C{i}')
            marker = ['o', 's', '^', 'D'][i % 4]  # Different markers
            
            plt.plot(sensor_values, data['means'], color=color, label=f'Mean {metric.upper()}')
            plt.scatter(sensor_values, data['means'], color=color, marker=marker, s=30)
            
            # Add confidence intervals
            if not np.array_equal(data['lower'], data['means']) or not np.array_equal(data['upper'], data['means']):
                plt.fill_between(sensor_values, data['lower'], data['upper'],
                               color=color, alpha=0.2, label=f'{metric.upper()} 95% CI')
        
        plt.title('Combined Metrics vs. Sensors')
        plt.xlabel('Number of Sensors (N)')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            save_plot("combined")
        
        if show_plots:
            plt.show()
        else:
            plt.close()


def plot_analytics_legacy(sensor_values, error_means, error_lower, error_upper,
                         ssim_means, ssim_lower, ssim_upper,
                         psnr_means, psnr_lower, psnr_upper,
                         save_path: Optional[str] = None):
    """Plots analytics using a legacy function for backward compatibility.

    This function is the original `plot_analytics` from `plots.py`, adapted for
    use with separate arrays instead of a DataFrame.

    Args:
        sensor_values (array_like): The values for the number of sensors.
        error_means (array_like): The mean error values.
        error_lower (array_like): The lower bound of the error confidence interval.
        error_upper (array_like): The upper bound of the error confidence interval.
        ssim_means (array_like): The mean SSIM values.
        ssim_lower (array_like): The lower bound of the SSIM confidence interval.
        ssim_upper (array_like): The upper bound of the SSIM confidence interval.
        psnr_means (array_like): The mean PSNR values.
        psnr_lower (array_like): The lower bound of the PSNR confidence interval.
        psnr_upper (array_like): The upper bound of the PSNR confidence interval.
        save_path (str, optional): The path to save the plot to. Defaults to None.
    """
    print("Warning: Using legacy plot function. Consider using plot_analytics with DataFrame.")
    
    # Create a DataFrame and use the new function
    df = pd.DataFrame({
        'sensors': sensor_values,
        'error_mean': error_means,
        'error_ci_lower': error_lower,
        'error_ci_upper': error_upper,
        'ssim_mean': ssim_means,
        'ssim_ci_lower': ssim_lower,
        'ssim_ci_upper': ssim_upper,
        'psnr_mean': psnr_means,
        'psnr_ci_lower': psnr_lower,
        'psnr_ci_upper': psnr_upper
    })
    
    plot_analytics(df, metrics=['error', 'ssim', 'psnr'], plot_type='all', save_path=save_path)