import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from typing import Union, Dict, List, Tuple, Optional

from TBMD.core.sensor_placement.tensor_qr_factorization import TensorTubeQRDecomposition
from TBMD.core.reconstruction.tensor_compressive_sensing import (
    TensorCompressiveSensing,
    CompressiveSensingMetrics
)
from TBMD.config import CompressiveSensingConfig, ExtensionCompressiveSensingConfig
from TBMD.core.metrics.metrics import compute_metrics
from TBMD.core.utils.misc import reconstruct_tensor, to_torch_tensor, build_Y_matrices, build_wells_matrix
from TBMD.config.experiments import ExperimentConfig


class ExperimentRunner:
    """
    Unified experiment runner for tensor-based modal decomposition analysis.
    
    All methods return pandas DataFrames for consistent data handling and analysis.
    """
    
    def __init__(self, config: ExperimentConfig = None):
        """
        Initialize the experiment runner with configuration.
        
        Parameters
        ----------
        config : ExperimentConfig, optional
            Configuration object. If None, uses default configuration.
        """
        self.config = config if config is not None else ExperimentConfig()
        self._setup_confidence_intervals()
    
    def _setup_confidence_intervals(self):
        """Setup z-scores for confidence interval calculations."""
        self.z_scores = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576
        }
    
    def _compute_confidence_intervals(self, means: List[float], stds: List[float], 
                                    num_samples: int) -> Tuple[List[float], List[float]]:
        """
        Compute confidence intervals given means, standard deviations, and number of samples.
        
        Returns
        -------
        Tuple[List[float], List[float]]
            Lower and upper bounds for confidence intervals.
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
        """Perform QR decomposition for sensor placement."""
        # Ensure number_sensors is Python int (not numpy int)
        if hasattr(number_sensors, 'item'):
            number_sensors = number_sensors.item()
        number_sensors = int(number_sensors)
        
        qr_decomp = TensorTubeQRDecomposition(
            tensor=A_tensor,
            N=number_sensors,
            rejection_domain=self.config.valid_mask,
            check_orthogonality=False,
            uniform_distribution=False,
            device=self.config.device,
            random_state=self.config.seed
        )
        return qr_decomp.factorize()
    
    def _solve_compressive_sensing(self, A_tensor: torch.Tensor, P: torch.Tensor, 
                                  Y: torch.Tensor) -> Tuple[torch.Tensor, CompressiveSensingMetrics]:
        """Solve compressive sensing problem.
        
        Returns
        -------
        Tuple[torch.Tensor, CompressiveSensingMetrics]
            Solution vector x_hat and metrics from the solver.
        """
        core_cfg = CompressiveSensingConfig(
            max_iter=self.config.max_iter,
            tol=self.config.convergence_tol,
            epsilon_l1=self.config.epsilon,
            delta_init=self.config.delta_0,
            delta_max=self.config.delta_max,
            relax_lambda=self.config.lambd,
            device=self.config.device
        )
        ext_cfg = ExtensionCompressiveSensingConfig(
            solver="cholesky",
            delta_policy="boyd",
            stop_policy="residual",
            collect_history=False
        )
        cs_solver = TensorCompressiveSensing(A_tensor, P, Y, core_cfg, ext_cfg)
        return cs_solver.solve()
    
    def _add_noise_to_measurements(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Add noise to measurements only for non-zero values.
        
        Important for reservoir data where 0 values represent 
        absence of fluid/rock and should not be corrupted with noise.
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
        """
        Run experiments across full dataset with all subjects and slices.
        
        Parameters
        ----------
        A_tensor : ndarray | torch.Tensor
            Basis tensor for decomposition.
        test_tensors : Dict[str, ndarray | torch.Tensor]
            Test data per subject.
        sensor_values : List[int]
            Range of sensor counts to evaluate.
            
        Returns
        -------
        pd.DataFrame
            Results with columns: ['sensors', 'error_mean', 'error_std', 'ssim_mean', 
            'ssim_std', 'psnr_mean', 'psnr_std', 'error_ci_lower', 'error_ci_upper',
            'ssim_ci_lower', 'ssim_ci_upper', 'psnr_ci_lower', 'psnr_ci_upper', 'num_samples']
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
                    x_hat, _ = self._solve_compressive_sensing(A_tensor, P, Y_slice)
                    X_reconstructed = reconstruct_tensor(A_tensor=A_tensor, x_hat=x_hat)
                    error_val, _, ssim_val, psnr_val = compute_metrics(X_slice, X_reconstructed)
                    
                    all_errors.append(error_val)
                    all_ssims.append(ssim_val)
                    all_psnrs.append(psnr_val)
                    
                    # Noise samples
                    for _ in range(self.config.num_noise_samples):
                        noisy_Y = self._add_noise_to_measurements(Y_slice)
                        x_hat, _ = self._solve_compressive_sensing(A_tensor, P, noisy_Y)
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
        """
        Run experiments for a specific slice of a specific subject.
        
        Parameters
        ----------
        A_tensor : ndarray | torch.Tensor
            Basis tensor for decomposition.
        test_tensors : Dict[str, ndarray | torch.Tensor]
            Test data per subject.
        subject_name : str
            Name of the subject to analyze.
        slice_idx : int
            Index of the slice to analyze.
        sensor_values : List[int]
            Range of sensor counts to evaluate.
            
        Returns
        -------
        pd.DataFrame
            Results with statistics and confidence intervals.
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
            x_hat, _ = self._solve_compressive_sensing(A_tensor, P, Y_slice)
            X_reconstructed = reconstruct_tensor(A_tensor=A_tensor, x_hat=x_hat)
            err_base, _, ssim_base, psnr_base = compute_metrics(X_slice, X_reconstructed)
            
            slice_errors.append(err_base)
            slice_ssims.append(ssim_base)
            slice_psnrs.append(psnr_base)
            
            # Noise samples
            for _ in range(self.config.num_noise_samples):
                noisy_Y_slice = self._add_noise_to_measurements(Y_slice)
                x_hat, _ = self._solve_compressive_sensing(A_tensor, P, noisy_Y_slice)
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
        """
        Run wells experiments for a specific slice of a specific subject with statistical analysis.
        
        Parameters
        ----------
        A_tensor : ndarray | torch.Tensor
            Basis tensor for decomposition.
        test_tensors : Dict[str, ndarray | torch.Tensor]
            Test data per subject.
        subject_name : str
            Name of the subject to analyze.
        slice_idx : int
            Index of the slice to analyze.
        sensor_values : List[int]
            Range of sensor counts to evaluate.
            
        Returns
        -------
        pd.DataFrame
            Results with statistics and confidence intervals.
            Columns: ['sensors', 'subject', 'slice_idx', 'error_mean', 'error_std', 
                     'ssim_mean', 'ssim_std', 'psnr_mean', 'psnr_std', 
                     'error_ci_lower', 'error_ci_upper', 'ssim_ci_lower', 'ssim_ci_upper',
                     'psnr_ci_lower', 'psnr_ci_upper', 'num_samples']
        """
        if self.config.wells is None:
            raise ValueError("Wells configuration must be provided for wells experiments")
        
        A_tensor = to_torch_tensor(A_tensor, device=self.config.device, dtype=torch.float32)
        test_data = to_torch_tensor(test_tensors[subject_name], device=self.config.device, dtype=torch.float32)
        X_slice = test_data[..., slice_idx]
        
        # Получить wells только для subject_name
        wells_list = self.config.wells.get(subject_name, [])

        # Удаляем дубликаты и невалидные координаты
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
            print(selected_wells)
            print(len(selected_wells))
            wells_dict = {subject_name: selected_wells}
            P = build_wells_matrix(wells_dict, A_tensor.shape, device=self.config.device)
            
            # Build Y matrix only for the specific subject
            Y_mats = build_Y_matrices({subject_name: test_tensors[subject_name]}, P[subject_name], device=self.config.device)
            Y_subject = Y_mats[subject_name]
            Y_slice = Y_subject[..., slice_idx]
            
            slice_errors, slice_ssims, slice_psnrs = [], [], []
            
            # Baseline (no noise)
            x_hat, _ = self._solve_compressive_sensing(A_tensor, P[subject_name], Y_slice)
            X_reconstructed = reconstruct_tensor(A_tensor=A_tensor, x_hat=x_hat)
            err_base, _, ssim_base, psnr_base = compute_metrics(X_slice, X_reconstructed)
            
            slice_errors.append(err_base)
            slice_ssims.append(ssim_base)
            slice_psnrs.append(psnr_base)
            
            # Noise samples
            for _ in range(self.config.num_noise_samples):
                noisy_Y_slice = self._add_noise_to_measurements(Y_slice)
                x_hat, _ = self._solve_compressive_sensing(A_tensor, P[subject_name], noisy_Y_slice)
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
        """
        Run wells experiments across full dataset with all subjects and slices.
        
        Parameters
        ----------
        A_tensor : ndarray | torch.Tensor
            Basis tensor for decomposition.
        test_tensors : Dict[str, ndarray | torch.Tensor]
            Test data per subject.
        sensor_values : List[int]
            Range of sensor counts to evaluate.
            
        Returns
        -------
        pd.DataFrame
            Results with columns: ['sensors', 'error_mean', 'error_std', 'mse_mean', 'mse_std',
            'ssim_mean', 'ssim_std', 'psnr_mean', 'psnr_std', 'error_ci_lower', 'error_ci_upper',
            'mse_ci_lower', 'mse_ci_upper', 'ssim_ci_lower', 'ssim_ci_upper', 
            'psnr_ci_lower', 'psnr_ci_upper', 'num_samples']
        """
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
                    x_hat, _ = self._solve_compressive_sensing(A_tensor, P, Y_slice)
                    X_reconstructed = reconstruct_tensor(A_tensor=A_tensor, x_hat=x_hat)
                    error_val, mse_val, ssim_val, psnr_val = compute_metrics(X_slice, X_reconstructed)
                    
                    all_errors.append(error_val)
                    all_mses.append(mse_val)
                    all_ssims.append(ssim_val)
                    all_psnrs.append(psnr_val)
                    
                    # Noise samples
                    for _ in range(self.config.num_noise_samples):
                        noisy_Y = self._add_noise_to_measurements(Y_slice)
                        x_hat, _ = self._solve_compressive_sensing(A_tensor, P, noisy_Y)
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
    """
    Ensure sensor_values are Python integers.
    
    Converts numpy integers or other numeric types to Python int.
    Useful for avoiding type validation errors.
    
    Parameters
    ----------
    sensor_values : List
        List of sensor counts (may contain numpy integers)
        
    Returns
    -------
    List[int]
        List of Python integers
        
    Examples
    --------
    >>> import numpy as np
    >>> sensor_values = [np.int64(5), np.int32(10), 15]
    >>> clean_values = ensure_sensor_values_are_int(sensor_values)
    >>> print(clean_values)  # [5, 10, 15] (all Python int)
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
    """Deprecated: Use ExperimentRunner class instead."""
    print("Warning: This function is deprecated. Use ExperimentRunner class instead.")
    config = ExperimentConfig(confidence_level=confidence_level)
    runner = ExperimentRunner(config)
    return runner._compute_confidence_intervals(means, stds, num_samples)


def run_experiments(*args, **kwargs):
    """Deprecated: Use ExperimentRunner.run_full_dataset_experiments() instead."""
    print("Warning: This function is deprecated. Use ExperimentRunner.run_full_dataset_experiments() instead.")
    config = ExperimentConfig()
    runner = ExperimentRunner(config)
    # This would need more complex mapping - recommend using the class directly
    raise NotImplementedError("Please use ExperimentRunner class directly")


def run_experiments_single_slice(*args, **kwargs):
    """Deprecated: Use ExperimentRunner.run_single_slice_experiments() instead."""
    print("Warning: This function is deprecated. Use ExperimentRunner.run_single_slice_experiments() instead.")
    raise NotImplementedError("Please use ExperimentRunner class directly")


def run_experiments_df(*args, **kwargs):
    """Deprecated: Use ExperimentRunner.run_experiments() instead."""
    print("Warning: This function is deprecated. Use ExperimentRunner.run_experiments() instead.")
    raise NotImplementedError("Please use ExperimentRunner class directly")


def run_experiments_wells_df(*args, **kwargs):
    """Deprecated: Use ExperimentRunner.run_wells_experiments() instead."""
    print("Warning: This function is deprecated. Use ExperimentRunner.run_wells_experiments() instead.")
    raise NotImplementedError("Please use ExperimentRunner class directly")