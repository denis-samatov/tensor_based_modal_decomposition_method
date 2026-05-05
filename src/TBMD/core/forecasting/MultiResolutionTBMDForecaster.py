"""
Multi-Resolution Cascaded Tucker (MRCT) Forecaster.

Improves upon the single-level LatentModalForecaster by decomposing the 
tensor at MULTIPLE resolution levels:
  Level 1: G₁ ×₁ A₁ ×₂ B₁ ×₃ C₁  — captures dominant/smooth energy modes
  Level 2: G₂ ×₁ A₂ ×₂ B₂ ×₃ C₂  — captures residual detail/turbulence modes
  ...

The residual from each level is decomposed by the next level.
Predictions from all levels are SUMMED in the spatial domain:
  X̂(t+1) = Σ_l reconstruct_l(ĉ_l(t+1))

This follows the article's hierarchical mode analysis (Eq. 12) where modes
are extracted at different energy levels.

Reference:
  "A novel tensor-based modal decomposition method for reduced order modeling
   and optimal sparse sensor placement"
"""

import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple, Union, List, Any
from dataclasses import dataclass

from .LatentModalForecaster import LatentModalForecaster, LatentModalResult

try:
    from TBMD.config import (
        MultiResolutionTBMDConfig,
        LatentModalForecasterConfig
    )
except ImportError:
    MultiResolutionTBMDConfig = None

logger = logging.getLogger(__name__)


@dataclass
class MultiResolutionResult:
    """Container for multi-resolution forecasting results."""
    n_levels: int
    level_results: List[LatentModalResult]
    level_decomp_errors: List[float]
    total_decomp_error: float  # Combined residual after all levels
    level_energy_fractions: List[float]  # How much energy each level captures


class MultiResolutionTBMDForecaster:
    """
    Multi-Resolution Cascaded Tucker Forecaster.
    
    Decomposes the tensor hierarchically: each level operates on the
    residual from the previous level. This separates smooth, predictable
    modes from chaotic, fine-scale modes, allowing each level to use
    an optimized sub-forecaster.
    
    Examples:
        >>> from TBMD.core.forecasting import MultiResolutionTBMDForecaster
        >>> from TBMD.config import MultiResolutionTBMDConfig
        >>>
        >>> config = MultiResolutionTBMDConfig(
        ...     level_ranks=[[64, 64, 5], [64, 64, 15]],
        ...     level_forecaster_types=['linear', 'linear'],
        ...     train_ratio=0.8
        ... )
        >>> forecaster = MultiResolutionTBMDForecaster(config=config)
        >>> forecaster.fit(X_tensor)  # X_tensor shape: (I1, I2, T)
        >>> metrics = forecaster.evaluate()
    """
    
    def __init__(self, config: Optional['MultiResolutionTBMDConfig'] = None, **kwargs):
        if config is not None:
            self.config = config
        elif MultiResolutionTBMDConfig is not None:
            self.config = MultiResolutionTBMDConfig(**kwargs)
        else:
            raise ImportError("MultiResolutionTBMDConfig not available.")
        
        self._fitted = False
        self._levels: List[LatentModalForecaster] = []
        self._residuals_train: List[np.ndarray] = []  # For diagnostics
        self._X_original: Optional[np.ndarray] = None
        self._result: Optional[MultiResolutionResult] = None
        self._T_train: int = 0
        self._T_test: int = 0
        
        n_levels = len(self.config.level_ranks)
        if self.config.verbose:
            logger.info(f"MultiResolutionTBMDForecaster initialized "
                       f"({n_levels} levels)")
            for i, (ranks, ftype) in enumerate(zip(
                    self.config.level_ranks, self.config.level_forecaster_types)):
                logger.info(f"  Level {i+1}: ranks={ranks}, forecaster={ftype}")
    
    def fit(self, X: Union[np.ndarray, torch.Tensor]) -> 'MultiResolutionResult':
        """
        Fit the multi-resolution cascade.
        
        For each level:
        1. Tucker-decompose the current residual
        2. Train sub-forecaster on the temporal coefficients
        3. Subtract the Tucker approximation to get the next residual
        
        Args:
            X: Input 3D tensor of shape (I₁, I₂, T)
            
        Returns:
            MultiResolutionResult with per-level decomposition artifacts
        """
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = np.array(X, dtype=np.float64)
        
        if X_np.ndim != 3:
            raise ValueError(f"Expected 3D tensor (I₁, I₂, T), got shape {X_np.shape}")
        
        self._X_original = X_np
        I1, I2, T = X_np.shape
        self._T_train = int(T * self.config.train_ratio)
        self._T_test = T - self._T_train
        
        X_norm = np.linalg.norm(X_np)
        
        if self.config.verbose:
            logger.info(f"Input tensor: ({I1}, {I2}, {T}), "
                       f"T_train={self._T_train}, T_test={self._T_test}")
        
        # Cascade: decompose → residual → decompose → ...
        residual = X_np.copy()
        self._levels = []
        self._residuals_train = []
        level_results = []
        level_decomp_errors = []
        level_energy_fractions = []
        
        for level_idx in range(len(self.config.level_ranks)):
            ranks = self.config.level_ranks[level_idx]
            ftype = self.config.level_forecaster_types[level_idx]
            
            if self.config.verbose:
                logger.info(f"\n{'='*60}")
                logger.info(f"Level {level_idx + 1}: ranks={ranks}, forecaster={ftype}")
                logger.info(f"  Residual norm: {np.linalg.norm(residual):.4f} "
                           f"({np.linalg.norm(residual)/X_norm*100:.1f}% of original)")
            
            # Create per-level config
            level_config = LatentModalForecasterConfig(
                ranks=ranks,
                forecaster_type=ftype,
                train_ratio=self.config.train_ratio,
                epsilon=self.config.epsilon,
                random_state=self.config.random_state,
                delta_forecast=self.config.delta_forecast,
                projection_refinement_steps=self.config.projection_refinement_steps,
                projection_refinement_alpha=self.config.projection_refinement_alpha,
                # MLP params
                mlp_hidden_size=self.config.mlp_hidden_size,
                mlp_num_layers=self.config.mlp_num_layers,
                mlp_dropout=self.config.mlp_dropout,
                mlp_num_epochs=self.config.mlp_num_epochs,
                mlp_learning_rate=self.config.mlp_learning_rate,
                mlp_weight_decay=self.config.mlp_weight_decay,
                mlp_batch_size=self.config.mlp_batch_size,
                mlp_val_split=self.config.mlp_val_split,
                mlp_early_stopping_patience=self.config.mlp_early_stopping_patience,
                # LSTM params
                lstm_hidden_size=self.config.lstm_hidden_size,
                lstm_num_layers=self.config.lstm_num_layers,
                lstm_seq_length=self.config.lstm_seq_length,
                lstm_num_epochs=self.config.lstm_num_epochs,
                lstm_learning_rate=self.config.lstm_learning_rate,
                lstm_batch_size=self.config.lstm_batch_size,
                lstm_val_split=self.config.lstm_val_split,
                lstm_early_stopping_patience=self.config.lstm_early_stopping_patience,
                verbose=self.config.verbose
            )
            
            # Fit this level on the current residual
            level_forecaster = LatentModalForecaster(config=level_config)
            result = level_forecaster.fit(residual, ranks=ranks)
            
            self._levels.append(level_forecaster)
            level_results.append(result)
            level_decomp_errors.append(result.decomposition_error)
            
            # Compute the Tucker approximation of the residual (training part only)
            # to subtract for the next level
            residual_train = residual[:, :, :self._T_train]
            
            # Reconstruct training data from this level's decomposition
            from tensorly.tucker_tensor import tucker_to_tensor
            import tensorly as tl
            tl.set_backend('numpy')
            X_approx_train = tucker_to_tensor((
                tl.tensor(result.core),
                [tl.tensor(result.spatial_mode_1),
                 tl.tensor(result.spatial_mode_2),
                 tl.tensor(result.temporal_coeffs_train)]
            ))
            X_approx_train = np.array(X_approx_train)
            
            # Also reconstruct test data through projection
            X_approx_test = np.zeros_like(residual[:, :, self._T_train:])
            C_test = result.temporal_coeffs_test
            for t in range(C_test.shape[0]):
                X_approx_test[:, :, t] = level_forecaster._reconstruct_from_latent(C_test[t])
            
            # Full approximation
            X_approx_full = np.concatenate([X_approx_train, X_approx_test], axis=2)
            
            # Energy fraction captured by this level
            energy_fraction = np.linalg.norm(X_approx_full) / X_norm
            level_energy_fractions.append(energy_fraction)
            
            # Subtract this level's approximation to get the new residual
            residual = residual - X_approx_full
            self._residuals_train.append(np.linalg.norm(residual))
            
            if self.config.verbose:
                logger.info(f"  Energy captured: {energy_fraction*100:.2f}%")
                logger.info(f"  Remaining residual: {np.linalg.norm(residual)/X_norm*100:.2f}%")
        
        # Total decomposition error (residual after all levels)
        total_decomp_error = np.linalg.norm(residual) / X_norm
        
        self._result = MultiResolutionResult(
            n_levels=len(self._levels),
            level_results=level_results,
            level_decomp_errors=level_decomp_errors,
            total_decomp_error=total_decomp_error,
            level_energy_fractions=level_energy_fractions
        )
        
        self._fitted = True
        
        if self.config.verbose:
            logger.info(f"\n{'='*60}")
            logger.info(f"Multi-Resolution fit complete!")
            logger.info(f"  Total decomposition error: {total_decomp_error:.4f} "
                       f"({total_decomp_error*100:.2f}%)")
            for i, ef in enumerate(level_energy_fractions):
                logger.info(f"  Level {i+1} energy: {ef*100:.2f}%")
        
        return self._result
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate multi-resolution forecasting performance.
        
        For each level, predicts c_{t+1} in latent space, reconstructs 
        in spatial domain. The spatial predictions from ALL levels are 
        summed to produce the final forecast.
        
        Returns:
            Dictionary with evaluation metrics in the spatial domain.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X = self._X_original
        I1, I2, T = X.shape
        
        # Determine the evaluation window (smallest across levels)
        # All levels share the same train_ratio, so T_test is the same
        n_test = self._T_test
        
        # For each level, get the minimum evaluation length
        # (accounting for LSTM seq_length etc.)
        min_eval = n_test - 1  # default for Linear/MLP
        for level in self._levels:
            seq_len = 1
            if level.config.forecaster_type == 'lstm':
                seq_len = getattr(level.config, 'lstm_seq_length', 5)
            level_eval = n_test - seq_len
            min_eval = min(min_eval, level_eval)
        
        if min_eval < 1:
            raise ValueError("Not enough test data for evaluation")
        
        # Ground truth: reconstruct test spatial fields from the original tensor
        X_test_gt = X[:, :, self._T_train:]  # (I1, I2, T_test)
        
        # Accumulate predictions from all levels
        X_pred_accumulated = np.zeros((min_eval, I1, I2), dtype=np.float64)
        X_target_accumulated = np.zeros((min_eval, I1, I2), dtype=np.float64)
        
        per_level_metrics = []
        
        # We need to track what the "target" is in the spatial domain:
        # the full original field at t+seq_len
        # For simplicity, use the last seq_len offset
        max_seq_len = 1
        for level in self._levels:
            if level.config.forecaster_type == 'lstm':
                sl = getattr(level.config, 'lstm_seq_length', 5)
                max_seq_len = max(max_seq_len, sl)
        
        # Target: X[:,:, T_train + max_seq_len : T_train + max_seq_len + min_eval]
        for t_eval in range(min_eval):
            X_target_accumulated[t_eval] = X[:, :, self._T_train + max_seq_len + t_eval]
        
        for level_idx, level in enumerate(self._levels):
            C_test = level.result.temporal_coeffs_test
            seq_len = 1
            if level.config.forecaster_type == 'lstm':
                seq_len = getattr(level.config, 'lstm_seq_length', 5)
            
            # Predict
            n_level_eval = C_test.shape[0] - seq_len
            C_pred = np.zeros((n_level_eval, C_test.shape[1]))
            
            for t in range(n_level_eval):
                window = C_test[t: t + seq_len, :]
                if seq_len == 1:
                    window = window[0]
                C_pred[t] = level.predict_next_latent(window)
            
            # Reconstruct spatial predictions
            # Offset alignment: predictions start at index=seq_len in C_test
            # Which corresponds to time T_train + seq_len in the full tensor
            offset = max_seq_len - seq_len  # alignment offset
            
            for t_eval in range(min_eval):
                src_idx = offset + t_eval
                if src_idx < n_level_eval:
                    X_pred_accumulated[t_eval] += level._reconstruct_from_latent(C_pred[src_idx])
                    
            # Per-level latent metrics
            C_target = C_test[seq_len:seq_len + n_level_eval]
            latent_r2 = 1.0 - (np.sum((C_target - C_pred[:n_level_eval])**2) /
                               max(np.sum((C_target - np.mean(C_target, axis=0))**2), 1e-10))
            per_level_metrics.append({
                'level': level_idx + 1,
                'ranks': level.result.ranks,
                'forecaster_type': level.config.forecaster_type,
                'latent_r2': latent_r2,
                'decomp_error': level.result.decomposition_error,
            })
        
        # Compute spatial metrics
        X_target_flat = X_target_accumulated.reshape(min_eval, -1)
        X_pred_flat = X_pred_accumulated.reshape(min_eval, -1)
        
        spatial_mse = np.mean((X_target_flat - X_pred_flat) ** 2)
        spatial_rmse = np.sqrt(spatial_mse)
        spatial_rel_frob = (np.linalg.norm(X_target_flat - X_pred_flat) / 
                           np.linalg.norm(X_target_flat))
        
        ss_res = np.sum((X_target_flat - X_pred_flat) ** 2)
        ss_tot = np.sum((X_target_flat - np.mean(X_target_flat, axis=0)) ** 2)
        spatial_r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
        
        # Store predictions for visualization
        self._eval_target = X_target_accumulated
        self._eval_pred = X_pred_accumulated
        
        metrics = {
            'spatial_mse': spatial_mse,
            'spatial_rmse': spatial_rmse,
            'spatial_rel_frob_err': spatial_rel_frob,
            'spatial_r2': spatial_r2,
            'total_decomp_error': self._result.total_decomp_error,
            'n_levels': self._result.n_levels,
            'n_eval_steps': min_eval,
            'per_level_metrics': per_level_metrics,
        }
        
        if self.config.verbose:
            logger.info("\n" + "=" * 60)
            logger.info("Multi-Resolution Evaluation Results")
            logger.info("=" * 60)
            logger.info(f"  Levels: {self._result.n_levels}")
            logger.info(f"  Total decomp error: {self._result.total_decomp_error:.4f}")
            for plm in per_level_metrics:
                logger.info(f"  Level {plm['level']}: ranks={plm['ranks']}, "
                           f"{plm['forecaster_type'].upper()}, "
                           f"latent_R²={plm['latent_r2']:.4f}")
            logger.info("-" * 60)
            logger.info(f"  Spatial MSE:  {spatial_mse:.6f}")
            logger.info(f"  Spatial RMSE: {spatial_rmse:.6f}")
            logger.info(f"  Spatial Rel. Frob. Error: {spatial_rel_frob:.6f}")
            logger.info(f"  Spatial R²:   {spatial_r2:.6f}")
            logger.info("=" * 60)
        
        return metrics
    
    def plot_spatial_comparison(self, time_idx: int = 0, 
                                title: str = "Multi-Resolution TBMD Forecast",
                                save_path: Optional[str] = None,
                                show: bool = True):
        """
        Visualize target vs prediction at a given time step.
        
        Must call evaluate() first to populate predictions.
        """
        if not hasattr(self, '_eval_target') or self._eval_target is None:
            raise RuntimeError("Call evaluate() first to generate predictions.")
        
        if time_idx >= len(self._eval_target):
            raise ValueError(f"time_idx {time_idx} out of range "
                           f"(max: {len(self._eval_target)-1})")
        
        X_target = self._eval_target[time_idx]
        X_pred = self._eval_pred[time_idx]
        err = np.abs(X_target - X_pred)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"{title} (t={time_idx})", fontsize=16)
        
        vmin = min(np.min(X_target), np.min(X_pred))
        vmax = max(np.max(X_target), np.max(X_pred))
        
        im0 = axes[0].imshow(X_target, cmap='jet', vmin=vmin, vmax=vmax)
        axes[0].set_title("Ground Truth")
        axes[0].axis('off')
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        im1 = axes[1].imshow(X_pred, cmap='jet', vmin=vmin, vmax=vmax)
        axes[1].set_title("Prediction")
        axes[1].axis('off')
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        im2 = axes[2].imshow(err, cmap='hot')
        axes[2].set_title(f"Absolute Error (Max: {np.max(err):.4f})")
        axes[2].axis('off')
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            logger.info(f"Saved visualization to {save_path}")
            
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    @property
    def result(self) -> Optional[MultiResolutionResult]:
        return self._result
    
    @property
    def is_fitted(self) -> bool:
        return self._fitted
    
    @property
    def levels(self) -> List[LatentModalForecaster]:
        return self._levels
