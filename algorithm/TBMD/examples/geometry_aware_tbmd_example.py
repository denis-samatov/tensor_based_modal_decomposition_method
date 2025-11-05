"""
Geometry-Aware TBMD Example: Complete Pipeline

This example demonstrates the full geometry-aware TBMD/QR/CS pipeline:

1. **Build mesh geometry** from data shape (structured) or coordinates (unstructured)
2. **Geometry-aware HOSVD** with Laplacian regularization for smoother spatial modes
3. **Geometry-aware QR sensor placement** with gradient weights and proximity penalties
4. **Standard tensor-CS reconstruction** using the geometric dictionary

The approach improves upon standard TBMD by:
- Better spatial mode quality (respecting mesh topology)
- More uniform sensor coverage (avoiding clustering)
- Prioritizing areas with sharp gradients (fronts, boundaries)
- Better transferability between similar meshes

Dataset: 2D flow field or reservoir pressure data
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import logging

# TBMD modules
from TBMD.utils.geometry import MeshGraphBuilder, GeometricWeightComputer
from TBMD.modules.GeometryAwareTensorHOSVD import (
    GeometryAwareTuckerDecomposer,
    GeometryAwareConfig
)
from TBMD.modules.GeometryAwareTensorQR import (
    GeometryAwareTensorQR,
    GeometricQRConfig
)
from TBMD.modules.TensorBasedCompressiveSensing import (
    TensorCompressiveSensing,
    CompressiveSensingConfig
)
from TBMD.utils.utils import to_torch_tensor
from TBMD.utils.metrics import compute_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeometryAwareTBMDPipeline:
    """
    Complete pipeline for geometry-aware tensor-based modal decomposition.
    
    Pipeline stages:
    1. Build mesh geometry
    2. Geometry-aware HOSVD decomposition
    3. Geometry-aware sensor placement (QR)
    4. Compressive sensing reconstruction
    5. Evaluation and comparison
    """
    
    def __init__(self,
                 spatial_shape: Tuple[int, ...],
                 ranks: Tuple[int, int, int],
                 n_sensors: int,
                 alpha_laplacian: float = 0.1,
                 gradient_weight: float = 0.5,
                 proximity_weight: float = 1.0,
                 connectivity_type: str = 'grid',
                 device: str = 'cpu'):
        """
        Parameters
        ----------
        spatial_shape : tuple
            Shape of spatial domain (H, W) or (H, W, D).
        ranks : tuple (r1, r2, r3)
            Tucker ranks for decomposition.
        n_sensors : int
            Number of sensors to place.
        alpha_laplacian : float, default=0.1
            Laplacian regularization strength in HOSVD.
        gradient_weight : float, default=0.5
            Weight for gradient-based sensor placement.
        proximity_weight : float, default=1.0
            Weight for proximity penalty in sensor placement.
        connectivity_type : str, default='grid'
            Mesh connectivity type ('grid', 'knn', 'delaunay').
        device : str, default='cpu'
            PyTorch device.
        """
        self.spatial_shape = spatial_shape
        self.ranks = ranks
        self.n_sensors = n_sensors
        self.device = device
        
        # Build mesh geometry
        logger.info(f"Building {connectivity_type} mesh for shape {spatial_shape}")
        mesh_builder = MeshGraphBuilder(connectivity_type=connectivity_type)
        self.mesh = mesh_builder.build_from_shape(spatial_shape)
        
        logger.info(f"Mesh: {self.mesh.adjacency_matrix.nnz} edges, "
                   f"{self.mesh.adjacency_matrix.shape[0]} cells")
        
        # Configure geometry-aware HOSVD
        self.geo_hosvd_config = GeometryAwareConfig(
            alpha=alpha_laplacian,
            spatial_modes=[0],  # Regularize spatial mode
            laplacian_type='normalized',
            connectivity_type=connectivity_type
        )
        
        # Configure geometry-aware QR
        self.geo_qr_config = GeometricQRConfig(
            gradient_weight=gradient_weight,
            proximity_weight=proximity_weight,
            min_distance_factor=2.0,
            gradient_method='graph',
            adaptive_weights=True
        )
        
        # Configure CS
        self.cs_config = CompressiveSensingConfig(
            max_iter=1000,
            tol=1e-4,
            epsilon_l1=1e-2,
            device=device
        )
        
        # Storage
        self.tensor_train = None
        self.tensor_test = None
        self.core = None
        self.factors = None
        self.P = None
        self.Q = None
        self.R = None
        self.reconstructed = None
        
    def fit(self, tensor_train: np.ndarray) -> None:
        """
        Fit the model: geometry-aware HOSVD + QR sensor placement.
        
        Parameters
        ----------
        tensor_train : np.ndarray
            Training tensor (spatial × time).
        """
        logger.info(f"=== Stage 1: Geometry-Aware HOSVD ===")
        self.tensor_train = tensor_train
        
        # Perform geometry-aware Tucker decomposition
        decomposer = GeometryAwareTuckerDecomposer(
            tensor=tensor_train,
            mesh=self.mesh,
            geo_config=self.geo_hosvd_config,
            ranks=self.ranks,
            epsilon=1e-2,
            max_iter=50,
            device=self.device
        )
        
        decomposer.decompose()
        
        self.core = decomposer.cores.detach().cpu().numpy()
        self.factors = [f.detach().cpu().numpy() for f in decomposer.factors]
        
        # Reconstruction error
        reconstructed_train = decomposer.reconstruct().detach().cpu().numpy()
        train_error = np.linalg.norm(tensor_train - reconstructed_train) / np.linalg.norm(tensor_train)
        logger.info(f"HOSVD train error: {train_error:.6f}")
        
        logger.info(f"Spatial modes shape: {self.factors[0].shape}")
        logger.info(f"Core shape: {self.core.shape}")
        
        # Stage 2: Geometry-aware sensor placement
        logger.info(f"\n=== Stage 2: Geometry-Aware QR Sensor Placement ===")
        
        # Build spatial basis tensor A
        # A shape: (spatial × W), where W is the rank of spatial+temporal modes
        A_spatial = self.factors[0]  # (N_cells, r1)
        
        # For simplicity, use spatial modes as basis for sensor placement
        # In full pipeline, would use A = U1 ⊗ U3 (spatial ⊗ temporal)
        
        # Perform geometry-aware QR on the spatial factor
        # We need to create a "pseudo-tensor" for QR
        # Use the spatial modes as "tubes" along a dummy dimension
        
        # Create extended tensor for QR: replicate spatial modes
        A_extended = np.repeat(A_spatial[:, :, np.newaxis], self.n_sensors, axis=2)
        A_extended = A_extended.reshape(self.spatial_shape + (self.n_sensors,))
        
        geo_qr = GeometryAwareTensorQR(
            tensor=A_extended,
            mesh=self.mesh,
            N=self.n_sensors,
            field_data=tensor_train,  # Use training data for gradient computation
            config=self.geo_qr_config,
            device=self.device
        )
        
        self.P, self.Q, self.R = geo_qr.factorize()
        
        logger.info(f"Placed {torch.sum(self.P).item()} sensors")
        
        # Get sensor coordinates
        sensor_coords = geo_qr.get_sensor_coordinates()
        logger.info(f"Sensor coordinates shape: {sensor_coords.shape}")
    
    def predict(self, tensor_test: np.ndarray) -> np.ndarray:
        """
        Reconstruct test tensor using learned basis and sensor measurements.
        
        Parameters
        ----------
        tensor_test : np.ndarray
            Test tensor (spatial × time).
        
        Returns
        -------
        np.ndarray
            Reconstructed tensor.
        """
        logger.info(f"\n=== Stage 3: Compressive Sensing Reconstruction ===")
        self.tensor_test = tensor_test
        
        # Build basis tensor A from factors
        # A: spatial_shape × W
        # For Tucker: A[i,j,:] = outer product of spatial modes at (i,j)
        
        # Simplified: use spatial factor directly
        A = self.factors[0].reshape(self.spatial_shape + (self.ranks[0],))
        
        # Apply sensor mask to test data
        P_np = self.P.detach().cpu().numpy()
        Y = tensor_test * P_np[..., np.newaxis]
        
        # For each time step, solve CS problem
        n_time = tensor_test.shape[-1]
        reconstructed = np.zeros_like(tensor_test)
        
        for t in range(n_time):
            # Solve CS for this time step
            cs_solver = TensorCompressiveSensing(
                A=A,
                P=P_np,
                Y=Y[..., t],
                core_cfg=self.cs_config
            )
            
            x_hat, metrics = cs_solver.solve()
            
            # Reconstruct field
            x_hat_np = x_hat.numpy()
            
            # Apply basis: X_reconstructed = A @ x_hat
            # This requires proper tensor contraction
            reconstructed[..., t] = self._apply_basis(A, x_hat_np)
            
            if t % 10 == 0:
                logger.info(f"  Time step {t}/{n_time}: "
                           f"converged={metrics.converged}, "
                           f"iterations={metrics.iterations}")
        
        self.reconstructed = reconstructed
        return reconstructed
    
    def _apply_basis(self, A: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
        """
        Apply basis tensor to coefficients: X = A @ x_hat.
        
        A: (spatial_shape, W)
        x_hat: (W,)
        Result: (spatial_shape)
        """
        # Reshape A to (N_cells, W)
        spatial_size = int(np.prod(A.shape[:-1]))
        A_flat = A.reshape(spatial_size, -1)
        
        # Matrix-vector product
        result_flat = A_flat @ x_hat
        
        # Reshape back to spatial shape
        result = result_flat.reshape(A.shape[:-1])
        
        return result
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate reconstruction quality.
        
        Returns
        -------
        Dict[str, float]
            Metrics: relative error, SSIM, PSNR, etc.
        """
        logger.info(f"\n=== Stage 4: Evaluation ===")
        
        if self.reconstructed is None or self.tensor_test is None:
            raise ValueError("Call predict() first")
        
        # Compute metrics
        metrics = compute_metrics(
            original=self.tensor_test,
            reconstructed=self.reconstructed,
            metrics=['relative_error', 'ssim', 'psnr']
        )
        
        logger.info(f"Reconstruction metrics:")
        for name, value in metrics.items():
            logger.info(f"  {name}: {value:.6f}")
        
        return metrics
    
    def visualize_results(self, time_index: int = 0, figsize: Tuple[int, int] = (16, 4)) -> None:
        """
        Visualize results: original, reconstructed, error, sensors.
        
        Parameters
        ----------
        time_index : int, default=0
            Which time step to visualize.
        figsize : tuple, default=(16, 4)
            Figure size.
        """
        if self.reconstructed is None or self.tensor_test is None:
            raise ValueError("Call predict() first")
        
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        
        # Original
        if len(self.spatial_shape) == 2:
            original_slice = self.tensor_test[..., time_index]
            im0 = axes[0].imshow(original_slice, cmap='viridis', origin='lower')
            axes[0].set_title('Original')
            plt.colorbar(im0, ax=axes[0])
            
            # Reconstructed
            reconstructed_slice = self.reconstructed[..., time_index]
            im1 = axes[1].imshow(reconstructed_slice, cmap='viridis', origin='lower')
            axes[1].set_title('Reconstructed')
            plt.colorbar(im1, ax=axes[1])
            
            # Error
            error_slice = np.abs(original_slice - reconstructed_slice)
            im2 = axes[2].imshow(error_slice, cmap='Reds', origin='lower')
            axes[2].set_title('Absolute Error')
            plt.colorbar(im2, ax=axes[2])
            
            # Sensor placement
            P_np = self.P.detach().cpu().numpy()
            axes[3].imshow(original_slice, cmap='gray', alpha=0.5, origin='lower')
            sensor_pos = np.argwhere(P_np == 1)
            if len(sensor_pos) > 0:
                axes[3].scatter(sensor_pos[:, 1], sensor_pos[:, 0],
                               c='red', s=50, marker='x', linewidths=2,
                               label=f'{len(sensor_pos)} sensors')
            axes[3].set_title('Sensor Placement')
            axes[3].legend()
        
        plt.tight_layout()
        plt.show()
    
    def compare_with_standard_tbmd(self, tensor_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Compare geometry-aware TBMD with standard TBMD.
        
        Parameters
        ----------
        tensor_test : np.ndarray
            Test tensor.
        
        Returns
        -------
        Dict[str, Dict[str, float]]
            Comparison metrics.
        """
        logger.info(f"\n=== Comparison: Geometry-Aware vs. Standard TBMD ===")
        
        # Geometry-aware metrics (already computed)
        geo_metrics = self.evaluate()
        
        # Standard TBMD (without geometry)
        # TODO: Implement standard TBMD comparison
        # For now, just return geometry-aware metrics
        
        comparison = {
            'geometry_aware': geo_metrics,
            'standard': {}  # Placeholder
        }
        
        return comparison


def demo_geometry_aware_tbmd():
    """
    Demonstration of geometry-aware TBMD on synthetic 2D flow data.
    """
    logger.info("=" * 70)
    logger.info("Geometry-Aware TBMD Demonstration")
    logger.info("=" * 70)
    
    # Generate synthetic data
    logger.info("\n=== Generating Synthetic Data ===")
    H, W, T = 50, 50, 100
    spatial_shape = (H, W)
    
    # Create a simple flow pattern (vortex + gradient)
    x = np.linspace(-2, 2, W)
    y = np.linspace(-2, 2, H)
    X, Y = np.meshgrid(x, y)
    
    # Generate time-varying field
    tensor_data = np.zeros((H, W, T))
    
    for t in range(T):
        # Vortex
        theta = 2 * np.pi * t / T
        vortex = np.sin(np.sqrt(X**2 + Y**2) - theta)
        
        # Gradient front
        front = np.tanh(X * np.cos(theta) + Y * np.sin(theta))
        
        # Combined field
        tensor_data[..., t] = 0.7 * vortex + 0.3 * front
    
    # Add noise
    tensor_data += 0.05 * np.random.randn(H, W, T)
    
    logger.info(f"Data shape: {tensor_data.shape}")
    logger.info(f"Data range: [{tensor_data.min():.3f}, {tensor_data.max():.3f}]")
    
    # Split train/test
    n_train = 70
    tensor_train = tensor_data[..., :n_train]
    tensor_test = tensor_data[..., n_train:]
    
    logger.info(f"Train: {tensor_train.shape}, Test: {tensor_test.shape}")
    
    # Initialize pipeline
    pipeline = GeometryAwareTBMDPipeline(
        spatial_shape=spatial_shape,
        ranks=(20, 5, 50),  # (spatial, modes, temporal)
        n_sensors=30,
        alpha_laplacian=0.1,
        gradient_weight=0.5,
        proximity_weight=1.0,
        connectivity_type='grid',
        device='cpu'
    )
    
    # Fit model
    pipeline.fit(tensor_train)
    
    # Predict on test data
    reconstructed = pipeline.predict(tensor_test)
    
    # Evaluate
    metrics = pipeline.evaluate()
    
    # Visualize
    pipeline.visualize_results(time_index=0)
    
    # Compare with standard
    comparison = pipeline.compare_with_standard_tbmd(tensor_test)
    
    logger.info("\n" + "=" * 70)
    logger.info("Demonstration Complete!")
    logger.info("=" * 70)
    
    return pipeline, metrics, comparison


if __name__ == "__main__":
    # Run demonstration
    pipeline, metrics, comparison = demo_geometry_aware_tbmd()
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print("\nGeometry-Aware TBMD Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")

