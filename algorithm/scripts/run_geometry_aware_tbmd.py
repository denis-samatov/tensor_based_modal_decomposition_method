import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# TBMD imports
from TBMD.utils.data_loader import DataLoader
from TBMD.utils.split_data import split_data_in_memory_ordered
from TBMD.utils.geometry import MeshGraphBuilder
from TBMD.utils.tbmd_utils import get_torch_device

# Geometry-Aware Modules
from TBMD.modules.GeometryAwareTensorHOSVD import (
    GeometryAwareTuckerDecomposer, 
    GeometryAwareConfig as HOSVDConfig
)
from TBMD.modules.GeometryAwareTensorQR import (
    GeometryAwareTensorQR, 
    GeometricQRConfig as QRConfig
)
from TBMD.modules.GeometryAwareTensorCS import (
    GeometryAwareTensorCS, 
    GeometryAwareCSConfig as CSConfig
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # --- 1. Load Data ---
    data_path = Path("/Users/denissamatov/Heriot-Watt/tensor-based-modal-decomposition-method/data/Brugge data/data_exp_4_.h5")
    wells_path = Path("/Users/denissamatov/Heriot-Watt/tensor-based-modal-decomposition-method/data/Brugge data/all_wells_exp_4.json")
    
    logger.info("Loading data...")
    tensors = DataLoader.load_h5_tensors(data_path)
    # wells = DataLoader.load_wells_from_json(wells_path) # Loaded but not strictly needed for the core alg unless we use specific well locations
    
    # Select a subject (e.g., the 3rd one as in the user example)
    subject_name = list(tensors['pressure'].keys())[2]
    tensor_data = tensors['pressure'][subject_name]
    
    logger.info(f"Selected subject: {subject_name}")
    logger.info(f"Tensor shape: {tensor_data.shape}")
    
    # --- 2. Split Data ---
    # Split into train (for HOSVD/QR) and test (for CS reconstruction)
    # Note: split_data_in_memory_ordered expects a dict of subjects
    train_data_dict, test_data_dict = split_data_in_memory_ordered(
        {subject_name: tensor_data}, 
        train_ratio=0.8
    )
    
    train_tensor = train_data_dict[subject_name]
    test_tensor = test_data_dict[subject_name]
    
    logger.info(f"Train shape: {train_tensor.shape}")
    logger.info(f"Test shape: {test_tensor.shape}")
    
    # --- 3. Build Mesh Geometry ---
    # Assuming the spatial dimensions are the first N-1 dimensions
    # The data shape is likely (X, Y, Z, Time) or (X, Y, Time)
    spatial_shape = train_tensor.shape[:-1]
    
    logger.info(f"Building mesh for spatial shape: {spatial_shape}")
    
    # Use Grid connectivity (standard for structured reservoir grids)
    # If you have explicit coordinates, use build_from_coordinates
    builder = MeshGraphBuilder(connectivity_type='grid')
    mesh = builder.build_from_shape(spatial_shape)
    
    # --- 4. Geometry-Aware HOSVD ---
    logger.info("Running Geometry-Aware HOSVD...")
    
    # Config: Regularize mode 0 (spatial) with Laplacian
    hosvd_config = HOSVDConfig(
        alpha=0.05,              # Regularization strength
        spatial_modes=[0],       # Flattened spatial mode is mode 0
        laplacian_type='normalized'
    )
    
    # Flatten spatial dimensions for decomposition if needed, or keep as is
    # The GeometryAwareTuckerDecomposer handles reshaping internally if we pass the mesh
    # But typically for TBMD we treat spatial as one mode (flattened) or multiple modes.
    # Let's check how the decomposer expects it.
    # The decomposer expects the input tensor to match the mesh. 
    # If mesh is built from shape (X,Y,Z), it has N_cells = X*Y*Z.
    # If we pass the raw (X,Y,Z,T) tensor, TensorProcessor might flatten it?
    # Let's manually flatten spatial dims to be safe and consistent with TBMD usually working on (Space, Time) or (X, Y, Z, T)
    # The GeometryAwareTuckerDecomposer docstring says:
    # "Tensor can be 2D (spatial_cells, time) or 3D+ (spatial_x, spatial_y, ..., time)"
    # So we can pass the original tensor.
    
    # Ranks: We need to specify ranks. 
    # If tensor is (X, Y, Z, T), ranks should be [Rx, Ry, Rz, Rt]
    # Let's pick some reasonable ranks or use a fraction of dimensions
    ranks = [min(s, 20) for s in train_tensor.shape]
    
    decomposer = GeometryAwareTuckerDecomposer(
        tensor=train_tensor,
        mesh=mesh,
        geo_config=hosvd_config,
        ranks=ranks,
        device='cpu' # Use 'cuda' if available
    )
    
    decomposer.decompose()
    
    # Get spatial basis (U_spatial)
    # If we kept dimensions separate, we have U_x, U_y, U_z.
    # But for Sensor Placement (QR), we usually want a "Spatial Basis" matrix (N_cells x K).
    # If we did HOSVD on (X,Y,Z,T), we get factors for X, Y, Z.
    # To get a single spatial basis for QR, we might need to reshape to (Space, Time) first?
    # OR we use the factors to reconstruct the "Tube" structure.
    # The GeometryAwareTensorQR expects a tensor (Spatial, Time) or (X, Y, ..., Time).
    # And it does QR on the *transposed* tensor or similar?
    # Wait, QR is usually done on the Basis matrix or the Data snapshot matrix.
    # In TBMD-QR (Algorithm 2), we often do QR on the spatial modes U.
    # If we have separated spatial modes, we might need the Kronecker product of them?
    # Let's simplify: Reshape data to (Space, Time) for the whole pipeline to be consistent with "Mode 0 is spatial".
    
    logger.info("Reshaping to (Space, Time) for consistent TBMD pipeline...")
    spatial_dim_prod = np.prod(spatial_shape)
    time_dim = train_tensor.shape[-1]
    
    train_matrix = train_tensor.reshape(spatial_dim_prod, time_dim)
    # Re-build mesh is fine (it's the same N_cells)
    
    # Re-run HOSVD on matrix (Space, Time) -> SVD essentially, but with Laplacian on Mode 0
    hosvd_config.spatial_modes = [0]
    ranks_2d = [min(spatial_dim_prod, 50), min(time_dim, 20)] # Keep 50 spatial modes, 20 temporal
    
    decomposer_2d = GeometryAwareTuckerDecomposer(
        tensor=train_matrix,
        mesh=mesh,
        geo_config=hosvd_config,
        ranks=ranks_2d
    )
    decomposer_2d.decompose()
    
    # The spatial basis is the factor for mode 0
    spatial_basis = decomposer_2d.factors[0] # Shape (N_cells, R_spatial)
    logger.info(f"Spatial basis shape: {spatial_basis.shape}")
    
    # --- 5. Geometry-Aware QR (Sensor Placement) ---
    logger.info("Running Geometry-Aware QR for Sensor Placement...")
    
    qr_config = QRConfig(
        gradient_weight=0.5,
        proximity_weight=1.0,
        min_distance_factor=1.5,
        amplitude_weight=1.0
    )
    
    n_sensors = 20
    
    # We perform QR on the Spatial Basis (U) to find best rows (sensors)
    # The GeometryAwareTensorQR expects "tensor" input. If we pass the Basis, it treats it as the data.
    # We want to select rows of U.
    
    qr_solver = GeometryAwareTensorQR(
        tensor=spatial_basis, # (N_cells, R_spatial)
        mesh=mesh,
        N=n_sensors,
        config=qr_config,
        field_data=train_matrix # Use full field data for gradient/amplitude computation
    )
    
    P, Q, R = qr_solver.factorize()
    
    # Get sensor indices
    sensor_indices = torch.nonzero(P.flatten(), as_tuple=False).flatten().cpu().numpy()
    logger.info(f"Selected {len(sensor_indices)} sensors.")
    
    # --- 6. Geometry-Aware CS (Reconstruction) ---
    logger.info("Running Geometry-Aware CS Reconstruction...")
    
    # Prepare Test Data
    # We want to reconstruct test snapshots from sparse measurements
    # y = P @ x_test
    # We want to recover x_test (or coefficients alpha such that x = U @ alpha)
    
    # In TBMD-CS:
    # We assume x ≈ U @ c
    # Measurements y = P @ x = P @ U @ c
    # We solve for c: min || P U c - y || + Reg
    # Then x_rec = U @ c
    
    # The GeometryAwareTensorCS class solves: min || A x - y || + ...
    # Here A = P @ U (sampled basis)
    # y = measurements
    # x = coefficients (c)
    
    # Let's take one test snapshot
    test_snapshot = test_tensor.reshape(spatial_dim_prod, -1)[:, 0] # First test snapshot
    test_snapshot_t = get_torch_device('cpu')
    test_snapshot_t = torch.from_numpy(test_snapshot).float()
    
    # Measurements
    # P is (N_cells,) mask or (N_cells, 1)?
    # GeometryAwareTensorQR returns P as (N_cells,) or (X,Y,Z) tensor of 0/1
    P_flat = P.flatten().bool()
    y_measurements = test_snapshot_t[P_flat]
    
    # Construct A = U (full basis)
    # The CS solver takes A (full), P (mask), Y (measurements)
    # It internally does As = A[mask]
    
    cs_config = CSConfig(
        alpha=0.01, # Laplacian weight
        auto_alpha=True
    )
    
    cs_solver = GeometryAwareTensorCS(
        A=spatial_basis,     # The basis (N_cells, R_spatial)
        P=P,                 # The sensor mask (N_cells,) or shaped
        Y=y_measurements,    # The measurements (N_sensors,)
        mesh=mesh,
        core_cfg=cs_config
    )
    
    # Solve for coefficients
    coeffs, metrics = cs_solver.solve()
    
    # Reconstruct full field
    # x_rec = U @ coeffs
    reconstructed_field = spatial_basis @ coeffs
    
    # --- 7. Metrics ---
    # Calculate errors
    rel_error = torch.norm(reconstructed_field - test_snapshot_t) / torch.norm(test_snapshot_t)
    logger.info(f"Reconstruction Relative Error: {rel_error.item():.4f}")
    
    # SSIM (if 2D/3D) - simple approximation or skip for now
    # We can reshape back to spatial shape
    rec_shaped = reconstructed_field.reshape(spatial_shape).detach().numpy()
    orig_shaped = test_snapshot.reshape(spatial_shape)
    
    # Simple visualization
    if len(spatial_shape) == 2:
        plt.figure(figsize=(10, 4))
        plt.subplot(131)
        plt.title("Original")
        plt.imshow(orig_shaped)
        plt.subplot(132)
        plt.title("Reconstructed")
        plt.imshow(rec_shaped)
        plt.subplot(133)
        plt.title("Error")
        plt.imshow(np.abs(orig_shaped - rec_shaped))
        plt.savefig("reconstruction_result.png")
        logger.info("Saved reconstruction_result.png")

if __name__ == "__main__":
    main()
