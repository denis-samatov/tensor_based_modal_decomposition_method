# TBMD Examples & Demos

The `algorithm/TBMD/examples` directory contains runnable scripts that illustrate end-to-end and component-level usage of the TBMD toolkit. Each script can be launched directly (`python examples/<script>.py`) after installing dependencies.

| Script | Focus | Highlights |
|--------|-------|------------|
| `tbmd_example.py` | Baseline HOSVD workflow | Demonstrates `TuckerDecomposerInterface` on single tensors and collections, showcases state management, reconstruction error reporting, and CPU strategy configuration. |
| `modal_tensor_processing_example.py` | Time-insensitive modes | Uses `TensorTimeInsensitiveModes` utilities to compute \( M_{:,n} \) slices, compare batching strategies, and validate output shapes. |
| `qr_example.py` | Sensor placement (Algorithm 2) | Runs `TensorTubeQRDecomposition` with default penalties, visualizes pivot selection, and inspects orthogonality statistics. |
| `sensor_values_fix_demo.py` | Sensor mask consistency | Highlights recent fixes ensuring consistent sensor values/masks when integrating QR with compressive sensing. Useful regression reference. |
| `compressive_sensing_example.py` | ADMM solver (Algorithm 3) | Walks through building sensing matrices, instantiating `TensorCompressiveSensing`, and plotting convergence history/metrics. |
| `geometry_aware_tbmd_example.py` | Full geometry-aware pipeline | Builds mesh graphs, applies Laplacian-regularized HOSVD, runs geometry-aware QR, solves compressive sensing, and evaluates SSIM/PSNR gains. |
| `unified_experiments_demo.py` | Batch experiment runner | Demonstrates `Analytics.ExperimentRunner` orchestration across multiple seeds/noise levels, producing pandas summaries and plots. |
| `test_geometry_aware_components.py` | Sanity checks | Lightweight unit-style tests for mesh construction, gradient weights, and geometry-aware decomposition core; serves as reference for expected interfaces. |

## Running the Demos
1. Activate the project environment (`pip install -r requirements.txt`).
2. Navigate to `algorithm/TBMD/examples`.
3. Execute scripts individually, adjusting parameters inside the files or by editing `config.py`.

Many examples accept optional command-line arguments; consult inline comments or function docstrings for available flags. For notebook-based exploration, `algorithm/TBMD/notebooks/` mirrors several workflows with richer visualization.
