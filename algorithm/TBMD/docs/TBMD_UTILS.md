# TBMD Utilities Reference

The `algorithm/TBMD/utils` package collects reusable helpers for data preparation, mesh construction, analytics, and visualization. This guide highlights each module and its primary entry points.

## Data Interfaces
- **`DataLoader.py`**
  - `DataLoader.load_static_tensor` – ingest CSV/Excel tables into `(H, W, T)` tensors with consistent ordering.
  - `load_images_tensor` – convert per-subject PNG folders into stacked 3-D tensors (RGB or grayscale).
  - `load_dynamic_tensor` – reshape time-varying data and optionally split 4-D snapshots into 3-D slices.
  - Unified wrapper `load_data(path, data_type, ...)` auto-selects the appropriate loader and returns NumPy or PyTorch tensors.
  - `load_h5_tensors` – convenience routine for HDF5 datasets storing multiple modalities (pressure, soil, metadata).

- **`data_generation.py`**
  - `GaussianRF` – sample Gaussian random fields with configurable smoothness and correlation length (supports 1D/2D/3D).
  - `navier_stokes_2d` and related helpers – generate synthetic CFD datasets for benchmarking TBMD end-to-end.

## Geometry & Sensor Support
- **`geometry.py`**
  - `MeshGraphBuilder` – create adjacency graphs from regular grids, k-NN, radius, or Delaunay constructions.
  - `MeshGeometry` / `TorchMeshGeometry` – containers storing sparse adjacency, Laplacians, coordinates, and gradient weights with conversion to PyTorch sparse tensors.
  - `GeometricWeightComputer`, `estimate_characteristic_length` – compute gradient-aware weights and mesh scales for sensor placement heuristics.

## Experiment Management
- **`Analytics.py`**
  - `ExperimentConfig` – dataclass bundling solver choices, device placement, noise modelling, and evaluation preferences.
  - `ExperimentRunner` – orchestrates QR factorization, compressive sensing solves, metric computation, and reporting; returns pandas DataFrames.
  - Integrates tightly with `TensorTubeQRDecomposition`, `TensorCompressiveSensing`, and `utils.metrics`.

- **`split_data.py`**
  - `split_data_in_memory` – perform repeated random train/test splits per subject, producing experiment dictionaries.
  - `split_data_in_memory_ordered` – deterministic split along the temporal dimension (first k% train, rest test).

## Metrics & Post-Processing
- **`metrics.py`**
  - `compute_metrics` – returns normalized Frobenius error, MSE, SSIM, and PSNR with optional foreground masks; works for NumPy or PyTorch inputs.
  - Handles skimage version differences gracefully (mask support, channel-wise SSIM).

- **`process_data.py`**
  - `safe_copy`, `inverse_normalization`, `calculate_global_minmax_params`, `normalise_tensor` (and companions) – utilities for preprocessing tensors, managing normalization statistics, and reconstructing original ranges.
  - `foreground_stats` – summarizes min/max/mean/std on foreground voxels using masks or sentinel background values.

- **`utils.py`**
  - `extract_step_number` – sort filenames by embedded timestep identifier.
  - `auto_select_mode` – detect tensor mode matching a coefficient vector for reconstruction.
  - `generate_noisy_datasets` – synthesize noisy variants and persist them to disk.
  - `reconstruct_tensor` – apply basis tensor to coefficient vector with rounding/thresholding.
  - Additional helpers for building measurement matrices (`build_Y_matrices`, `build_wells_matrix`) leveraged by analytics scripts.

## Visualization
- **`plots.py`**
  - `visualize_tensor` – grid-based rendering for 3-D/4-D tensors with optional colorbars, value clipping, and overlayed well coordinates.
  - Smart framing (`frame_step`) for long sequences and support for exporting publication-ready figures.

## Convenience Re-Exports
- `utils/__init__.py` exposes the most frequently used helpers, making imports shorter (e.g., `from TBMD.utils import DataLoader, MeshGraphBuilder`).

Combine these utilities when constructing custom TBMD experiments: load raw datasets, normalize/augment them, build mesh-aware structures, run analytics, and visualize outcomes without rewriting boilerplate code.
