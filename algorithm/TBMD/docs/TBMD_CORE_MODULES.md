# TBMD Core Modules Reference

This document summarizes the responsibilities, key classes, and extension hooks for each module in `algorithm/TBMD/modules`. Use it as a quick API map when integrating or extending the Tensor-Based Modal Decomposition workflow.

---

## `TensorHOSVD.py`
**Purpose**: Canonical Tucker/HOSVD decomposition with robust preprocessing, validation, and CPU/GPU execution strategies.

- **Key Classes**
  - `TensorProcessor` – normalizes inputs (NumPy / TensorLy / PyTorch) onto the desired device/dtype.
  - `TuckerDecomposerCore` – wraps `tensorly.decomposition.tucker`, handling rank validation and tolerances.
  - `CPUStrategy` / `GPUStrategy` – execution policies for batched decompositions and reconstructions.
  - `TensorReconstructor` – rebuilds tensors and computes relative Frobenius error.
  - `TensorVisualizer` – matplotlib helpers for comparing original vs reconstructed slices.
- **Error Handling**: Custom exceptions (`ValidationError`, `InvalidRankError`, `TensorDecompositionError`) for precise diagnostics.
- **Extension Points**
  - Implement new `ProcessingStrategy` subclasses (e.g., distributed execution).
  - Replace `tucker` call with alternate decompositions (HOOI iterations, regularized variants).
  - Hook into logging to capture per-tensor performance metrics.

---

## `GeometryAwareTensorHOSVD.py`
**Purpose**: Extends Tucker decomposition with Laplacian regularization derived from mesh connectivity.

- **Configuration**
  - `GeometryAwareConfig` – controls α (regularization strength), Laplacian type, connectivity strategy, and generalized eigenmode solving.
- **Core Components**
  - `GeometryAwareTuckerCore` – performs alternating updates with Laplacian penalty on specified modes.
  - Integration with `MeshGeometry` (from `TBMD.utils.geometry`) ensures adjacency matrices and Laplacians stay consistent.
- **Usage Pattern**
  1. Build mesh via `MeshGraphBuilder`.
  2. Instantiate `GeometryAwareTuckerCore` with desired ranks.
  3. Call `.decompose(tensor)` to obtain regularized factors.
- **Notes**: Supports both normalized and standard Laplacians; can fall back to standard Tucker when `alpha=0`.

---

## `TensorBasedTubeFiberPivotQRFactorization.py`
**Purpose**: Implements Algorithm 2 – sensor selection via tensor tube fiber-pivot QR with distribution penalties.

- **Key Dataclasses / Components**
  - `TensorQRConfig` – aggregating numerical tolerances, distribution weights, and chunk sizes.
  - `TensorValidator` – ensures tensors and masks are well-formed before factorization.
  - `NumericallyStableOperations` – Householder reflections and stable norm computations.
  - `UniformDistributionManager` – balances sensor placement across slices/regions.
  - `TensorTubeQRDecomposition` – orchestrates the full factorization and sensor extraction loop.
- **Diagnostics**: Optional matplotlib visualizations for placement patterns; condition number monitoring.
- **Customization**: Modify `DISTRIBUTION_PENALTY_WEIGHT` or plug in new penalty terms for domain-specific constraints.

---

## `GeometryAwareTensorQR.py`
**Purpose**: Geometry-aware variant of QR pivot selection prioritizing gradients, spacing, and mesh topology.

- **Configuration**
  - `GeometricQRConfig` extends `TensorQRConfig` with `gradient_weight`, `proximity_weight`, `distribution_weight`, and `min_distance_factor`.
- **Core Classes**
  - `GeometryAwarePivotSelector` – augments residual-based scores with geometry-derived weights, distance penalties, and adaptive thresholds.
  - Integrates with `GeometricWeightComputer` and `estimate_characteristic_length` from `utils.geometry`.
- **Workflow**
  1. Provide `MeshGeometry` and optional field snapshots to estimate gradients.
  2. Run `select_pivots` to obtain sensor indices respecting mesh constraints.
  3. Feed indices into downstream compressive sensing stages.
- **Mesh Support**: Works with Euclidean or graph distances; toggled via `use_graph_distance`.

---

## `TensorBasedCompressiveSensing.py`
**Purpose**: ADMM solver for recovering sparse modal coefficients (Algorithm 3).

- **Current Status**: The module contains a detailed, commented blueprint (v2.1) referencing canonical helper usage (`get_torch_device`, `to_torch_tensor`). The final implementation mirrors the documented methods.
- **Planned Structure**
  - `CompressiveSensingConfig` – ADMM hyper-parameters (max iterations, relaxation, tolerances).
  - `TensorCompressiveSensing` – encapsulates preparation of sensing matrices, ADMM loop, adaptive penalty updates, and convergence metrics.
  - `SolverMetrics` – provides iterations count, residuals, timing, and objective values.
- **Extensibility**: Hooks for alternative linear solvers (`triangular`, `direct`, `robust`), delta-update policies, and diagnostic logging.
- **Action Item**: When enabling the solver, ensure the commented stub is fully implemented or import the finalized version from branch history.

---

## `TensorTimeInsensitiveModes.py`
**Purpose**: Computes time-insensitive modes \( M_{:,n} = A \times G_{:,n} \times B \) and manages batch processing strategies.

- **Configurations & Types**
  - `ModalProcessorConfig` – controls device placement, batching, memory usage, and validation.
  - Enumerations & exceptions (`ProcessingStrategy`, `ValidationError`, `ComputationError`) for robust workflows.
- **Core Logic**
  - `TensorValidator` – ensures compatibility between core tensor slices and factor matrices.
  - `TimeInsensitiveModeComputer` – materializes modes sequentially or in batches, with optional PyTorch acceleration.
- **Usage**
  - Supply single-subject tensors or dictionaries keyed by subject/experiment IDs.
  - Choose strategy (`SEQUENTIAL`, `BATCH`, `MEMORY_EFFICIENT`) depending on hardware limits.
  - Request NumPy or PyTorch outputs via `return_numpy`.

---

## `modules/docs/` Cross References
- `MODAL_TENSOR_IMPROVEMENTS.md` – contextual notes on past refactors for modal processing.
- `QR_ALGORITHM_FIXES.md` & `ALGORITHM_3_FIXES.md` – changelogs explaining numerical fixes for QR and compressive sensing.
- `README_ExperimentRunner.md` – instructions for orchestrating experiments programmatically.

Refer to these ancillary documents when diving into historical rationale or design decisions.
