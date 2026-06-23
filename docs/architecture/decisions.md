# Current Architecture Decisions

## Purpose
This document logs the core architectural and methodological decisions embedded in the TBMD codebase.

## Audience
Maintainers and core contributors who need to understand why the repository is structured the way it is.

## Current Decisions

### ADR-001: Separation of Research Scripts and Core Modules
#### Status
Current

#### Context
The project needs to support reusable mathematical functions while allowing researchers to rapidly iterate on field-specific experiments (e.g., Brugge Digital Twin).

#### Decision
The core algorithmic components (decomposition, forecasting, reconstruction, sensor placement) are strictly decoupled from application-specific logic. 

#### Evidence in repository
- `src/TBMD/core/` contains generalized abstractions (e.g., `BatchModalProcessor`).
- `examples/applications/` and orchestration classes like `DigitalTwin` handle the dataset-specific data loading, scaling, and execution.

#### Consequences
- **Pros**: Clean testability of core math; easy to apply to new datasets.
- **Cons**: Requires mapping local data formats (e.g., NumPy/HDF5) to the required PyTorch tensor formats in the orchestration layer.

---

### ADR-002: Tensor-Based Modal Decomposition (Tucker) as the Core Method
#### Status
Current

#### Context
Handling multi-dimensional spatiotemporal data (e.g., 3D space + 1D time) efficiently requires preserving the tensor structure rather than flattening it into matrices.

#### Decision
The primary dimensionality reduction method is Tucker Decomposition (specifically HOSVD-based approaches).

#### Evidence in repository
- `src/TBMD/core/decomposition/` implements `TuckerDecomposer` and `GeometryAwareTuckerDecomposer`.
- Configuration objects explicitly define tensor ranks for core tensors.

#### Consequences
- **Pros**: Exploits spatial correlation across multiple dimensions efficiently.
- **Cons**: Requires tuning of multi-dimensional rank configurations, which can be computationally expensive to optimize.

---

### ADR-003: Compressive Sensing via ADMM
#### Status
Current

#### Context
Reconstructing full high-dimensional tensor states from a sparse set of sensor measurements is an ill-posed inverse problem.

#### Decision
The repository implements the Alternating Direction Method of Multipliers (ADMM) to solve the sparse reconstruction problem.

#### Evidence in repository
- `src/TBMD/core/reconstruction/admm.py` contains the ADMM solver loops.
- `ReconstructionConfig` contains ADMM hyperparameters (rho, iterations).

#### Consequences
- **Pros**: Robust convergence for convex optimization formulations.
- **Cons**: Iterative nature can be slow for real-time digital twin scenarios; requires tuning the penalty parameter (rho).

---

### ADR-004: Local Artifacts Ignored in Version Control
#### Status
Current

#### Context
Research experiments generate large files: trained model weights (`.npz`, `.pt`), raw datasets, and analytical plots.

#### Decision
All generated artifacts and local datasets are strictly excluded from git version control.

#### Evidence in repository
- `.gitignore` (inferred from repository hygiene policies) excludes `data/`, `results/`, and `scripts/plots/`.
- The `test_no_tracked_generated_or_local_artifacts` test in `tests/audit/test_governance.py` enforces this.

#### Consequences
- **Pros**: Keeps the repository lightweight and prevents accidental data leaks.
- **Cons**: Reproducing experiments requires out-of-band data sharing.
