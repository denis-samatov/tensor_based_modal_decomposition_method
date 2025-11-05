# Tensor-Based Modal Decomposition (TBMD) Overview

This repository implements an end-to-end **Tensor-Based Modal Decomposition** pipeline for fluid-dynamics simulation data and other high-dimensional fields. It combines tensor decompositions, geometry-aware sensor placement, and compressive sensing to reduce high-fidelity simulations into compact, interpretable modal representations.

## Why TBMD?
- Extract low-rank tensor structure (spatial × spatial × temporal) with minimal information loss.
- Deploy sparse sensor networks that target informative spatial regions.
- Reconstruct full states from limited measurements using domain-aware compressive sensing.
- Support both structured grids and highly irregular meshes through graph-based regularization.

## Pipeline at a Glance
1. **Data Preparation** – load CFD snapshots or synthetic data, optionally augment with noise (`TBMD.utils`).
2. **Tensor Decomposition** – apply HOSVD/Tucker to obtain core tensor and factor matrices (`TBMD.modules.TensorHOSVD`).
3. **Mode Post-Processing** – compute time-insensitive modes or other modal statistics (`TensorTimeInsensitiveModes`).
4. **Sensor Placement** – run geometry-aware QR with optional mesh penalties (`GeometryAwareTensorQR`).
5. **Compressive Sensing** – solve ADMM-based recovery problem for sparse coefficients (`TensorBasedCompressiveSensing`).
6. **Evaluation & Visualization** – compare reconstructions, compute error metrics, and plot results (`TBMD.utils.metrics`, `TBMD.utils.plots`).

## Repository Layout
```
algorithm/TBMD/
├── config.py            # Global algorithm defaults (sensor count, optimization tolerances)
├── modules/             # Core algorithm implementations (HOSVD, QR, CS, modal processing)
├── utils/               # Data I/O, mesh utilities, analytics, plotting, metrics
├── examples/            # Ready-to-run demos covering the full pipeline
└── docs/                # Narrative and API-style documentation (this folder)
```

Complementary top-level documents:
- `README.md` – project mission, installation instructions, entry points.
- `GEOMETRY_AWARE_QUICKSTART.md` – hands-on tutorial for running geometry-aware TBMD.
- `GEOMETRY_AWARE_TBMD.md` – in-depth discussion of Laplacian-regularized decomposition.

## Data & Prerequisites
- Tensor data can be loaded from NumPy arrays, PyTorch tensors, or on-disk snapshot folders.
- For geometry-aware modes, provide cell-center coordinates or a mesh file that can be converted into adjacency graphs.
- GPU acceleration is optional but recommended for large spatial domains (PyTorch backend).
- Key dependencies: `tensorly`, `torch`, `numpy`, `scipy`, `matplotlib`, `tqdm`.

## Typical Workflow
1. **Configure Experiment** – adjust `config.py` or pass custom settings to module constructors.
2. **Load/Generate Dataset** – use `TBMD.utils.DataLoader` or `TBMD.utils.data_generation`.
3. **Run Baseline HOSVD** – `TensorHOSVD.TuckerDecomposerCore` with CPU or GPU strategies.
4. **Enable Geometry Awareness** – build `MeshGeometry` and call `GeometryAwareTensorHOSVD`.
5. **Deploy Sensor Selection** – instantiate `GeometryAwareTensorQR` (or standard QR) with the same mesh.
6. **Recover Fields** – solve for coefficients via `TensorBasedCompressiveSensing`.
7. **Validate** – compare against ground truth with `TBMD.utils.metrics` and produce diagnostic plots.

## Additional Resources
- See `TBMD_CONFIGURATION.md` for an explanation of tunable parameters.
- `TBMD_CORE_MODULES.md` describes class responsibilities and extension hooks.
- `TBMD_UTILS.md` and `TBMD_EXAMPLES.md` document reusable helpers and demos.

Use these documents to navigate the codebase quickly and adapt TBMD to new datasets or research experiments.
