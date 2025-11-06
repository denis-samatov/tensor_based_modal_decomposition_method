# TBMD Configuration Reference

`algorithm/TBMD/config.py` centralizes default hyper-parameters used across tensor decomposition, sensor placement, and compressive sensing routines. Override these values per experiment either by editing the module, passing custom arguments into constructors, or supplying configuration dataclasses.

## Global Seeds
| Setting | Default | Purpose | Used by |
|---------|---------|---------|---------|
| `SEED` | `42` | Ensures reproducible shuffling, randomized initializations, and algorithmic tie-breaking. | Any module that calls `torch.manual_seed`, NumPy RNG, or `random`. |

## Tensor-Based Tube Fiber-Pivot QR (Algorithm 2)
| Setting | Default | Description |
|---------|---------|-------------|
| `NUMBER_SENSORS` | `200` | Desired number of sensors/measurement locations. Sets the third-order dimension (K) when building sensor masks. Adjust according to domain size and budget. |

The QR implementation (`TensorBasedTubeFiberPivotQRFactorization` and `GeometryAwareTensorQR`) can still accept runtime overrides via their `TensorQRConfig` / `GeometricQRConfig` dataclasses. Use `NUMBER_SENSORS` for quick global experiments; use configs for fine-tuning.

## Tensor-Based Compressive Sensing (Algorithm 3)
| Setting | Default | Role | Notes |
|---------|---------|------|-------|
| `MAX_ITERATIONS` | `100` | Maximum ADMM iterations for recovering sparse coefficients. | Set higher for challenging inverse problems; monitors `convergence_tol` inside solver configs. |
| `CONVERGENCE_EPS` | `1e-2` | Target tolerance for consecutive iterate difference. | Matches ADMM stopping criteria. Reduce for higher accuracy at the cost of runtime. |
| `DAMPING_FACTOR` | `0.95` | Relaxation parameter (λ) stabilizing updates. | Values in `(0,1)` encourage convergence; 1.0 reverts to classical ADMM. |
| `INITIAL_STEP_SIZE` | `1.0` | Initial penalty parameter Δ₀ in adaptive δ policy. | Works with Boyd-style heuristic; only used if auto-tuning enabled. |
| `MAX_STEP_SIZE` | `1.0` | Upper bound Δ_max for adaptive penalty updates. | Increase if residuals oscillate; decrease when solver diverges. |

Detailed solver-specific hyper-parameters live inside `TensorBasedCompressiveSensing.CompressiveSensingConfig`. Instantiate that dataclass to override regularization strength, solver tolerance, or linear solver backend at runtime.

## Backend & Precision
| Setting | Default | Options | Impact |
|---------|---------|---------|--------|
| `SET_BACKEND` | `'pytorch'` | `'pytorch'`, `'numpy'`, `'jax'`, ... | Sets TensorLy backend. Keep `pytorch` when leveraging GPU or mixed-precision support. |
| `DTYPE` | `'float32'` | `'float32'`, `'float64'` | Global floating-point precision for tensors. Use `'float64'` when conditioning is poor (with higher memory cost). |

These backend settings are mirrored by helper functions in `TBMD.utils.utils` (`get_torch_device`, `to_torch_tensor`) to ensure consistent dtype/device placement.

## Customizing in Practice
1. **Per-experiment overrides**: pass explicit arguments to constructors, e.g. `TuckerDecomposerCore(ranks=[30, 30, 10], epsilon=1e-3)`.
2. **Configuration dataclasses**: instantiate `TensorQRConfig`, `GeometricQRConfig`, `CompressiveSensingConfig`, or `ModalProcessorConfig` with desired values.
3. **Environment scripts**: create experiment notebooks/scripts that read YAML/JSON and populate these dataclasses before invoking the modules.

Whenever you deviate from defaults, document the changes alongside your experiment (see `modules/docs/README_ExperimentRunner.md`) to maintain reproducibility.
