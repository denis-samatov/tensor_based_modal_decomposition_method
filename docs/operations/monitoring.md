# Experiment Monitoring

## Purpose
Explains how to monitor the health and performance of the mathematical algorithms during an experiment run.

## Audience
Researchers running the TBMD pipeline.

## Summary
The TBMD project is not a web service; it does not use Prometheus or Grafana. Monitoring refers to tracking the convergence of iterative solvers and the decay of singular values during decomposition.

## Details

### Tracking ADMM Convergence
The `ADMMReconstructor` iteratively minimizes a loss function.
- **What to monitor**: The primal and dual residuals at each iteration.
- **Expected behavior**: Residuals should strictly decrease. If they plateau early or diverge, the penalty parameter `rho` needs adjustment or the tensor rank is misaligned with the measurement count.

### Tracking Tensor Truncation
When computing the HOSVD:
- **What to monitor**: Singular value decay across the unfolded tensor modes.
- **Expected behavior**: A sharp drop-off (scree plot) indicates that the chosen truncation rank is appropriate. If the values do not decay, the system dynamics are too complex for low-rank approximations.

## Validation
Always output these metrics (e.g., residual history, singular values) to logs or local artifacts during script execution for post-analysis.
