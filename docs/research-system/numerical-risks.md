# Numerical Risks and Methodological Pitfalls

## Purpose
Highlights the known risks and common methodological errors when running the TBMD pipeline.

## Audience
Researchers and Engineers designing experiments.

## Summary
The pipeline involves sensitive hyperparameters and temporal splits. Careless configuration can lead to data leakage, numerical instability, or misleading accuracy claims.

## Details

### 1. Temporal Data Leakage
- **Risk**: Fitting the data normalizer (scaler) or the Tucker decomposition basis using the entire dataset, including the test set.
- **Mitigation**: Ensure that the `DigitalTwin.train_offline()` method is only provided with the training split.

### 2. Rank Selection Instability
- **Risk**: Choosing ranks that are too large relative to the number of samples can lead to overfitting or ill-conditioned matrices during inversion steps.
- **Mitigation**: Perform singular value decay analysis to select optimal truncation points.

### 3. ADMM Convergence
- **Risk**: The ADMM solver for compressive sensing may fail to converge if the penalty parameter `rho` is poorly tuned for the scale of the data.
- **Mitigation**: Ensure data is standard-scaled before reconstruction. Monitor the ADMM residual logs.

### 4. Overclaiming Accuracy
- **Risk**: Achieving low reconstruction error on the training set does not guarantee accurate multi-step physical forecasting.
- **Mitigation**: Always report test-set forecast errors independently of offline reconstruction errors.

## Related docs
- [Digital Twin Analysis](../product/brugge-digital-twin-analysis.md)
