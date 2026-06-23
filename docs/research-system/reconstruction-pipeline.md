# Reconstruction Pipeline

## Purpose
Explains the mathematical and computational workflow used to reconstruct full high-dimensional spatiotemporal fields from sparse sensor measurements.

## Audience
ML/AI Engineers and Researchers studying compressive sensing applied to tensor decomposition.

## Summary
The reconstruction pipeline leverages the Tucker decomposition basis learned during the offline phase. During the online phase, real-time measurements from sparsely placed sensors are combined with the offline basis to solve an inverse problem using the Alternating Direction Method of Multipliers (ADMM).

## Details
The pipeline consists of the following steps:
1. **Offline Training**: `TuckerDecomposer` factorizes the historical reservoir state tensor into core tensors and spatial/temporal factor matrices.
2. **Sensor Placement**: `QRBasedSensorPlacement` determines optimal grid locations for physical or virtual sensors using pivoted QR factorization on the spatial basis matrices.
3. **Online Measurement**: The system receives a sparse vector of measurements at the predetermined sensor locations.
4. **ADMM Reconstruction**: `ADMMReconstructor` formulates the compressive sensing problem as an optimization task, balancing data fidelity (matching the sensor readings) with a structural penalty (e.g., sparsity in the core tensor), solving it iteratively.

## Related docs
- [Current Architecture Decisions](../architecture/decisions.md)
- [Digital Twin Analysis](../product/brugge-digital-twin-analysis.md)
