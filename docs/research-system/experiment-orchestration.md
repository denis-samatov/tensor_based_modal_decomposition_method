# Experiment Orchestration

## Purpose
Describes how the `DigitalTwin` module orchestrates data flow between the core mathematical components.

## Audience
Developers and ML Engineers setting up new experiments or benchmarks.

## Summary
The `DigitalTwin` class acts as the highest-level orchestrator. It manages the lifecycle of offline model training, online forecasting, and state reconstruction, allowing researchers to run full end-to-end benchmarks on datasets like Brugge.

## Details
### Orchestration Flow
1. **Configuration**: The pipeline receives a `FullPipelineConfig` that groups `DecompositionConfig`, `SensorPlacementConfig`, and `ReconstructionConfig`.
2. **Data Ingestion**: The Digital Twin accepts raw tensors (e.g., `[features, x, y, time]`) and handles normalization.
3. **Offline Phase**: Calls the decomposer to extract the spatial basis and temporal modes.
4. **Forecasting**: Calls the numerical forecaster to predict future temporal modes based on historical modes and well controls.
5. **Reconstruction**: Integrates predictions and sparse measurements to project the state back into the original high-dimensional space.

### Interfaces
The `DigitalTwin` exposes methods like `.train_offline()` and `.run_online_scenario()` to standardize evaluation scripts.

## Related docs
- [Current Architecture Decisions](../architecture/decisions.md)
