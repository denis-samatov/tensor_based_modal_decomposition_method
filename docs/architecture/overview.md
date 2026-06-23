# Architecture Overview

## Purpose
To provide a high-level view of how the TBMD library is structured and how its major subsystems interact.

## Audience
Developers and technical contributors looking to understand the overall design and system boundaries before modifying code.

## Summary
The TBMD architecture is designed as a series of composable mathematical modules: Decomposition, Modal Processing, Sensor Placement, Reconstruction, and Forecasting. These are orchestrated via the `DigitalTwin` layer or can be used independently.

## Details
The codebase is structured into clear, decoupled layers:

1. **Configuration (`TBMD.config`)**: Standardized Python dataclasses define the runtime parameters for each component. No environment variables or external credential managers are required.
2. **Core Components (`TBMD.core`)**: Contains the core mathematical operations.
   - **Decomposition**: Implements Tucker/HOSVD.
   - **Modal Processor**: Reshapes decomposition outputs into a modal basis.
   - **Sensor Placement**: Implements Tensor Tube QR algorithms.
   - **Reconstruction**: Implements ADMM-based Compressive Sensing.
   - **Geometry**: Provides graph/mesh awareness for non-Euclidean data.
3. **Digital Twin Orchestration (`TBMD.digital_twin`)**: A high-level layer that binds the core components together to train a forecaster and generate future state predictions.
4. **Experiments and Visualization**: Isolated scripts for running specific benchmarks, tuning hyper-parameters, and generating plots.

### Architectural Principles
- **Composability**: Core components should not be tightly coupled to the Digital Twin. They must remain usable as standalone utilities.
- **Statelessness**: Decomposition and reconstruction instances store the results of their computation, but do not secretly mutate external state.
- **Separation of Data**: The repository source code and the execution artifacts (models, plots, data) are strictly separated. Data and results are isolated in `.gitignore`d directories.

## Validation
To verify the structural integrity of the project (ensuring no unexpected cross-dependencies or structural violations), run the audit tests:
```bash
pytest tests/audit -q
```
Expected result: The audit tests complete without failing, meaning file structure and basic import conventions match the expected architecture.

## Related docs
- [Data Flow](data-flow.md)
- [Components](components.md)
- [Architecture Decisions](decisions.md)
