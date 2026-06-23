# Product Overview

## Purpose
This document provides a high-level overview of the Tensor-Based Modal Decomposition Method (TBMD). It explains the core capabilities and the primary value proposition of the library.

## Audience
Product managers, business readers, and technical leads who need to understand what the product does, what problems it solves, and how it is structured without diving into deep implementation details.

## Summary
TBMD is a Python research library designed for reduced-order modeling of spatiotemporal tensor data. It reduces high-dimensional data into a compact modal representation, enabling efficient sensor placement, field reconstruction, and forecasting.

## Details
The primary focus of TBMD is to handle spatiotemporal data (such as data generated from computational fluid dynamics or reservoir-modeling experiments) and build computationally efficient representations. 

The core workflow consists of several key stages:
1. **Data Preparation**: Loading and structuring tensor data (e.g., `(x, y, time)` or `(active_cells, time)`).
2. **Decomposition**: Applying Tucker or HOSVD decomposition to extract core tensors and factor matrices.
3. **Modal Processing**: Building a modal basis from the decomposition outputs.
4. **Sensor Placement**: Selecting optimal informative sensor locations using tensor QR factorization.
5. **Reconstruction**: Using Compressive Sensing (ADMM-based solvers) to reconstruct full fields from sparse measurements.
6. **Digital Twin Forecasting**: Training a forecaster in the modal space to predict future states based on current or historical data.

The project is heavily geared towards scientific computing and should be treated as a research and engineering codebase rather than a validated production simulator.

## Examples
Typical input data shapes handled by the system:
```text
(x, y, time)
(x, y, z, time)
(active_cells, time)
```

## Validation
To verify the core decomposition capabilities, run the corresponding unit tests:
```bash
pytest tests/unit/core/decomposition -q
```
Expected result: All tests pass, validating that the mathematical operations for decomposition execute without errors.

## Related docs
- [Use Cases](use-cases.md)
- [Limitations](limitations.md)
- [Architecture Overview](../architecture/overview.md)
