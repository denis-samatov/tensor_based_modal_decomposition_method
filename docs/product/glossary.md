# Glossary and Terminology

## Purpose
To provide a unified dictionary of terms used across the TBMD documentation and codebase.

## Audience
All audiences (Product, Developer, ML/AI engineer, Maintainer) who need clarity on specific terminology.

## Summary
A centralized list of definitions to prevent ambiguous or contradictory use of terms across the project.

## Details

### Core TBMD Terms
- **Tensor**: A multi-dimensional array of data (e.g., spatiotemporal data).
- **Modal Basis / Modal Tensor**: A reduced representation of the original tensor after decomposition.
- **Decomposition**: The process of factoring a high-dimensional tensor into a core tensor and factor matrices (e.g., Tucker/HOSVD).
- **Sensor Placement**: The algorithmic selection of physical or virtual locations to take measurements, optimizing information gain.
- **Reconstruction**: The process of approximating a full tensor field from sparse measurements.
- **Brugge dataset**: A standard reservoir engineering benchmark field often used for testing dynamic simulation workflows.
- **Digital Twin**: A high-level orchestration pipeline that combines decomposition, forecasting, and sparse reconstruction to emulate and predict a physical system's state over time.

### Standardized AI/System Terms (If Applicable)
*(Note: TBMD focuses on numerical decomposition and forecasting, not generative AI. These terms are defined here for standardization across organizational projects).*

- **Model**: A mathematical or machine learning model (e.g., an LSTM or linear forecaster) used to predict modal coefficients.
- **Evaluation**: The process of measuring a model's accuracy or performance against a validation dataset.
- **Production-ready**: A status indicating the code is fully validated, secure, and robust for live user traffic. **TBMD is a research codebase and is NOT production-ready by default.**
- **Staging / Release**: Environments or phases for deploying software. TBMD does not currently define a formal release pipeline.

## Validation
Ensure that when writing new documentation, the terms used align with the definitions listed here. If a new term is introduced, add it to this file.

## Related docs
- [Product Overview](overview.md)
- [Architecture Overview](../architecture/overview.md)
