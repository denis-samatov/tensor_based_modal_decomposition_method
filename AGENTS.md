# Agent Instructions

## Purpose
This document provides guidelines for any autonomous coding agent assisting with the Tensor-Based Modal Decomposition Method (TBMD) repository.

## Project Context
The TBMD project is a Python research library for reduced-order modeling of spatiotemporal tensor data. Core capabilities include Tucker/HOSVD decomposition, sparse sensor placement (using Tensor QR), tensor reconstruction (ADMM/Compressive Sensing), and Digital Twin numerical forecasting (e.g., Navier-Stokes and Brugge reservoir experiments). **This is a numerical machine learning and scientific computing codebase. It does not implement LLMs, RAG, chatbots, or autonomous agent frameworks.**

## Repository Rules
- **No LLM/RAG hallucination**: Do not attempt to add or document LLM, GenAI, or conversational features.
- **Artifact Isolation**: Datasets, trained model files (`.npz`), and generated plots must be placed in ignored directories (`data/`, `results/`, `scripts/plots/`) and not committed.
- **No "Production-Ready" claims**: Treat the code as a research codebase. Do not make claims about scalability, production-readiness, or unmatched accuracy.

## Development Workflow
1. Read the `README.md` and `docs/architecture/overview.md` to understand the mathematical flow.
2. Ensure you understand the distinction between core reusable modules (`src/TBMD/core/`) and experiment orchestration (`src/TBMD/digital_twin/`).
3. Maintain test parity when modifying public configurations.

## Documentation Rules
- Write documentation in the Google Developer Documentation style: direct, verifiable, and structured.
- Always include `Validation` instructions using reproducible commands (e.g., `pytest`).
- Avoid empty `TODO` stubs. If an architectural decision is unclear, use an `Owner decision required` block.

## Testing and Validation
After making changes, run the following to ensure syntax and structural hygiene:
```bash
python -m compileall src tests examples scripts
pytest tests/audit -q
pytest tests/unit -q
```

## Safety and Correctness Constraints
- Do not invent non-existent mathematical features.
- Do not modify core algorithm logic (like ADMM steps or Tucker decomposition factors) without explicit user instruction.
- Do not add arbitrary dependencies to `pyproject.toml` unless strictly necessary for an approved feature.

## Final Response Format
When your task is complete, clearly list the files modified, the tests run to validate the changes, and any edge cases that might require the repository owner's attention.
