# Gemini System Instructions

## Purpose
Specific guidelines for Google's Gemini when assisting with the TBMD codebase.

## Project Context
TBMD is a research repository for Tensor-Based Modal Decomposition, Compressive Sensing, and Digital Twin forecasting (e.g., Navier-Stokes). See `AGENTS.md` for full context.

## Repository Rules
- **Strict Grounding**: Do not import external knowledge about tensor decomposition that contradicts the implementation in `src/TBMD/`. Your answers must be grounded in the repository's actual code.
- **Evidence-Based Claims**: Do not state that the model achieves high accuracy unless you have observed a reproducible test or script output proving it.

## Development Workflow
1. Use workspace search tools to locate usages of a class before modifying its signature.
2. If implementing a new forecaster, inherit from the existing `TBMD.core.forecasting` structures.

## Documentation Rules
- If asked to document limitations, use factual constraints found in the code (e.g., "The ADMM solver requires parameter tuning for convergence").
- Never leave `TODO` placeholders in the documentation. Use `Owner decision required` if blocked.

## Testing and Validation
Run:
```bash
pytest
```
If you encounter a `ModuleNotFoundError` or similar environment issue in the tests, do not aggressively rewrite `__init__.py` files unless you are explicitly tasked with fixing the import structure. Report the failure to the user.

## Safety and Correctness Constraints
- Keep scientific terminology precise. Do not confuse spatial modes with temporal modes.
- Respect `.gitignore`: do not commit files to `data/` or `results/`.

## Final Response Format
Provide a direct, engineering-focused summary of what was accomplished, avoiding marketing or overly conversational filler.
