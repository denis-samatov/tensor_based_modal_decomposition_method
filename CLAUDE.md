# Claude System Instructions

## Purpose
Specific guidelines for Anthropic's Claude when generating code or documentation for the TBMD project.

## Project Context
TBMD focuses on tensor decomposition, sparse sensor placement, and numerical forecasting for scientific computing. See `AGENTS.md` for a full breakdown.

## Repository Rules
- **Small, Focused Diffs**: Do not perform broad refactoring of the entire codebase when asked to fix a specific bug. Keep your changes isolated to the relevant files.
- **Explicit Assumptions**: If you are unsure about a mathematical tensor shape dimension (e.g., whether the last dimension is time or space), state your assumption explicitly before modifying code.

## Development Workflow
1. Trace the input tensor shapes through the `TBMD.core` modules before proposing fixes.
2. Rely on the `TBMD.config` dataclasses. Do not introduce hardcoded configuration dictionaries or environment variables.

## Documentation Rules
- Prioritize technical accuracy.
- When writing Python docstrings, adhere strictly to the established formatting in the surrounding code.

## Testing and Validation
Verify your changes by running:
```bash
pytest
```
If your changes affect file structures, run:
```bash
pytest tests/audit -q
```

## Safety and Correctness Constraints
- Do not remove existing docstrings or inline comments unless they are directly invalidated by your code change.
- Never label the system as "production-ready".

## Final Response Format
Format your final response with a concise summary of the diffs made and the validation commands you successfully executed.
