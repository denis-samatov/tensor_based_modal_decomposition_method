# Code Style

## Purpose
Establishes the code style rules for contributing to the repository.

## Audience
Developers and Researchers modifying the `src/` directory.

## Summary
The project follows standard Python data science guidelines, emphasizing readability, type hinting, and consistent docstrings.

## Details
- **Formatting**: Use standard formatters (like `black` or `ruff`) if available.
- **Type Hinting**: All functions in `src/TBMD/` should have type hints (e.g., `Tensor`, `np.ndarray`).
- **Docstrings**: Use Google-style docstrings. Document the expected shapes of all tensor inputs and outputs.
- **Imports**: Organize imports logically. Absolute imports are preferred over relative imports for external modules.

## Validation
Always ensure the code passes compilation checks before committing:
```bash
python -m compileall .
```
