# Limitations and Data Handling

## Purpose
To outline the known limitations of the TBMD library, explain how local datasets are managed, and clarify the privacy and security expectations.

## Audience
Product managers, developers, and ML engineers evaluating the suitability of TBMD for their data and scenarios.

## Summary
TBMD is an experimental research codebase. Claims regarding accuracy, performance, or "production-readiness" should be treated as `Owner decision required` unless explicitly backed by a reproducible benchmark. Local datasets and trained models must remain uncommitted.

## Details

### Experimental Nature
The repository is aimed at scientific computing and reservoir-modeling experiments. It should be treated as a research and engineering codebase rather than a validated production simulator.

### Model Performance and Accuracy
> Documentation conflict / Policy note: Any claims regarding prediction accuracy, system scalability, or being "best-in-class" require explicit verification.

### Known Algorithm Constraints
1. **Computational overhead in offline phase**: Finding optimal multi-dimensional ranks for the Tucker Decomposition requires multiple dense SVD operations, which are prohibitively expensive on extremely high-resolution grids without batched processing.
2. **Compressive Sensing assumptions**: The ADMM solver assumes that the underlying physical field can be accurately represented by a sparse combination of the learned basis modes. Highly chaotic or purely random fields will result in poor reconstruction accuracy.
4. **Scalability Constraints**: The maximum tested grid size for the offline phase corresponds to the downscaled Brugge benchmark resolution. Processing significantly larger grids (e.g., multi-million cells) requires batched SVD approaches or memory-mapped tensors.

If testing complex forecasters (like `mlp` or `lstm`), they require dataset-specific validation. Simple forecasters (like `linear` or `persistence`) are primarily for smoke tests and synthetic examples.

### Local Data Handling
Local datasets should be stored under the `data/` directory. This directory is ignored by git because datasets can be large, private, or derived from external sources. When documenting an experiment, record the dataset name, local path convention, tensor shape, preprocessing steps, and train/test split policy.

### Generated Results and Artifacts
Generated figures, metrics, model checkpoints (`.npz`), and sweep outputs must stay under ignored output directories such as `results/` or `scripts/plots/`. Do not commit them unless they are small, curated documentation assets.

### Privacy and Security
Do not commit:
- Credentials, tokens, or API keys
- Private simulator outputs or proprietary datasets
- `.env` files
- Local absolute paths in stable documentation

## Validation
To verify that no private data or generated results are being tracked by git, run:
```bash
git status --ignored
```
Expected result: The `data/`, `results/`, and `scripts/plots/` directories (if they contain files) should appear under the ignored files list, ensuring they are not staged for commit.

## Related docs
- [Product Overview](overview.md)
- [Testing Guide](../development/testing.md)
