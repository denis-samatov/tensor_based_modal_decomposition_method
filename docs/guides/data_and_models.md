# Model and Data Handling

## Data

Local datasets should be stored under `data/`. This directory is ignored by git because datasets can be large, private, or derived from external sources.

When documenting an experiment, record:

- dataset name and version;
- local path convention;
- tensor shape and axis meaning;
- preprocessing steps;
- train/test split policy.

## Generated Results

Generated figures, metrics, model checkpoints, and sweep outputs should stay under ignored output directories such as:

- `results/`
- `scripts/plots/`
- `output/`
- `outputs/`

Commit generated artifacts only when they are intentionally curated documentation assets and are small enough for normal repository review.

Experiment scripts may create files under `scripts/plots/models_eval/`. Those files are generated artifacts, not source files. Regenerate them by running the relevant script with the required local dataset and configuration.

## Model Artifacts

Trained models, `.npz` checkpoints, and intermediate sweep outputs are local artifacts by default. Do not commit them unless the repository maintainers explicitly decide to version a small reproducible fixture.

## Privacy

Do not commit:

- credentials or tokens;
- private simulator outputs;
- proprietary datasets;
- local absolute paths in stable documentation;
- personal contact details that are not intended for public project metadata.
