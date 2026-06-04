# Brugge Digital Twin Analysis Notes

This page documents the intended Brugge digital twin workflow and the checks maintainers should perform when local Brugge data is available.

The repository does not include a reproducible public Brugge benchmark fixture. Avoid treating any historical metric in local reports as a general project claim unless the data source, split, configuration, and command are recorded.

## Script

```bash
python examples/applications/brugge_field/run_brugge_enhanced.py
```

## Expected Inputs

The script is dataset-specific and may require local files under `data/`. Verify the actual paths in the script before running it.

Typical inputs include:

- reservoir state tensors;
- well coordinates or well-control metadata;
- train/test split settings;
- normalization parameters derived from training data only.

## Validation Checklist

When running the workflow, record:

1. Dataset version and local file paths.
2. Selected case or trajectory identifiers.
3. Tensor shape and time-axis interpretation.
4. Train/test split policy.
5. Normalization method and whether statistics were fit on training data only.
6. TBMD ranks, sensor count, and forecaster/proxy configuration.
7. Reconstruction and forecast metrics.
8. Any warnings about unrealistic well controls or mass imbalance.

## Known Risks

- Synthetic or placeholder well controls can make scenario-analysis metrics misleading.
- A low decomposition reconstruction error does not guarantee accurate multi-step forecasts.
- Dataset paths and formats may differ between local environments.
- Generated figures and model artifacts should stay out of git unless they are intentionally curated documentation assets.

## Recommended Reporting Format

Use a neutral report table:

| Item | Value |
| --- | --- |
| Command | `python examples/applications/brugge_field/run_brugge_enhanced.py` |
| Dataset | Not specified in the repository; record the local dataset version. |
| Case | Not specified; record selected case or trajectory identifiers. |
| Train/test split | Not specified; record the split policy. |
| Configuration | Not specified; record ranks, sensors, forecaster, and proxy settings. |
| Metrics | Not specified; record reconstruction and forecast metrics from the run. |
| Limitations | Not specified; record data, model, and scenario limitations. |

Do not publish accuracy, speed, or deployment-readiness claims without a reproducible command and dataset description.
