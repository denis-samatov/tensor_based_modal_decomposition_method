# Navier-Stokes Model Examples And Tuning Design

Date: 2026-04-06

## Context

The Navier-Stokes experiment pipeline has already been corrected to use trajectory-aware train and test splits. The benchmark now reports honest one-step and rollout metrics on the official test trajectories, but two gaps remain:

1. Qualitative comparison is weak. The current benchmark only emits a few isolated `pred_t*.png` images per model, which is not enough for consistent side-by-side inspection.
2. The latest trajectory-aware pipeline improved `LSTM` and stabilized `MLP`, but the tuning logic and visualization logic are still mixed into the benchmark script.

The next iteration should improve model quality further and add reproducible example generation for every model without creating another legacy evaluation path.

## Goals

- Keep a single trajectory-aware Navier-Stokes protocol for metrics and qualitative examples.
- Generate reproducible qualitative artifacts for every benchmarked model.
- Improve the final `LSTM` and `MLP` presets through a controlled tuning pass.
- Keep the benchmark report and the visual examples aligned to the same trained model configurations.

## Non-Goals

- No changes to the shared forecasting APIs outside the Navier-Stokes experiment layer.
- No notebook-only manual workflow for final artifacts.
- No per-run manual selection of "nice-looking" trajectories.
- No changes to unrelated datasets or legacy experiment families outside the current Navier-Stokes scope.

## Design Summary

The implementation will separate three responsibilities:

1. `model registry`
   A single source of truth for Navier-Stokes benchmark model specs.
2. `benchmark runner`
   Produces metrics and summary reports from the registry.
3. `example generator`
   Produces contact sheets, frame dumps, animations, and a manifest from the same registry and the same evaluation protocol.

This avoids configuration drift between reported metrics and qualitative examples.

## Architecture

### 1. Model Registry

Add a small experiment-scoped registry helper that returns the final benchmark model set and configuration objects. This registry must be used by both the benchmark runner and the example generator.

Expected models:

- `Linear Forecaster`
- `MLP Forecaster`
- `LSTM Forecaster`
- `Multi-Resolution Linear`

The registry is also the place where final tuned presets are frozen after the controlled search.

### 2. Benchmark Runner

`notebooks/experiments/evaluate_all_models.py` remains the source of truth for quantitative evaluation, but it should stop inlining model construction logic. It should instead consume the shared model registry, run the same honest trajectory-aware evaluation protocol, and write:

- `metrics_summary.json`
- `model_comparison_chart.png`
- minimal per-model benchmark plots if still useful

### 3. Example Generator

Add a dedicated script:

- `notebooks/experiments/generate_model_examples.py`

This script will:

- load the same train/test trajectories as the benchmark runner;
- instantiate the same model presets via the shared registry;
- fit and evaluate each model with the same protocol;
- generate reproducible qualitative artifacts for fixed test trajectories and fixed rollout steps.

## Artifact Layout

Output root:

- `notebooks/experiments/plots/models_eval/examples/`

Per model:

- `examples/<model_slug>/contact_sheet.png`
- `examples/<model_slug>/rollout.gif`
- `examples/<model_slug>/frames/frame_tXX.png`

Cross-model comparison:

- `examples/comparison/fixed_trajectory_<k>.png`

Machine-readable metadata:

- `examples/manifest.json`

The manifest must include:

- model name and slug;
- selected trajectory indices;
- selected rollout step indices;
- artifact paths;
- summary metrics for the same run;
- image/gif generation settings.

## Qualitative Example Policy

Default qualitative examples must use fixed test trajectories instead of per-model best/worst picks.

Reasoning:

- fixed trajectories keep the comparison honest;
- every model sees the exact same test case;
- the report can directly reference those examples without ambiguity.

The first iteration should use a small fixed set of trajectory indices and rollout steps chosen deterministically in code.

## Quality Improvement Plan

The next quality pass will be controlled and narrow:

- keep `Linear` and `MR Linear` as baselines;
- retune `LSTM` around:
  - `hidden_size=128`
  - `seq_length=6..8`
  - `num_epochs=150..200`
  - lower learning rate if needed;
- retune `MLP` with `delta_forecast=False` around:
  - `hidden_size=128`
  - `num_layers=2..3`
  - keep latent normalization and spatial mean centering enabled.

Only the winning preset per model family should be promoted into the shared model registry. The example generator must never use exploratory or intermediate presets.

## Testing Strategy

Use test-first changes for new helper behavior.

Required tests:

- unit test for deterministic fixed trajectory and fixed step selection;
- unit test for manifest generation;
- smoke test for contact sheet generation on a tiny synthetic dataset;
- smoke test for generated frame paths;
- keep existing Navier-Stokes trajectory-aware tests green.

Verification after implementation:

- run the updated Navier-Stokes unit test file;
- run the benchmark script end-to-end;
- run the example generator end-to-end;
- confirm that `all_models_report.md` reflects the latest final benchmark metrics and references the new qualitative artifacts where appropriate.

## Risks And Safeguards

### Risk: protocol drift

If the benchmark and the example generator build models separately, the qualitative outputs can silently stop matching the reported metrics.

Safeguard:

- use one shared registry helper for both entrypoints.

### Risk: visual artifacts become expensive or noisy

Too many trajectories or too many frames will produce slow and cluttered outputs.

Safeguard:

- start with a fixed, compact default set of trajectories and steps;
- keep frame generation deterministic and bounded.

### Risk: tuning results are overfit to local experimentation

Exploratory preset changes can leak into the report without a clean comparison baseline.

Safeguard:

- evaluate only final candidate presets on the full official test set;
- freeze the chosen presets in the shared model registry;
- regenerate both report metrics and examples from those frozen presets.

## Acceptance Criteria

- There is one shared Navier-Stokes model registry used by both metrics and examples.
- `evaluate_all_models.py` no longer inlines final model definitions.
- A dedicated example generator produces `contact_sheet`, `gif`, `frames`, comparison sheets, and `manifest.json`.
- The default examples use fixed official test trajectories.
- `LSTM` and `MLP` receive one controlled tuning pass and the final chosen presets are reflected in the benchmark output.
- `all_models_report.md` is updated to the latest final benchmark results and the new qualitative artifacts.
