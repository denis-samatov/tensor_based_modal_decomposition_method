# Navier-Stokes Model Examples And Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible qualitative examples for every Navier-Stokes model, centralize benchmark model definitions, and retune the final `LSTM` and `MLP` presets without breaking the trajectory-aware protocol.

**Architecture:** Keep the current trajectory-aware forecasting core in `navier_stokes_forecasting.py`, but stop growing that file further. Add one shared model-registry module plus one example-generation module under `src/TBMD/experiments`, and make both benchmark/report entrypoints consume those helpers. Run a controlled preset search for `LSTM` and `MLP`, freeze the winning configs in the registry, then regenerate metrics, examples, and the report from those frozen configs.

**Tech Stack:** Python, NumPy, Matplotlib, imageio/Pillow-compatible GIF writing, pytest, existing TBMD configs and forecasters.

---

### Task 1: Define the file structure and shared registry boundary

**Files:**
- Create: `src/TBMD/experiments/navier_stokes_model_registry.py`
- Create: `src/TBMD/experiments/navier_stokes_examples.py`
- Modify: `src/TBMD/experiments/__init__.py`
- Reference: `src/TBMD/experiments/navier_stokes_forecasting.py`

- [ ] **Step 1: Write the failing test**

Add tests that assert the registry returns the expected model names/slugs and that example selection is deterministic.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_navier_stokes_pipeline.py -k "registry or manifest or selection" -v`
Expected: FAIL because the registry and example helpers do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create a small registry module that exposes:
- a model spec structure with `name`, `slug`, and config factory;
- a `get_navier_stokes_model_specs()` helper;
- deterministic default trajectory/step selection helpers.

Create a separate examples module that holds artifact-generation helpers rather than growing `navier_stokes_forecasting.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_navier_stokes_pipeline.py -k "registry or selection" -v`
Expected: PASS

### Task 2: Add failing tests for qualitative artifact metadata

**Files:**
- Modify: `tests/unit/test_navier_stokes_pipeline.py`
- Create/Modify: `src/TBMD/experiments/navier_stokes_examples.py`

- [ ] **Step 1: Write the failing test**

Add tests for:
- manifest structure;
- frame-path naming;
- compact synthetic contact-sheet generation.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_navier_stokes_pipeline.py -k "manifest or contact_sheet or frame" -v`
Expected: FAIL because those helpers are not implemented.

- [ ] **Step 3: Write minimal implementation**

Implement helpers that:
- build `manifest.json` payloads;
- generate deterministic frame filenames;
- render a compact contact sheet from synthetic target/prediction arrays.

Keep these helpers pure where possible so unit tests stay fast.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_navier_stokes_pipeline.py -k "manifest or contact_sheet or frame" -v`
Expected: PASS

### Task 3: Move benchmark model construction to the shared registry

**Files:**
- Modify: `notebooks/experiments/evaluate_all_models.py`
- Modify: `src/TBMD/experiments/__init__.py`
- Create/Modify: `src/TBMD/experiments/navier_stokes_model_registry.py`

- [ ] **Step 1: Write the failing test**

Add a regression test that validates the registry contains the four benchmarked models and that the benchmark script can derive model specs without duplicating inline config definitions.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_navier_stokes_pipeline.py -k "model_specs" -v`
Expected: FAIL because the registry-driven contract is not wired yet.

- [ ] **Step 3: Write minimal implementation**

Refactor `evaluate_all_models.py` to:
- import the shared registry;
- instantiate models from registry entries;
- keep current metrics output contract intact.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_navier_stokes_pipeline.py -k "model_specs" -v`
Expected: PASS

### Task 4: Add the dedicated example generator entrypoint

**Files:**
- Create: `notebooks/experiments/generate_model_examples.py`
- Create/Modify: `src/TBMD/experiments/navier_stokes_examples.py`
- Create/Modify: `src/TBMD/experiments/navier_stokes_model_registry.py`

- [ ] **Step 1: Write the failing test**

Add smoke tests that exercise the example helpers on a tiny synthetic trajectory batch and verify:
- contact sheet path creation;
- frame list creation;
- manifest payload contents.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_navier_stokes_pipeline.py -k "examples" -v`
Expected: FAIL because the entrypoint/helpers do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Implement the generator so it:
- loads trajectory-aware Navier-Stokes data;
- fits the registry models;
- evaluates rollout predictions;
- writes per-model contact sheets, GIFs, frame dumps, comparison sheets, and `manifest.json` under `notebooks/experiments/plots/models_eval/examples/`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_navier_stokes_pipeline.py -k "examples" -v`
Expected: PASS

### Task 5: Run the controlled preset search for `LSTM` and `MLP`

**Files:**
- Modify: `src/TBMD/experiments/navier_stokes_model_registry.py`
- Modify: `notebooks/experiments/evaluate_all_models.py`
- Optional helper edits: `src/TBMD/experiments/navier_stokes_examples.py`

- [ ] **Step 1: Establish the search matrix**

Freeze a small manual sweep only for:
- `LSTM`: `hidden_size=128`, `seq_length in {6, 7, 8}`, `epochs in {150, 200}`, lower LR if needed;
- `MLP`: `hidden_size=128`, `num_layers in {2, 3}`, `delta_forecast=False`.

- [ ] **Step 2: Run the benchmark search on the real dataset**

Run targeted benchmark commands and record the best final preset per family using the official test trajectories.

- [ ] **Step 3: Promote only the winning presets**

Update the shared registry so the benchmark and example generator both use the final chosen configs and never intermediate search configs.

- [ ] **Step 4: Re-run the final benchmark**

Run: `python3 notebooks/experiments/evaluate_all_models.py`
Expected: fresh `metrics_summary.json` and `model_comparison_chart.png` using the final shared registry.

### Task 6: Generate final example artifacts and refresh the report

**Files:**
- Modify: `all_models_report.md`
- Create/Modify: `notebooks/experiments/generate_model_examples.py`
- Output: `notebooks/experiments/plots/models_eval/examples/**`

- [ ] **Step 1: Run the example generator**

Run: `python3 notebooks/experiments/generate_model_examples.py`
Expected: per-model `contact_sheet.png`, `rollout.gif`, frame dumps, comparison sheets, and `manifest.json`.

- [ ] **Step 2: Refresh the report**

Update `all_models_report.md` to:
- reflect the final tuned benchmark metrics;
- state which presets won;
- reference the generated qualitative artifact directory.

- [ ] **Step 3: Run final verification**

Run:
- `pytest tests/unit/test_navier_stokes_pipeline.py -q`
- `python3 -m py_compile src/TBMD/experiments/navier_stokes_model_registry.py src/TBMD/experiments/navier_stokes_examples.py notebooks/experiments/evaluate_all_models.py notebooks/experiments/generate_model_examples.py`

Expected: PASS
