# Navier-Stokes T+1 Residual Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an experiment-layer residual correction head that improves TBMD one-step (`t+1`) prediction for Navier-Stokes without changing the shared forecasting API.

**Architecture:** Keep the current trajectory-aware latent forecaster as the base stage, then train a lightweight MLP correction head on top of its one-step latent residuals. The wrapper exposes the same `fit`, `evaluate_one_step`, and `evaluate_rollout` interface as existing experiment forecasters so scripts and registry integration stay simple.

**Tech Stack:** Python, NumPy, PyTorch, existing `MLPForecaster`/`LSTMForecaster`, pytest

---

### Task 1: Add failing tests for residual correction helpers and wrapper contract

**Files:**
- Modify: `tests/unit/test_navier_stokes_pipeline.py`
- Test: `tests/unit/test_navier_stokes_pipeline.py`

- [ ] **Step 1: Write the failing tests**
- [ ] **Step 2: Run `pytest tests/unit/test_navier_stokes_pipeline.py -q` to verify the new tests fail**
- [ ] **Step 3: Implement only the missing helper/wrapper behavior required by the failing tests**
- [ ] **Step 4: Re-run `pytest tests/unit/test_navier_stokes_pipeline.py -q`**

### Task 2: Implement the residual correction wrapper in the experiment layer

**Files:**
- Modify: `src/TBMD/experiments/navier_stokes_forecasting.py`
- Modify: `src/TBMD/experiments/__init__.py`
- Test: `tests/unit/test_navier_stokes_pipeline.py`

- [ ] **Step 1: Add helper functions for correction feature/target construction**
- [ ] **Step 2: Add a wrapper forecaster that fits base TBMD first and correction head second**
- [ ] **Step 3: Extend one-step and rollout evaluation to expose corrected predictions**
- [ ] **Step 4: Re-run focused tests**

### Task 3: Integrate the corrected model into the experiment registry and scripts

**Files:**
- Modify: `src/TBMD/experiments/navier_stokes_model_registry.py`
- Modify: `scripts/evaluate_all_models.py`
- Modify: `scripts/generate_model_examples.py`
- Test: `tests/unit/test_navier_stokes_pipeline.py`

- [ ] **Step 1: Add a dedicated model spec for the corrected one-step model**
- [ ] **Step 2: Ensure scripts can benchmark/render it without special cases**
- [ ] **Step 3: Add/adjust tests for registry exposure if needed**
- [ ] **Step 4: Re-run tests**

### Task 4: Verify on real data and update artifacts if the corrected model helps

**Files:**
- Modify: `all_models_report.md`
- Modify: `walkthrough.md`
- Modify: `task_plan.md`
- Test: `scripts/evaluate_all_models.py`

- [ ] **Step 1: Run the benchmark on real Navier-Stokes data**
- [ ] **Step 2: Compare baseline LSTM vs corrected model on one-step metrics**
- [ ] **Step 3: Regenerate example artifacts if the corrected model is kept**
- [ ] **Step 4: Update reports with evidence-backed numbers**
