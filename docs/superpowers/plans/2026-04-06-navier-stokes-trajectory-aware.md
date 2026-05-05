# Navier-Stokes Trajectory-Aware Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Navier-Stokes-specific loader/evaluation pipeline that respects trajectory boundaries and reports trustworthy one-step and rollout metrics.

**Architecture:** Add one experiment-layer module under `src/TBMD/experiments` to own trajectory reconstruction, latent projection, explicit sub-forecaster training, and evaluation. Update the existing example/report scripts to call this module instead of flattening `train/inputs.npy` into a fake single sequence.

**Tech Stack:** Python, NumPy, TensorLy, PyTorch, pytest, existing TBMD forecasters/configs.

---

### Task 1: Add failing tests for dataset interpretation

**Files:**
- Create: `tests/unit/test_navier_stokes_pipeline.py`
- Test: `tests/unit/test_navier_stokes_pipeline.py`

- [ ] **Step 1: Write the failing test**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement minimal loader/state-stitching code**
- [ ] **Step 4: Run test to verify it passes**

### Task 2: Add failing tests for trajectory-safe training samples

**Files:**
- Modify: `tests/unit/test_navier_stokes_pipeline.py`
- Create/Modify: `src/TBMD/experiments/navier_stokes_forecasting.py`

- [ ] **Step 1: Write the failing test**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement explicit pair/window builders without cross-trajectory leakage**
- [ ] **Step 4: Run test to verify it passes**

### Task 3: Add failing tests for official-test evaluation contract

**Files:**
- Modify: `tests/unit/test_navier_stokes_pipeline.py`
- Modify: `src/TBMD/experiments/navier_stokes_forecasting.py`

- [ ] **Step 1: Write the failing test**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement one-step and rollout evaluation helpers**
- [ ] **Step 4: Run test to verify it passes**

### Task 4: Wire scripts to the new module

**Files:**
- Modify: `examples/02_navier_stokes_optimal_forecasting.py`
- Modify: `notebooks/experiments/evaluate_all_models.py`
- Modify: `notebooks/experiments/latent_modal_visualize.py`

- [ ] **Step 1: Update scripts to use trajectory-aware loader/evaluator**
- [ ] **Step 2: Keep outputs backward-compatible where practical**
- [ ] **Step 3: Run targeted verification commands**

### Task 5: Refresh report wording

**Files:**
- Modify: `all_models_report.md`

- [ ] **Step 1: Update report assumptions and evaluation wording**
- [ ] **Step 2: Keep old numerical claims only if re-verified**

