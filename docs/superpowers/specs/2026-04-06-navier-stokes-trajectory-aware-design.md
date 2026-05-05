# Navier-Stokes Trajectory-Aware Evaluation Design

## Goal
Make the Navier-Stokes forecasting experiments trajectory-aware without changing the shared forecaster APIs.

## Scope
- Fix only the Navier-Stokes loader, evaluation, visualization, and report pipeline.
- Keep `LatentModalForecaster` and `MultiResolutionTBMDForecaster` public APIs unchanged.
- Regenerate metrics against the official dataset split instead of an internal temporal split over flattened train samples.

## Confirmed Dataset Facts
- `train/inputs.npy` and `train/labels.npy` are flattened transition pairs with shape `(19000, 64, 64, 1, 1)`.
- Transition continuity holds inside trajectories, and breaks every `19` train samples.
- `test/inputs.npy` and `test/labels.npy` already preserve trajectory structure with shape `(200, 19, 64, 64, 1, 1)`.
- `labels[t]` is the next state for `inputs[t]`.

## Design
1. Add a Navier-Stokes-specific experiment module that:
   - loads `inputs.npy` and `labels.npy`,
   - restores train transitions into explicit trajectories,
   - stitches each trajectory into full state sequences,
   - prepares trajectory-safe training pairs and rollout seeds.
2. Reuse Tucker decomposition/projection from `LatentModalForecaster`, but perform training/evaluation at the experiment layer so cross-trajectory transitions never enter training.
3. Report two evaluation modes:
   - `one_step`: predict next state from ground-truth current state,
   - `rollout`: autoregressive multi-step prediction over each official test trajectory.
4. Update the existing Navier-Stokes example/report scripts to consume this new module.

## Non-Goals
- No changes to generic forecasting APIs.
- No physics-informed losses in this pass.
- No dataset augmentation in this pass.

## Risks
- The new trajectory-aware leaderboard may differ materially from the current report.
- LSTM/MLP training time may increase slightly because training windows are built explicitly instead of inferred from one flat history.
