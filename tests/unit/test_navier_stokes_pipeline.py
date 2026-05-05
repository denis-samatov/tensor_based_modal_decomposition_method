from pathlib import Path

import numpy as np
import pytest
import torch

from TBMD.config import LatentModalForecasterConfig
from TBMD.experiments.navier_stokes_examples import (
    build_examples_manifest,
    make_frame_filename,
    save_t_plus_one_diagnostics_sheet,
    save_contact_sheet,
    select_fixed_rollout_steps,
    select_fixed_trajectory_indices,
)
from TBMD.experiments.navier_stokes_forecasting import (
    TrajectoryAwareLatentForecaster,
    TrajectoryAwareResidualCorrectedForecaster,
    _build_t_plus_one_correction_features,
    _apply_latent_standardization,
    _compute_mixed_one_step_loss_terms,
    _compute_latent_standardization_stats,
    _invert_latent_standardization,
    _split_trajectory_series_for_validation,
    build_lagged_windows,
    build_one_step_pairs,
    load_navier_stokes_trajectory_dataset,
    reshape_flattened_train_transitions,
    stitch_inputs_and_labels_to_states,
)
from TBMD.experiments.navier_stokes_model_registry import get_navier_stokes_model_specs


def test_reshape_flattened_train_transitions_restores_trajectory_axis():
    flat_inputs = np.arange(6, dtype=np.float32).reshape(6, 1, 1)
    flat_labels = np.arange(1, 7, dtype=np.float32).reshape(6, 1, 1)

    train_inputs, train_labels = reshape_flattened_train_transitions(
        flat_inputs,
        flat_labels,
        trajectory_length=3,
    )

    assert train_inputs.shape == (2, 3, 1, 1)
    assert train_labels.shape == (2, 3, 1, 1)
    np.testing.assert_array_equal(train_inputs[:, :, 0, 0], [[0, 1, 2], [3, 4, 5]])
    np.testing.assert_array_equal(train_labels[:, :, 0, 0], [[1, 2, 3], [4, 5, 6]])


def test_stitch_inputs_and_labels_to_states_builds_full_state_sequence():
    train_inputs = np.array([[[[0.0]], [[1.0]], [[2.0]]]], dtype=np.float32)
    train_labels = np.array([[[[1.0]], [[2.0]], [[3.0]]]], dtype=np.float32)

    states = stitch_inputs_and_labels_to_states(train_inputs, train_labels)

    assert states.shape == (1, 4, 1, 1)
    np.testing.assert_array_equal(states[0, :, 0, 0], [0.0, 1.0, 2.0, 3.0])


def test_stitch_inputs_and_labels_to_states_rejects_broken_transition_chain():
    train_inputs = np.array([[[[0.0]], [[1.0]], [[2.0]]]], dtype=np.float32)
    broken_labels = np.array([[[[1.0]], [[9.0]], [[3.0]]]], dtype=np.float32)

    with pytest.raises(ValueError, match="transition continuity"):
        stitch_inputs_and_labels_to_states(train_inputs, broken_labels)


def test_build_one_step_pairs_flattens_only_within_trajectories():
    states = np.array(
        [
            [[0.0], [1.0], [2.0], [3.0]],
            [[10.0], [11.0], [12.0], [13.0]],
        ],
        dtype=np.float32,
    )

    x_pairs, y_pairs = build_one_step_pairs(states)

    np.testing.assert_array_equal(x_pairs[:, 0], [0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
    np.testing.assert_array_equal(y_pairs[:, 0], [1.0, 2.0, 3.0, 11.0, 12.0, 13.0])


def test_build_one_step_pairs_can_return_delta_targets():
    states = np.array(
        [
            [[1.0], [3.0], [6.0], [10.0]],
        ],
        dtype=np.float32,
    )

    x_pairs, y_pairs = build_one_step_pairs(states, predict_deltas=True)

    np.testing.assert_array_equal(x_pairs[:, 0], [1.0, 3.0, 6.0])
    np.testing.assert_array_equal(y_pairs[:, 0], [2.0, 3.0, 4.0])


def test_build_lagged_windows_never_crosses_trajectory_boundaries():
    states = np.array(
        [
            [[0.0], [1.0], [2.0], [3.0]],
            [[10.0], [11.0], [12.0], [13.0]],
        ],
        dtype=np.float32,
    )

    windows, targets = build_lagged_windows(states, seq_length=2)

    assert windows.shape == (4, 2, 1)
    assert targets.shape == (4, 1)
    np.testing.assert_array_equal(windows[:, :, 0], [[0.0, 1.0], [1.0, 2.0], [10.0, 11.0], [11.0, 12.0]])
    np.testing.assert_array_equal(targets[:, 0], [2.0, 3.0, 12.0, 13.0])


def test_build_lagged_windows_can_return_delta_targets():
    states = np.array(
        [
            [[1.0], [3.0], [6.0], [10.0]],
        ],
        dtype=np.float32,
    )

    windows, targets = build_lagged_windows(states, seq_length=2, predict_deltas=True)

    np.testing.assert_array_equal(windows[:, :, 0], [[1.0, 3.0], [3.0, 6.0]])
    np.testing.assert_array_equal(targets[:, 0], [3.0, 4.0])


def test_split_trajectory_series_for_validation_keeps_trajectory_boundaries():
    states = np.arange(4 * 3, dtype=np.float32).reshape(4, 3, 1)

    train_states, val_states = _split_trajectory_series_for_validation(states, val_split=0.25)

    assert train_states.shape == (3, 3, 1)
    assert val_states.shape == (1, 3, 1)
    np.testing.assert_array_equal(train_states[:, :, 0], [[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    np.testing.assert_array_equal(val_states[:, :, 0], [[9, 10, 11]])


def test_latent_standardization_roundtrip_restores_original_values():
    latent = np.array(
        [
            [[1.0, 10.0], [3.0, 20.0]],
            [[5.0, 30.0], [7.0, 40.0]],
        ],
        dtype=np.float32,
    )

    mean, std = _compute_latent_standardization_stats(latent)
    normalized = _apply_latent_standardization(latent, mean, std)
    restored = _invert_latent_standardization(normalized, mean, std)

    np.testing.assert_allclose(restored, latent)
    np.testing.assert_allclose(normalized.reshape(-1, 2).mean(axis=0), [0.0, 0.0], atol=1e-7)


def test_build_t_plus_one_correction_features_concatenates_state_prediction_and_delta():
    last_state = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )
    baseline_pred = np.array(
        [
            [5.0, 7.0],
            [11.0, 13.0],
        ],
        dtype=np.float32,
    )

    features = _build_t_plus_one_correction_features(last_state, baseline_pred)

    np.testing.assert_array_equal(
        features,
        np.array(
            [
                [1.0, 2.0, 5.0, 7.0, 4.0, 5.0],
                [3.0, 4.0, 11.0, 13.0, 8.0, 9.0],
            ],
            dtype=np.float32,
        ),
    )


def test_compute_mixed_one_step_loss_terms_are_zero_for_perfect_prediction():
    pred_residual = torch.tensor([[0.5, -0.25]], dtype=torch.float32)
    target_residual = torch.tensor([[0.5, -0.25]], dtype=torch.float32)
    pred_spatial = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
    target_spatial = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)

    losses = _compute_mixed_one_step_loss_terms(
        pred_residual_normalized=pred_residual,
        target_residual_normalized=target_residual,
        pred_spatial=pred_spatial,
        target_spatial=target_spatial,
        latent_loss_weight=1.0,
        spatial_loss_weight=0.25,
        rel_frob_loss_weight=0.1,
    )

    assert losses["total"].item() == pytest.approx(0.0)
    assert losses["latent"].item() == pytest.approx(0.0)
    assert losses["spatial"].item() == pytest.approx(0.0)
    assert losses["rel_frob"].item() == pytest.approx(0.0)


def test_load_navier_stokes_trajectory_dataset_restores_train_and_preserves_test(tmp_path):
    root = Path(tmp_path)
    (root / "train").mkdir()
    (root / "test").mkdir()

    train_states = np.array(
        [
            [[1.0], [2.0], [4.0], [8.0]],
            [[3.0], [6.0], [12.0], [24.0]],
        ],
        dtype=np.float32,
    ).reshape(2, 4, 1, 1)
    test_states = np.array(
        [
            [[5.0], [10.0], [20.0], [40.0]],
        ],
        dtype=np.float32,
    ).reshape(1, 4, 1, 1)

    np.save(root / "train" / "inputs.npy", train_states[:, :-1, :, :, None, None].reshape(6, 1, 1, 1, 1))
    np.save(root / "train" / "labels.npy", train_states[:, 1:, :, :, None, None].reshape(6, 1, 1, 1, 1))
    np.save(root / "test" / "inputs.npy", test_states[:, :-1, :, :, None, None])
    np.save(root / "test" / "labels.npy", test_states[:, 1:, :, :, None, None])

    dataset = load_navier_stokes_trajectory_dataset(root, trajectory_length=3)

    assert dataset.train_inputs.shape == (2, 3, 1, 1)
    assert dataset.train_states.shape == (2, 4, 1, 1)
    assert dataset.test_inputs.shape == (1, 3, 1, 1)
    assert dataset.test_states.shape == (1, 4, 1, 1)
    np.testing.assert_array_equal(dataset.test_states[0, :, 0, 0], [5.0, 10.0, 20.0, 40.0])


def test_trajectory_aware_linear_forecaster_scores_perfect_on_geometric_sequences():
    train_states = np.array(
        [
            [[1.0], [2.0], [4.0], [8.0]],
            [[3.0], [6.0], [12.0], [24.0]],
        ],
        dtype=np.float32,
    ).reshape(2, 4, 1, 1)
    test_states = np.array(
        [
            [[5.0], [10.0], [20.0], [40.0]],
        ],
        dtype=np.float32,
    ).reshape(1, 4, 1, 1)

    forecaster = TrajectoryAwareLatentForecaster(
        config=LatentModalForecasterConfig(
            ranks=[1, 1, 1],
            forecaster_type="linear",
            spatial_mean_centering=False,
            latent_normalization=False,
            delta_forecast=False,
            verbose=False,
        )
    )
    forecaster.fit(train_states)

    one_step = forecaster.evaluate_one_step(test_states)
    rollout = forecaster.evaluate_rollout(test_states)

    assert one_step["n_eval_samples"] == 3
    assert rollout["n_rollout_steps"] == 3
    assert one_step["spatial_r2"] > 0.999999
    assert rollout["spatial_r2"] > 0.999999


def test_trajectory_aware_linear_forecaster_supports_latent_plus_delta_features():
    train_states = np.array(
        [
            [[1.0], [2.0], [4.0], [8.0], [16.0]],
            [[3.0], [6.0], [12.0], [24.0], [48.0]],
        ],
        dtype=np.float32,
    ).reshape(2, 5, 1, 1)
    test_states = np.array(
        [
            [[5.0], [10.0], [20.0], [40.0], [80.0]],
        ],
        dtype=np.float32,
    ).reshape(1, 5, 1, 1)

    forecaster = TrajectoryAwareLatentForecaster(
        config=LatentModalForecasterConfig(
            ranks=[1, 1, 1],
            forecaster_type="linear",
            spatial_mean_centering=False,
            latent_normalization=False,
            delta_forecast=False,
            verbose=False,
        ),
        feature_mode="latent_plus_delta",
    )
    forecaster.fit(train_states)

    one_step = forecaster.evaluate_one_step(test_states)
    rollout = forecaster.evaluate_rollout(test_states)

    assert one_step["n_eval_samples"] == 4
    assert rollout["n_rollout_steps"] == 4
    assert one_step["spatial_r2"] > 0.999999
    assert rollout["spatial_r2"] > 0.999999


def test_residual_corrected_forecaster_with_disabled_head_matches_baseline_one_step():
    train_states = np.array(
        [
            [[1.0], [2.0], [4.0], [8.0], [16.0]],
            [[3.0], [6.0], [12.0], [24.0], [48.0]],
        ],
        dtype=np.float32,
    ).reshape(2, 5, 1, 1)
    test_states = np.array(
        [
            [[5.0], [10.0], [20.0], [40.0], [80.0]],
        ],
        dtype=np.float32,
    ).reshape(1, 5, 1, 1)

    config = LatentModalForecasterConfig(
        ranks=[1, 1, 1],
        forecaster_type="linear",
        spatial_mean_centering=False,
        latent_normalization=False,
        delta_forecast=False,
        verbose=False,
    )

    baseline = TrajectoryAwareLatentForecaster(config=config, feature_mode="latent_plus_delta")
    baseline.fit(train_states)
    baseline_one_step = baseline.evaluate_one_step(test_states)

    corrected = TrajectoryAwareResidualCorrectedForecaster(
        config=config,
        feature_mode="latent_plus_delta",
        correction_num_epochs=0,
    )
    corrected.fit(train_states)
    corrected_one_step = corrected.evaluate_one_step(test_states)

    assert corrected_one_step["n_eval_samples"] == baseline_one_step["n_eval_samples"]
    np.testing.assert_allclose(corrected_one_step["pred_spatial"], baseline_one_step["pred_spatial"])
    np.testing.assert_allclose(corrected_one_step["pred_latent"], baseline_one_step["pred_latent"])


def test_residual_corrected_forecaster_supports_mixed_one_step_loss_training():
    train_states = np.array(
        [
            [[1.0], [2.0], [4.0], [8.0], [16.0]],
            [[3.0], [6.0], [12.0], [24.0], [48.0]],
            [[5.0], [10.0], [20.0], [40.0], [80.0]],
        ],
        dtype=np.float32,
    ).reshape(3, 5, 1, 1)
    test_states = np.array(
        [
            [[7.0], [14.0], [28.0], [56.0], [112.0]],
        ],
        dtype=np.float32,
    ).reshape(1, 5, 1, 1)

    corrected = TrajectoryAwareResidualCorrectedForecaster(
        config=LatentModalForecasterConfig(
            ranks=[1, 1, 1],
            forecaster_type="linear",
            spatial_mean_centering=False,
            latent_normalization=False,
            delta_forecast=False,
            verbose=False,
        ),
        feature_mode="latent_plus_delta",
        correction_hidden_size=8,
        correction_num_layers=1,
        correction_num_epochs=8,
        correction_batch_size=4,
        correction_val_split=0.34,
        correction_spatial_loss_weight=0.25,
        correction_rel_frob_loss_weight=0.1,
    )
    corrected.fit(train_states)
    one_step = corrected.evaluate_one_step(test_states)

    assert np.isfinite(one_step["spatial_r2"])
    assert "train_spatial_loss" in corrected._correction_training_history
    assert len(corrected._correction_training_history["train_spatial_loss"]) >= 1


def test_model_registry_exposes_expected_benchmark_models():
    specs = get_navier_stokes_model_specs()

    assert [spec.name for spec in specs] == [
        "Linear Forecaster",
        "MLP Forecaster",
        "LSTM Forecaster",
        "LSTM + T+1 Residual Corrector",
        "Multi-Resolution Linear",
    ]
    assert [spec.slug for spec in specs] == [
        "linear_forecaster",
        "mlp_forecaster",
        "lstm_forecaster",
        "lstm_t_plus_1_residual_corrected",
        "multi_resolution_linear",
    ]


def test_model_registry_promotes_latent_plus_delta_lstm():
    specs = {spec.slug: spec for spec in get_navier_stokes_model_specs()}

    lstm_model = specs["lstm_forecaster"].factory()

    assert lstm_model._feature_mode == "latent_plus_delta"


def test_model_registry_exposes_residual_corrected_lstm_candidate():
    specs = {spec.slug: spec for spec in get_navier_stokes_model_specs()}

    corrected_model = specs["lstm_t_plus_1_residual_corrected"].factory()

    assert isinstance(corrected_model, TrajectoryAwareResidualCorrectedForecaster)
    assert corrected_model._feature_mode == "latent_plus_delta"
    assert corrected_model._correction_spatial_loss_weight == pytest.approx(0.0)
    assert corrected_model._correction_rel_frob_loss_weight == pytest.approx(0.0)
    assert specs["lstm_t_plus_1_residual_corrected"].notes["mixed_one_step_loss_available"] is True


def test_select_fixed_trajectory_indices_is_deterministic():
    first = select_fixed_trajectory_indices(10, count=3)
    second = select_fixed_trajectory_indices(10, count=3)

    assert first == second == [0, 4, 9]


def test_select_fixed_rollout_steps_is_deterministic():
    first = select_fixed_rollout_steps(11, count=4)
    second = select_fixed_rollout_steps(11, count=4)

    assert first == second == [0, 3, 6, 10]


def test_make_frame_filename_uses_zero_padded_step_index():
    assert make_frame_filename(0) == "frame_t00.png"
    assert make_frame_filename(7) == "frame_t07.png"
    assert make_frame_filename(18) == "frame_t18.png"


def test_build_examples_manifest_tracks_artifact_paths_and_settings():
    manifest = build_examples_manifest(
        output_root="plots/models_eval/examples",
        trajectory_indices=[0, 9],
        rollout_steps=[0, 3, 6, 10],
        image_settings={"dpi": 150, "fps": 2},
        per_model=[
            {
                "name": "LSTM Forecaster",
                "slug": "lstm_forecaster",
                "metrics": {"rollout_r2_common": 0.42},
                "artifacts": {
                    "contact_sheet": "examples/lstm_forecaster/contact_sheet.png",
                    "gif": "examples/lstm_forecaster/rollout.gif",
                    "frames": ["examples/lstm_forecaster/frames/frame_t00.png"],
                    "t_plus_one_diagnostics": "examples/lstm_forecaster/t_plus_one_diagnostics.png",
                },
            }
        ],
        comparison_artifacts=["examples/comparison/fixed_trajectory_0.png"],
    )

    assert manifest["output_root"] == "plots/models_eval/examples"
    assert manifest["trajectory_indices"] == [0, 9]
    assert manifest["rollout_steps"] == [0, 3, 6, 10]
    assert manifest["image_settings"] == {"dpi": 150, "fps": 2}
    assert manifest["models"][0]["slug"] == "lstm_forecaster"
    assert manifest["models"][0]["artifacts"]["t_plus_one_diagnostics"] == "examples/lstm_forecaster/t_plus_one_diagnostics.png"
    assert manifest["comparison_artifacts"] == ["examples/comparison/fixed_trajectory_0.png"]


def test_save_contact_sheet_writes_png_file(tmp_path):
    target = np.stack(
        [
            np.zeros((4, 4), dtype=np.float32),
            np.ones((4, 4), dtype=np.float32),
        ],
        axis=0,
    )
    pred = np.stack(
        [
            np.full((4, 4), 0.25, dtype=np.float32),
            np.full((4, 4), 0.75, dtype=np.float32),
        ],
        axis=0,
    )

    out_path = tmp_path / "contact_sheet.png"
    save_contact_sheet(
        target_frames=target,
        pred_frames=pred,
        step_indices=[0, 1],
        title="Synthetic Example",
        save_path=out_path,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_save_t_plus_one_diagnostics_sheet_writes_png_file(tmp_path):
    target = np.stack(
        [
            np.zeros((4, 4), dtype=np.float32),
            np.ones((4, 4), dtype=np.float32),
        ],
        axis=0,
    )
    baseline = np.stack(
        [
            np.full((4, 4), 0.5, dtype=np.float32),
            np.full((4, 4), 1.5, dtype=np.float32),
        ],
        axis=0,
    )
    corrected = np.stack(
        [
            np.full((4, 4), 0.25, dtype=np.float32),
            np.full((4, 4), 1.25, dtype=np.float32),
        ],
        axis=0,
    )

    out_path = tmp_path / "t_plus_one_diagnostics.png"
    save_t_plus_one_diagnostics_sheet(
        target_frames=target,
        baseline_frames=baseline,
        corrected_frames=corrected,
        step_indices=[0, 1],
        title="T+1 Diagnostics",
        save_path=out_path,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0
