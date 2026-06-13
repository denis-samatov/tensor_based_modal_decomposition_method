from pathlib import Path

import numpy as np
import pytest
import torch

from TBMD.config import LatentModalForecasterConfig
from TBMD.experiments.navier_stokes_examples import (
    build_examples_manifest,
    compute_common_horizon_diagnostics,
    make_frame_filename,
    save_t_plus_one_diagnostics_sheet,
    save_contact_sheet,
    select_fixed_rollout_steps,
    select_fixed_trajectory_indices,
)
from TBMD.experiments.navier_stokes_forecasting import (
    TrajectoryAwareCSForecaster,
    TrajectoryAwareDMDForecaster,
    TrajectoryAwareEigenvalueProjectedDMDForecaster,
    TrajectoryAwareLatentForecaster,
    TrajectoryAwareResidualCorrectedForecaster,
    TrajectoryAwarePersistenceForecaster,
    TrajectoryAwareMultiResolutionForecaster,
    TrajectoryAwareStableDMDForecaster,
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
from TBMD.experiments.navier_stokes_model_registry import (
    DEFAULT_NAVIER_STOKES_RANKS,
    get_navier_stokes_model_specs,
)


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


def test_tuning_split_uses_train_trajectory_holdout_not_official_test():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "tuning"
        / "tune_navier_stokes_models.py"
    )
    spec = importlib.util.spec_from_file_location("tune_navier_stokes_models", script_path)
    tune_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tune_module)

    states = np.arange(5 * 2, dtype=np.float32).reshape(5, 2, 1, 1)

    train_states, dev_states = tune_module.split_train_dev_trajectories(states, dev_split=0.4)

    assert train_states.shape == (3, 2, 1, 1)
    assert dev_states.shape == (2, 2, 1, 1)
    np.testing.assert_array_equal(train_states[:, :, 0, 0], [[0, 1], [2, 3], [4, 5]])
    np.testing.assert_array_equal(dev_states[:, :, 0, 0], [[6, 7], [8, 9]])


def test_tuning_candidates_support_feature_modes_and_residual_mixed_loss():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "tuning"
        / "tune_navier_stokes_models.py"
    )
    spec = importlib.util.spec_from_file_location("tune_navier_stokes_models", script_path)
    tune_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tune_module)

    candidates = {
        candidate.name: candidate
        for candidate in tune_module.build_candidates(groups=("lstm", "residual"))
    }

    lstm_candidate = candidates["lstm_h128_l2_s7_e150_plus_delta_features"]
    lstm_model = lstm_candidate.factory()
    assert lstm_model._feature_mode == "latent_plus_delta"
    assert lstm_candidate.metadata["feature_mode"] == "latent_plus_delta"

    mixed_candidate = candidates["lstm_residual_mixed_spatial_rel"]
    mixed_model = mixed_candidate.factory()
    assert isinstance(mixed_model, TrajectoryAwareResidualCorrectedForecaster)
    assert mixed_model._correction_spatial_loss_weight > 0.0
    assert mixed_model._correction_rel_frob_loss_weight > 0.0
    assert mixed_candidate.metadata["correction_loss"]["spatial"] > 0.0


def test_stage3_stable_dmd_candidate_grid_includes_unconstrained_and_damped_models():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "evaluation"
        / "evaluate_stage3_stable_dmd.py"
    )
    spec = importlib.util.spec_from_file_location("evaluate_stage3_stable_dmd", script_path)
    stage3_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage3_module)

    candidates = stage3_module.build_stable_dmd_candidates(rank=3, rhos=(1.0, 0.8, 0.6))

    assert [candidate["name"] for candidate in candidates] == [
        "dmd_unconstrained",
        "stable_dmd_rho_1_0",
        "projected_dmd_rho_1_0",
        "stable_dmd_rho_0_8",
        "projected_dmd_rho_0_8",
        "stable_dmd_rho_0_6",
        "projected_dmd_rho_0_6",
    ]
    assert candidates[0]["max_spectral_radius"] is None
    assert candidates[-1]["max_spectral_radius"] == pytest.approx(0.6)
    assert candidates[-1]["stabilization"] == "eigenvalue_projection"


def test_stage3_stable_dmd_selects_candidate_by_dev_rollout_metric():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "evaluation"
        / "evaluate_stage3_stable_dmd.py"
    )
    spec = importlib.util.spec_from_file_location("evaluate_stage3_stable_dmd", script_path)
    stage3_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage3_module)

    selected = stage3_module.select_best_result(
        [
            {"candidate": "stable_dmd_rho_0_9", "rollout_r2_common": -0.2},
            {"candidate": "stable_dmd_rho_0_6", "rollout_r2_common": 0.1},
            {"candidate": "stable_dmd_rho_0_4", "rollout_r2_common": -0.1},
        ]
    )

    assert selected["candidate"] == "stable_dmd_rho_0_6"


def test_stage4_candidate_grid_covers_rank_correction_lstm_and_optional_spatial_sweeps():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "tuning"
        / "tune_stage4_rank_sweep.py"
    )
    spec = importlib.util.spec_from_file_location("tune_stage4_rank_sweep", script_path)
    stage4_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage4_module)

    candidates = stage4_module.build_stage4_candidates(include_spatial=True)
    candidate_names = [candidate.name for candidate in candidates]

    assert len(candidate_names) == len(set(candidate_names))
    assert sorted(
        candidate.r3
        for candidate in candidates
        if "rank_sweep" in candidate.groups
    ) == [3, 5, 8, 10, 15]
    assert {
        candidate.correction_label
        for candidate in candidates
        if "correction_head_sweep" in candidate.groups
    } == {
        "corr_h64_l2_e120",
        "corr_h128_l2_e120",
        "corr_h64_l3_e150",
        "corr_h128_l2_e200",
    }
    assert {
        candidate.lstm_label
        for candidate in candidates
        if "lstm_backbone_sweep" in candidate.groups
    } == {
        "lstm_h128_l2",
        "lstm_h256_l2",
        "lstm_h128_l3",
    }
    assert [
        candidate.ranks
        for candidate in candidates
        if "spatial_rank_sweep" in candidate.groups
    ] == [[32, 32, 5], [48, 48, 5]]

    baseline = candidates[0]
    model = baseline.factory()
    assert isinstance(model, TrajectoryAwareResidualCorrectedForecaster)
    assert baseline.config().ranks == [64, 64, 5]
    assert model._feature_mode == "latent_plus_delta"


def test_stage4_selects_candidate_by_dev_rollout_metric():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "tuning"
        / "tune_stage4_rank_sweep.py"
    )
    spec = importlib.util.spec_from_file_location("tune_stage4_rank_sweep", script_path)
    stage4_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage4_module)

    selected = stage4_module.select_best_result(
        [
            {"candidate": "rank_r3_3", "rollout_r2_common": 0.2},
            {"candidate": "rank_r3_8", "rollout_r2_common": 0.45},
            {"candidate": "corr_h128_l2_e120", "rollout_r2_common": 0.3},
        ]
    )

    assert selected["candidate"] == "rank_r3_8"


def test_cs_forecasting_sweep_candidate_grid_and_selection():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "tuning"
        / "tune_cs_based_forecasting.py"
    )
    spec = importlib.util.spec_from_file_location("tune_cs_based_forecasting", script_path)
    cs_sweep_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs_sweep_module)

    candidates = cs_sweep_module.build_candidates(groups=("oracle", "lstsq", "cs"))
    candidate_names = [candidate.name for candidate in candidates]

    assert "oracle_full_projection_r30" in candidate_names
    assert "lstsq_r30_s25" in candidate_names
    assert "cs_r30_s15_i200" in candidate_names
    assert "cs_r30_s30_i100_eps001" in candidate_names
    assert "cs_r30_s30_i100_eps001_corr_h128_l2_e80" in candidate_names
    assert "cs_r45_s45_i100_eps001_corr_h128_l2_e80_spatial025_rel01" in candidate_names
    assert "cs_r45_s45_i100_eps001_corr_h256_l2_e120_spatial025_rel01" in candidate_names
    assert "cs_r45_s45_i100_eps001_lstm_h256_l2_corr_h512_l2_e120_spatial025_rel01" in candidate_names
    assert "cs_r45_s45_i100_eps001_lstm_h256_l2_corr_h512_l2_e120_latent" in candidate_names
    assert "cs_r45_s60_i100_eps001_lstm_h256_l2_corr_h512_l2_e120_latent" in candidate_names
    assert "cs_r45_s45_i100_eps001_lstm_h256_l2_corr_h512_l2_e120_latent_scale125" in candidate_names
    assert "cs_r45_s45_i100_eps001_corr_h256_l2_e120_window_spatial025_rel01" in candidate_names
    assert "cs_r30_s30_i100_eps001_lstsq_init" in candidate_names
    assert {
        candidate.metadata["group"]
        for candidate in candidates
    } == {"oracle", "lstsq", "cs"}
    promoted_candidate = candidates[
        candidate_names.index(
            "cs_r45_s45_i100_eps001_lstm_h256_l2_corr_h512_l2_e120_spatial025_rel01"
        )
    ]
    promoted_model = promoted_candidate.factory()
    assert promoted_model.lstm_hidden_size == 256
    assert promoted_model.correction_hidden_size == 512
    latent_candidate = candidates[
        candidate_names.index(
            "cs_r45_s45_i100_eps001_lstm_h256_l2_corr_h512_l2_e120_latent"
        )
    ]
    latent_model = latent_candidate.factory()
    assert latent_model.correction_spatial_loss_weight == pytest.approx(0.0)
    assert latent_model.correction_rel_frob_loss_weight == pytest.approx(0.0)
    scaled_candidate = candidates[
        candidate_names.index(
            "cs_r45_s45_i100_eps001_lstm_h256_l2_corr_h512_l2_e120_latent_scale125"
        )
    ]
    assert scaled_candidate.factory().correction_scale == pytest.approx(1.25)

    selected = cs_sweep_module.select_best_result(
        [
            {"candidate": "lstsq_r30_s15", "rollout_r2_common": 0.1},
            {"candidate": "cs_r30_s15_i200", "rollout_r2_common": 0.3},
            {"candidate": "oracle_full_projection_r30", "rollout_r2_common": 0.2},
        ]
    )
    assert selected["candidate"] == "cs_r30_s15_i200"


def test_cs_recovery_diagnostics_helpers_report_conditioning_and_sparsity():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "analysis"
        / "analyze_cs_recovery_diagnostics.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_cs_recovery_diagnostics", script_path)
    diagnostics_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostics_module)

    sensor_matrix = np.array([[1.0, 0.0], [0.5, 1.0], [0.0, 1.0]], dtype=np.float64)
    coeffs = np.array([[[1.0, 0.0], [0.2, 0.1]]], dtype=np.float64)

    matrix_metrics = diagnostics_module._sensor_matrix_diagnostics(sensor_matrix)
    sparsity_metrics = diagnostics_module._compute_sparsity_diagnostics(coeffs)

    assert matrix_metrics["shape"] == [3, 2]
    assert matrix_metrics["rank"] == 2
    assert matrix_metrics["condition_number"] >= 1.0
    assert 0.0 <= matrix_metrics["column_coherence"] <= 1.0
    assert sparsity_metrics["mean_l1_over_l2"] >= 1.0
    assert "top_3_energy_fraction_mean" not in sparsity_metrics


def test_temporal_regularized_recovery_smooths_noisy_sensor_coefficients():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "analysis"
        / "analyze_cs_recovery_diagnostics.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_cs_recovery_diagnostics", script_path)
    diagnostics_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostics_module)

    true_coeffs = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.5],
            [2.0, 1.0],
            [3.0, 1.5],
        ],
        dtype=np.float64,
    )
    sensor_dictionary = np.eye(2, dtype=np.float64)
    measurements = true_coeffs.copy()
    measurements[2] += np.array([3.0, -2.0])

    snapshot_lstsq = measurements @ np.linalg.pinv(sensor_dictionary).T
    temporal = diagnostics_module._solve_temporal_regularized_lstsq(
        measurements,
        sensor_dictionary,
        temporal_weight=5.0,
        ridge_weight=1e-8,
    )

    assert np.linalg.norm(temporal - true_coeffs) < np.linalg.norm(
        snapshot_lstsq - true_coeffs
    )


def test_windowed_tbmd_diagnostics_builds_causal_windows_and_dictionary():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "analysis"
        / "analyze_windowed_tbmd_qr_cs.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_windowed_tbmd_qr_cs", script_path)
    diagnostics_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostics_module)

    states = np.arange(1 * 5 * 2 * 2, dtype=np.float64).reshape(1, 5, 2, 2)
    windows = diagnostics_module._build_window_tensor(
        states,
        window_length=3,
        stride=1,
        max_windows=None,
    )

    assert windows.shape == (3, 2, 2, 3)
    np.testing.assert_array_equal(windows[:, :, :, 0], states[0, 0:3])
    np.testing.assert_array_equal(windows[:, :, :, 2], states[0, 2:5])

    core = np.zeros((2, 1, 1, 2), dtype=np.float64)
    core[0, 0, 0, 0] = 1.0
    core[1, 0, 0, 1] = 1.0
    factors = [
        np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float64),
        np.array([[2.0], [3.0]], dtype=np.float64),
        np.array([[5.0], [7.0]], dtype=np.float64),
        np.eye(2, dtype=np.float64),
    ]

    dictionary = diagnostics_module._compute_window_dictionary_from_tucker(core, factors)

    assert dictionary.shape == (3, 2, 2, 2)
    np.testing.assert_allclose(dictionary[0, :, :, 0], [[10.0, 14.0], [15.0, 21.0]])
    np.testing.assert_allclose(dictionary[1:, :, :, 0], 0.0)
    np.testing.assert_allclose(dictionary[1, :, :, 1], [[10.0, 14.0], [15.0, 21.0]])
    np.testing.assert_allclose(dictionary[[0, 2], :, :, 1], 0.0)


def test_windowed_tbmd_forecasting_predicts_next_from_history_coefficients():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "evaluation"
        / "evaluate_windowed_tbmd_qr_cs_forecasting.py"
    )
    spec = importlib.util.spec_from_file_location(
        "evaluate_windowed_tbmd_qr_cs_forecasting",
        script_path,
    )
    forecasting_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(forecasting_module)

    states = np.array(
        [[[[1.0]], [[2.0]], [[4.0]], [[8.0]]]],
        dtype=np.float64,
    )
    segments = forecasting_module._build_forecast_segment_tensor(
        states,
        history_length=2,
        stride=1,
        max_segments=None,
    )
    dictionary = np.array([[[[1.0]]], [[[2.0]]], [[[4.0]]]], dtype=np.float64)

    assert segments.shape == (3, 1, 1, 2)
    np.testing.assert_array_equal(segments[:, 0, 0, 0], [1.0, 2.0, 4.0])
    np.testing.assert_array_equal(segments[:, 0, 0, 1], [2.0, 4.0, 8.0])

    predictions, coefficients = forecasting_module._predict_next_full_history_lstsq(
        segments,
        dictionary,
        rcond=1e-10,
    )

    np.testing.assert_allclose(coefficients[:, 0], [1.0, 2.0])
    np.testing.assert_allclose(predictions[:, 0, 0], [4.0, 8.0], atol=1e-10)


def test_windowed_tbmd_forecasting_ridge_corrector_learns_target_residual():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "evaluation"
        / "evaluate_windowed_tbmd_qr_cs_forecasting.py"
    )
    spec = importlib.util.spec_from_file_location(
        "evaluate_windowed_tbmd_qr_cs_forecasting",
        script_path,
    )
    forecasting_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(forecasting_module)

    coefficients = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    base_predictions = coefficients.reshape(3, 1, 1) * 2.0
    target_frames = coefficients.reshape(3, 1, 1) * 3.0

    corrector = forecasting_module._fit_ridge_residual_corrector(
        target_frames,
        base_predictions,
        coefficients,
        alpha=1e-10,
    )
    corrected = forecasting_module._apply_ridge_residual_corrector(
        base_predictions,
        coefficients,
        corrector,
        scale=1.0,
    )
    half_corrected = forecasting_module._apply_ridge_residual_corrector(
        base_predictions,
        coefficients,
        corrector,
        scale=0.5,
    )

    np.testing.assert_allclose(corrected[:, 0, 0], [3.0, 6.0, 9.0], atol=1e-8)
    np.testing.assert_allclose(half_corrected[:, 0, 0], [2.5, 5.0, 7.5], atol=1e-8)


def test_windowed_tbmd_forecasting_rollout_recurses_on_predicted_history():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "evaluation"
        / "evaluate_windowed_tbmd_qr_cs_forecasting.py"
    )
    spec = importlib.util.spec_from_file_location(
        "evaluate_windowed_tbmd_qr_cs_forecasting",
        script_path,
    )
    forecasting_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(forecasting_module)

    trajectories = np.array(
        [[[[1.0]], [[2.0]], [[4.0]], [[8.0]], [[16.0]]]],
        dtype=np.float64,
    )
    dictionary = np.array([[[[1.0]]], [[[2.0]]], [[[4.0]]]], dtype=np.float64)
    spatial_sensor_indices = np.array([0])

    rollout = forecasting_module._evaluate_recursive_rollout(
        trajectories,
        dictionary,
        spatial_mask=np.array([[True]]),
        spatial_sensor_indices=spatial_sensor_indices,
        sensor_rcond=1e-10,
        cs_max_iter=10,
        cs_tol=1e-6,
        cs_epsilon_l1=1e-6,
        ridge_corrector=None,
        recovery_source="sensor_lstsq",
    )

    np.testing.assert_allclose(
        rollout["pred_spatial"][:, 0, 0],
        [4.0, 8.0, 16.0],
        atol=1e-10,
    )
    np.testing.assert_allclose(
        rollout["target_spatial"][:, 0, 0],
        [4.0, 8.0, 16.0],
        atol=1e-10,
    )
    assert rollout["n_rollout_steps"] == 3

    damped_rollout = forecasting_module._evaluate_recursive_rollout(
        trajectories,
        dictionary,
        spatial_mask=np.array([[True]]),
        spatial_sensor_indices=spatial_sensor_indices,
        sensor_rcond=1e-10,
        cs_max_iter=10,
        cs_tol=1e-6,
        cs_epsilon_l1=1e-6,
        ridge_corrector=None,
        recovery_source="sensor_lstsq",
        rollout_update_blend=0.0,
    )

    np.testing.assert_allclose(
        damped_rollout["pred_spatial"][:, 0, 0],
        [2.0, 2.0, 2.0],
        atol=1e-10,
    )


def test_windowed_tbmd_closed_loop_training_pairs_use_rollout_history():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "evaluation"
        / "evaluate_windowed_tbmd_qr_cs_forecasting.py"
    )
    spec = importlib.util.spec_from_file_location(
        "evaluate_windowed_tbmd_qr_cs_forecasting",
        script_path,
    )
    forecasting_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(forecasting_module)

    trajectories = np.array([[[[1.0]], [[2.0]], [[4.0]]]], dtype=np.float64)
    dictionary = np.array([[[[1.0]]], [[[2.0]]], [[[3.0]]]], dtype=np.float64)

    targets, base_predictions, coefficients = (
        forecasting_module._collect_closed_loop_residual_pairs(
            trajectories,
            dictionary,
            spatial_mask=np.array([[True]]),
            spatial_sensor_indices=np.array([0]),
            sensor_rcond=1e-10,
            cs_max_iter=10,
            cs_tol=1e-6,
            cs_epsilon_l1=1e-6,
            history_corrector=None,
            history_correction_scale=1.0,
            recovery_source="sensor_lstsq",
        )
    )

    np.testing.assert_allclose(coefficients[:, 0], [1.0], atol=1e-10)
    np.testing.assert_allclose(base_predictions[:, 0, 0], [3.0], atol=1e-10)
    np.testing.assert_allclose(targets[:, 0, 0], [4.0], atol=1e-10)


def test_windowed_tbmd_sensor_budget_sweep_selects_by_dev_corrected_r2():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "tuning"
        / "tune_windowed_tbmd_qr_cs_sensor_budget.py"
    )
    spec = importlib.util.spec_from_file_location(
        "tune_windowed_tbmd_qr_cs_sensor_budget",
        script_path,
    )
    sweep_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep_module)

    results = [
        {
            "n_spatial_sensors": 300,
            "selected_ridge": {
                "dev_fixed_sensor_cs_r2": 0.80,
                "test_fixed_sensor_cs_r2": 0.82,
            },
        },
        {
            "n_spatial_sensors": 500,
            "selected_ridge": {
                "dev_fixed_sensor_cs_r2": 0.81,
                "test_fixed_sensor_cs_r2": 0.83,
            },
        },
        {
            "n_spatial_sensors": 600,
            "selected_ridge": {
                "dev_fixed_sensor_cs_r2": 0.805,
                "test_fixed_sensor_cs_r2": 0.84,
            },
        },
    ]

    best = sweep_module._select_best_budget_result(results)

    assert best["n_spatial_sensors"] == 500
    assert best["selected_ridge"]["test_fixed_sensor_cs_r2"] == pytest.approx(0.83)

    practical = sweep_module._select_practical_budget_result(results, tolerance=0.01)

    assert practical["n_spatial_sensors"] == 300


def test_hybrid_tbmd_qr_cs_blends_backbone_and_sensor_forecasts():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "evaluation"
        / "evaluate_hybrid_tbmd_qr_cs_forecasting.py"
    )
    spec = importlib.util.spec_from_file_location(
        "evaluate_hybrid_tbmd_qr_cs_forecasting",
        script_path,
    )
    hybrid_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hybrid_module)

    backbone = np.array([[[1.0]], [[5.0]]], dtype=np.float64)
    sensor = np.array([[[3.0]], [[1.0]]], dtype=np.float64)

    blended = hybrid_module._blend_predictions(backbone, sensor, beta=0.25)

    np.testing.assert_allclose(blended[:, 0, 0], [1.5, 4.0])


def test_hybrid_tbmd_qr_cs_selects_weight_by_dev_rollout():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "evaluation"
        / "evaluate_hybrid_tbmd_qr_cs_forecasting.py"
    )
    spec = importlib.util.spec_from_file_location(
        "evaluate_hybrid_tbmd_qr_cs_forecasting",
        script_path,
    )
    hybrid_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hybrid_module)

    candidates = [
        {"beta": 0.0, "dev": {"spatial_r2": 0.74}, "test": {"spatial_r2": 0.77}},
        {"beta": 0.25, "dev": {"spatial_r2": 0.78}, "test": {"spatial_r2": 0.70}},
        {"beta": 0.5, "dev": {"spatial_r2": 0.76}, "test": {"spatial_r2": 0.81}},
    ]

    selected = hybrid_module._select_beta_by_dev_rollout(candidates)

    assert selected["beta"] == pytest.approx(0.25)
    assert selected["test"]["spatial_r2"] == pytest.approx(0.70)


def test_fast_tplus1_extracts_fixed_spatial_history_measurements():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "evaluation"
        / "evaluate_windowed_tbmd_qr_cs_fast_tplus1.py"
    )
    spec = importlib.util.spec_from_file_location(
        "evaluate_windowed_tbmd_qr_cs_fast_tplus1",
        script_path,
    )
    fast_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fast_module)

    segments = np.array(
        [
            [[[1.0, 10.0], [2.0, 20.0]], [[3.0, 30.0], [4.0, 40.0]]],
            [[[5.0, 50.0], [6.0, 60.0]], [[7.0, 70.0], [8.0, 80.0]]],
            [[[9.0, 90.0], [10.0, 100.0]], [[11.0, 110.0], [12.0, 120.0]]],
        ],
        dtype=np.float64,
    )
    dictionary = np.zeros((3, 2, 2, 1), dtype=np.float64)

    measurements = fast_module._history_sensor_measurements_from_segments(
        segments,
        dictionary,
        spatial_sensor_indices=np.array([0, 3]),
    )

    assert measurements.shape == (2, 4)
    np.testing.assert_allclose(measurements[0], [1.0, 4.0, 5.0, 8.0])
    np.testing.assert_allclose(measurements[1], [10.0, 40.0, 50.0, 80.0])


def test_fast_tplus1_standardized_ridge_corrector_learns_sensor_residual():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "evaluation"
        / "evaluate_windowed_tbmd_qr_cs_fast_tplus1.py"
    )
    spec = importlib.util.spec_from_file_location(
        "evaluate_windowed_tbmd_qr_cs_fast_tplus1",
        script_path,
    )
    fast_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fast_module)

    features = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float64)
    base_predictions = (2.0 * features).reshape(4, 1, 1)
    target_frames = (3.0 * features).reshape(4, 1, 1)

    corrector = fast_module._fit_standardized_ridge_residual_corrector(
        target_frames,
        base_predictions,
        features,
        alpha=1e-10,
    )
    corrected = fast_module._apply_standardized_ridge_residual_corrector(
        base_predictions,
        features,
        corrector,
    )

    np.testing.assert_allclose(corrected[:, 0, 0], [3.0, 6.0, 9.0, 12.0], atol=1e-8)


def test_fast_tplus1_low_rank_residual_corrector_learns_compressed_residual():
    from TBMD.experiments.navier_stokes_fast_tplus1 import (
        apply_ridge_residual_corrector,
        fit_ridge_residual_corrector,
    )

    coeffs = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float64)
    base_predictions = np.zeros((4, 2, 2), dtype=np.float64)
    residual_pattern = np.array([[1.0, -1.0], [0.5, -0.5]], dtype=np.float64)
    target_frames = coeffs.reshape(4, 1, 1) * residual_pattern

    corrector = fit_ridge_residual_corrector(
        target_frames,
        base_predictions,
        coeffs,
        alpha=1e-10,
        residual_rank=1,
    )
    corrected = apply_ridge_residual_corrector(
        base_predictions,
        coeffs,
        corrector,
        scale=1.0,
    )

    assert corrector["residual_rank"] == 1
    assert corrector["residual_basis"].shape == (1, 4)
    np.testing.assert_allclose(corrected, target_frames, atol=1e-8)


def test_fast_tplus1_weighted_residual_basis_stores_error_sensitive_weights():
    from TBMD.experiments.navier_stokes_fast_tplus1 import (
        apply_ridge_residual_corrector,
        fit_ridge_residual_corrector,
    )

    coeffs = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float64)
    base_predictions = np.zeros((4, 1, 3), dtype=np.float64)
    target_frames = np.array(
        [
            [[1.0, 0.1, 0.2]],
            [[2.0, 0.0, 0.1]],
            [[3.0, -0.1, 0.0]],
            [[4.0, 0.1, -0.1]],
        ],
        dtype=np.float64,
    )

    corrector = fit_ridge_residual_corrector(
        target_frames,
        base_predictions,
        coeffs,
        alpha=1e-10,
        residual_rank=1,
        residual_weighting="residual_energy",
        residual_weight_floor=0.1,
    )
    corrected = apply_ridge_residual_corrector(
        base_predictions,
        coeffs,
        corrector,
        scale=1.0,
    )

    weights = corrector["residual_weights"]
    assert corrector["residual_weighting"] == "residual_energy"
    assert weights.shape == (3,)
    assert weights[0] > weights[1]
    assert corrected.shape == target_frames.shape


def test_fast_tplus1_highpass_residual_target_ignores_constant_bias():
    from TBMD.experiments.navier_stokes_fast_tplus1 import (
        apply_ridge_residual_corrector,
        fft_highpass_frames,
        fit_ridge_residual_corrector,
    )

    coeffs = np.linspace(0.0, 1.0, 4, dtype=np.float64).reshape(-1, 1)
    base_predictions = np.zeros((4, 8, 8), dtype=np.float64)
    constant_targets = np.ones_like(base_predictions) * 2.0
    checker = (np.indices((8, 8)).sum(axis=0) % 2).astype(np.float64)
    checker = 2.0 * checker - 1.0

    constant_highpass = fft_highpass_frames(constant_targets, cutoff_fraction=0.4)
    checker_highpass = fft_highpass_frames(checker[None], cutoff_fraction=0.4)
    highpass_corrector = fit_ridge_residual_corrector(
        constant_targets,
        base_predictions,
        coeffs,
        alpha=1e-8,
        residual_rank=None,
        residual_target="highpass",
        highpass_cutoff_fraction=0.4,
    )
    full_corrector = fit_ridge_residual_corrector(
        constant_targets,
        base_predictions,
        coeffs,
        alpha=1e-8,
        residual_rank=None,
    )
    highpass_pred = apply_ridge_residual_corrector(
        base_predictions,
        coeffs,
        highpass_corrector,
        scale=1.0,
    )
    full_pred = apply_ridge_residual_corrector(
        base_predictions,
        coeffs,
        full_corrector,
        scale=1.0,
    )

    assert np.linalg.norm(constant_highpass) < 1e-8
    assert np.linalg.norm(checker_highpass) > 0.9 * np.linalg.norm(checker)
    assert np.max(np.abs(highpass_pred)) < 1e-6
    assert np.mean(full_pred) == pytest.approx(2.0, abs=1e-4)
    assert highpass_corrector["residual_target"] == "highpass"


def test_fast_tplus1_highpass_energy_weighting_emphasizes_textured_region():
    from TBMD.experiments.navier_stokes_fast_tplus1 import fit_ridge_residual_corrector

    checker = (np.indices((8, 8)).sum(axis=0) % 2).astype(np.float64)
    checker = 2.0 * checker - 1.0
    textured_half = np.zeros((8, 8), dtype=np.float64)
    textured_half[:4] = checker[:4]
    scales = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float64)
    target_frames = scales[:, None, None] * textured_half[None]
    base_predictions = np.zeros_like(target_frames)
    coeffs = scales.reshape(-1, 1)

    corrector = fit_ridge_residual_corrector(
        target_frames,
        base_predictions,
        coeffs,
        alpha=1e-8,
        residual_rank=1,
        residual_weighting="highpass_energy",
        highpass_cutoff_fraction=0.35,
    )
    weights = corrector["residual_weights"].reshape(8, 8)

    assert corrector["residual_weighting"] == "highpass_energy"
    assert np.mean(weights[:4]) > np.mean(weights[4:])


def test_fast_tplus1_composite_residual_corrector_combines_patch_and_hf_components():
    from TBMD.experiments.navier_stokes_fast_tplus1 import (
        apply_ridge_residual_corrector,
        fit_composite_patch_hf_residual_corrector,
    )

    yy, xx = np.indices((8, 8))
    smooth = np.sin(2 * np.pi * xx / 8.0)
    checker = 2.0 * ((yy + xx) % 2).astype(np.float64) - 1.0
    pattern = smooth + 0.25 * checker
    coeffs = np.linspace(0.5, 2.0, 5, dtype=np.float64).reshape(-1, 1)
    base_predictions = np.zeros((5, 8, 8), dtype=np.float64)
    target_frames = coeffs[:, None] * pattern.reshape(1, -1)
    target_frames = target_frames.reshape(5, 8, 8)

    corrector = fit_composite_patch_hf_residual_corrector(
        target_frames,
        base_predictions,
        coeffs,
        alpha=1e-8,
        patch_size=4,
        patch_residual_rank=1,
        hf_residual_rank=1,
        patch_scale=1.0,
        hf_scale=0.25,
        highpass_cutoff_fraction=0.35,
    )
    pred = apply_ridge_residual_corrector(
        base_predictions,
        coeffs,
        corrector,
        scale=1.0,
    )

    assert corrector["mode"] == "composite_patch_hf_svd"
    assert len(corrector["components"]) == 2
    assert corrector["components"][0]["name"] == "patch"
    assert corrector["components"][1]["name"] == "hfweighted"
    assert pred.shape == target_frames.shape
    assert np.mean((target_frames - pred) ** 2) < np.mean(target_frames**2)


def test_fast_tplus1_hard_frame_sample_weighting_prioritizes_large_errors():
    from TBMD.experiments.navier_stokes_fast_tplus1 import (
        compute_residual_sample_weights,
        fit_ridge_residual_corrector,
    )

    base_predictions = np.zeros((4, 1, 1), dtype=np.float64)
    target_frames = np.array([[[0.1]], [[0.2]], [[0.3]], [[4.0]]], dtype=np.float64)
    coeffs = np.arange(4, dtype=np.float64).reshape(-1, 1)

    sample_weights = compute_residual_sample_weights(
        target_frames,
        base_predictions,
        weighting="hard_frame_rmse",
        power=1.0,
        floor=0.25,
        clip=4.0,
    )
    corrector = fit_ridge_residual_corrector(
        target_frames,
        base_predictions,
        coeffs,
        alpha=1e-8,
        residual_rank=1,
        sample_weighting="hard_frame_rmse",
        sample_weight_power=1.0,
        sample_weight_floor=0.25,
        sample_weight_clip=4.0,
    )

    assert sample_weights.shape == (4,)
    assert np.mean(sample_weights) == pytest.approx(1.0)
    assert sample_weights[-1] > sample_weights[0]
    assert corrector["sample_weighting"] == "hard_frame_rmse"
    assert corrector["sample_weight_max"] > corrector["sample_weight_min"]


def test_fast_tplus1_sensor_innovation_gate_shrinks_inconsistent_correction():
    from TBMD.experiments.navier_stokes_fast_tplus1 import (
        apply_ridge_residual_corrector,
        attach_coefficient_gate,
    )

    train_coeffs = np.array([[1.0], [1.0], [1.0]], dtype=np.float64)
    train_measurements = np.array([[1.0], [1.2], [1.4]], dtype=np.float64)
    decoder_payload = {"sensing_matrix": np.array([[1.0]], dtype=np.float64)}
    corrector = {
        "alpha": 0.0,
        "weights": np.array([[1.0], [0.0]], dtype=np.float64),
        "feature_dim": 1,
        "coefficient_dim": 1,
        "output_dim": 1,
        "mode": "full",
        "residual_rank": None,
        "residual_weighting": "uniform",
    }
    gated = attach_coefficient_gate(
        corrector,
        train_coeffs,
        gate_type="sensor_innovation_rms",
        threshold=0.0,
        strength=1.0,
        gate_min=0.25,
        measurements=train_measurements,
        decoder_payload=decoder_payload,
    )

    base = np.zeros((2, 1, 1), dtype=np.float64)
    coeffs = np.array([[1.0], [1.0]], dtype=np.float64)
    measurements = np.array([[1.0], [4.0]], dtype=np.float64)
    pred = apply_ridge_residual_corrector(
        base,
        coeffs,
        gated,
        scale=1.0,
        measurements=measurements,
        decoder_payload=decoder_payload,
    )

    assert gated["gate_type"] == "sensor_innovation_rms"
    assert pred[0, 0, 0] > pred[1, 0, 0]
    assert pred[1, 0, 0] >= 0.25


def test_fast_tplus1_mlp_residual_corrector_learns_nonlinear_compressed_residual():
    from TBMD.experiments.navier_stokes_fast_tplus1 import (
        apply_ridge_residual_corrector,
        fit_mlp_residual_corrector,
    )

    coeff_values = np.linspace(-1.0, 1.0, 32, dtype=np.float64)
    coeffs = coeff_values.reshape(-1, 1)
    base_predictions = np.zeros((coeffs.shape[0], 1, 1), dtype=np.float64)
    target_frames = (coeff_values**2).reshape(-1, 1, 1)

    corrector = fit_mlp_residual_corrector(
        target_frames,
        base_predictions,
        coeffs,
        residual_rank=1,
        hidden_size=8,
        num_epochs=300,
        batch_size=32,
        learning_rate=0.03,
        weight_decay=0.0,
        random_state=0,
    )
    corrected = apply_ridge_residual_corrector(
        base_predictions,
        coeffs,
        corrector,
        scale=1.0,
    )

    assert corrector["mode"] == "residual_svd_mlp"
    assert corrected.shape == target_frames.shape
    assert np.sqrt(np.mean((corrected - target_frames) ** 2)) < 0.08


def test_fast_tplus1_predicts_next_from_sparse_history_measurements():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "evaluation"
        / "evaluate_windowed_tbmd_qr_cs_fast_tplus1.py"
    )
    spec = importlib.util.spec_from_file_location(
        "evaluate_windowed_tbmd_qr_cs_fast_tplus1",
        script_path,
    )
    fast_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fast_module)

    model = {
        "dictionary": np.array([[[[1.0]]], [[[2.0]]], [[[4.0]]]], dtype=np.float64),
        "spatial_mean": np.zeros((1, 1), dtype=np.float64),
        "spatial_sensor_indices": np.array([0]),
        "sensor_rcond": 1e-10,
        "coefficient_corrector": None,
        "correction_scale": 1.0,
    }
    history = np.array([[[[1.0]], [[2.0]]]], dtype=np.float64)

    pred, coeffs = fast_module._predict_fast_next_from_history(history, model)

    np.testing.assert_allclose(coeffs[:, 0], [1.0], atol=1e-10)
    np.testing.assert_allclose(pred[:, 0, 0], [4.0], atol=1e-10)


def test_fast_tplus1_sensor_decoders_predict_from_precomputed_maps():
    from TBMD.experiments.navier_stokes_fast_tplus1 import (
        fit_sensor_coefficient_decoder,
        predict_from_history_sensor_decoder,
    )

    dictionary = np.zeros((2, 1, 2, 2), dtype=np.float64)
    dictionary[0, 0, 0] = [1.0, 0.0]
    dictionary[0, 0, 1] = [0.0, 1.0]
    dictionary[1, 0, 0] = [2.0, 0.0]
    dictionary[1, 0, 1] = [0.0, 3.0]
    history = np.array([[[[3.0, 4.0]]]], dtype=np.float64)
    sensor_indices = np.array([0, 1])

    ridge_decoder = fit_sensor_coefficient_decoder(
        dictionary,
        sensor_indices,
        decoder="ridge",
        ridge_lambda=1e-10,
    )
    ridge_pred, ridge_coeffs = predict_from_history_sensor_decoder(
        history,
        dictionary,
        sensor_indices,
        ridge_decoder,
    )
    np.testing.assert_allclose(ridge_coeffs, [[3.0, 4.0]], atol=1e-8)
    np.testing.assert_allclose(ridge_pred[0, 0], [6.0, 12.0], atol=1e-8)

    fista_decoder = fit_sensor_coefficient_decoder(
        dictionary,
        sensor_indices,
        decoder="fista",
        l1_lambda=1e-8,
        max_iter=40,
        tol=0.0,
    )
    fista_pred, fista_coeffs = predict_from_history_sensor_decoder(
        history,
        dictionary,
        sensor_indices,
        fista_decoder,
    )
    assert fista_coeffs.shape == (1, 2)
    np.testing.assert_allclose(fista_pred[0, 0], [6.0, 12.0], atol=1e-5)


def test_fast_windowed_tbmd_qr_cs_forecaster_save_load_roundtrip(tmp_path):
    from TBMD.experiments.navier_stokes_fast_tplus1 import (
        FastWindowedTBMDQRCSConfig,
        FastWindowedTBMDQRCSForecaster,
    )

    pattern = np.array([[1.0, -0.5], [0.25, 0.75]], dtype=np.float64)
    scales = np.array(
        [
            [1.0, 1.2, 1.44, 1.728],
            [0.8, 0.96, 1.152, 1.3824],
            [1.4, 1.68, 2.016, 2.4192],
        ],
        dtype=np.float64,
    )
    states = np.asarray([[scale * pattern for scale in traj] for traj in scales])

    model = FastWindowedTBMDQRCSForecaster(
        FastWindowedTBMDQRCSConfig(
            history_length=2,
            ranks=[3, 2, 2, 1],
            n_spatial_sensors=2,
            max_train_segments=None,
            coefficient_calibration_type="ridge",
            coefficient_calibration_alpha=1e-8,
            coefficient_calibration_blend=0.5,
            correction_alpha=1e-8,
            correction_residual_rank=1,
            correction_gate_type="coefficient_rms",
            correction_gate_threshold=1.0,
            correction_gate_strength=0.5,
            correction_gate_min=0.75,
            correction_innovation_rank=1,
            correction_innovation_include_norms=True,
            sensor_rcond=1e-10,
            random_state=0,
        )
    )
    model.fit(states)
    history = states[:1, :2]
    pred_before = model.predict_next(history)

    path = tmp_path / "fast_predictor.npz"
    model.save(path)
    loaded = FastWindowedTBMDQRCSForecaster.load(path)
    pred_after = loaded.predict_next(history)

    np.testing.assert_allclose(pred_before, pred_after, atol=1e-6)
    assert loaded.get_config()["n_spatial_sensors"] == 2
    assert loaded.get_config()["correction_residual_rank"] == 1
    assert loaded.get_config()["coefficient_calibration_type"] == "ridge"
    assert loaded.get_config()["coefficient_calibration_blend"] == pytest.approx(0.5)
    assert loaded.get_config()["correction_gate_type"] == "coefficient_rms"
    assert loaded.get_config()["correction_gate_min"] == pytest.approx(0.75)
    assert loaded.get_config()["correction_innovation_rank"] == 1
    assert loaded.get_config()["correction_innovation_include_norms"]
    assert loaded.get_metrics()["fit"]["n_train_segments"] == 6


def test_fast_tplus1_delta_coefficient_calibrator_learns_conservative_latent_update():
    from TBMD.experiments.navier_stokes_fast_tplus1 import (
        apply_coefficient_calibrator,
        fit_coefficient_calibrator,
    )

    dictionary = np.zeros((2, 1, 2, 2), dtype=np.float64)
    dictionary[-1, 0, 0, 0] = 1.0
    dictionary[-1, 0, 1, 1] = 1.0
    recovered = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]], dtype=np.float64)
    target_coeffs = recovered + np.array([1.0, -0.5], dtype=np.float64)
    target_frames = target_coeffs.reshape(target_coeffs.shape[0], 1, 2)

    calibrator = fit_coefficient_calibrator(
        recovered,
        target_frames,
        dictionary,
        calibration_type="delta_ridge",
        target="target",
        alpha=1e-10,
        blend=0.5,
        rcond=1e-10,
    )
    corrected = apply_coefficient_calibrator(recovered, calibrator)

    assert calibrator["type"] == "delta_ridge"
    assert calibrator["feature_dim"] == recovered.shape[1]
    np.testing.assert_allclose(corrected, recovered + 0.5 * (target_coeffs - recovered), atol=1e-6)


def test_fast_windowed_tbmd_qr_cs_delta_coefficient_calibrator_save_load_roundtrip(tmp_path):
    from TBMD.experiments.navier_stokes_fast_tplus1 import (
        FastWindowedTBMDQRCSConfig,
        FastWindowedTBMDQRCSForecaster,
    )

    pattern = np.array([[1.0, -0.5], [0.25, 0.75]], dtype=np.float64)
    scales = np.array(
        [
            [1.0, 1.2, 1.44, 1.728],
            [0.8, 0.96, 1.152, 1.3824],
            [1.4, 1.68, 2.016, 2.4192],
        ],
        dtype=np.float64,
    )
    states = np.asarray([[scale * pattern for scale in traj] for traj in scales])

    model = FastWindowedTBMDQRCSForecaster(
        FastWindowedTBMDQRCSConfig(
            history_length=2,
            ranks=[3, 2, 2, 1],
            n_spatial_sensors=2,
            max_train_segments=None,
            sensor_decoder="ridge",
            decoder_ridge_lambda=1e-8,
            coefficient_calibration_type="delta_ridge",
            coefficient_calibration_alpha=1e-8,
            coefficient_calibration_blend=0.25,
            coefficient_calibration_innovation_rank=1,
            coefficient_calibration_include_norms=True,
            coefficient_temporal_smoothing_alpha=0.2,
            correction_head_type="none",
            sensor_rcond=1e-10,
            random_state=0,
        )
    )
    model.fit(states)
    assert model.get_metrics()["fit"]["coefficient_temporal_smoothing_alpha"] == pytest.approx(0.2)
    history = states[:1, :2]
    pred_before = model.predict_next(history)

    path = tmp_path / "fast_delta_coeff_predictor.npz"
    model.save(path)
    loaded = FastWindowedTBMDQRCSForecaster.load(path)
    pred_after = loaded.predict_next(history)

    np.testing.assert_allclose(pred_before, pred_after, atol=1e-6)
    assert loaded.get_config()["coefficient_calibration_type"] == "delta_ridge"
    assert loaded.get_config()["coefficient_calibration_blend"] == pytest.approx(0.25)
    assert loaded.get_config()["coefficient_calibration_innovation_rank"] == 1
    assert loaded.get_config()["coefficient_calibration_include_norms"]
    assert loaded.get_config()["coefficient_temporal_smoothing_alpha"] == pytest.approx(0.2)


def test_fast_windowed_tbmd_qr_cs_mlp_corrector_save_load_roundtrip(tmp_path):
    from TBMD.experiments.navier_stokes_fast_tplus1 import (
        FastWindowedTBMDQRCSConfig,
        FastWindowedTBMDQRCSForecaster,
    )

    pattern = np.array([[1.0, -0.5], [0.25, 0.75]], dtype=np.float64)
    scales = np.array(
        [
            [1.0, 1.2, 1.44, 1.728],
            [0.8, 0.96, 1.152, 1.3824],
            [1.4, 1.68, 2.016, 2.4192],
        ],
        dtype=np.float64,
    )
    states = np.asarray([[scale * pattern for scale in traj] for traj in scales])
    model = FastWindowedTBMDQRCSForecaster(
        FastWindowedTBMDQRCSConfig(
            history_length=2,
            ranks=[3, 2, 2, 1],
            n_spatial_sensors=2,
            max_train_segments=None,
            correction_head_type="mlp_residual_svd",
            correction_residual_rank=1,
            correction_hidden_size=4,
            correction_num_epochs=3,
            correction_batch_size=3,
            correction_learning_rate=0.01,
            sensor_rcond=1e-10,
            random_state=0,
        )
    )
    model.fit(states)
    history = states[:1, :2]
    pred_before = model.predict_next(history)

    path = tmp_path / "fast_mlp_predictor.npz"
    model.save(path)
    loaded = FastWindowedTBMDQRCSForecaster.load(path)
    pred_after = loaded.predict_next(history)

    np.testing.assert_allclose(pred_before, pred_after, atol=1e-6)
    assert loaded.get_config()["correction_head_type"] == "mlp_residual_svd"


def test_fast_windowed_tbmd_qr_cs_patch_residual_save_load_roundtrip(tmp_path):
    from TBMD.experiments.navier_stokes_fast_tplus1 import (
        FastWindowedTBMDQRCSConfig,
        FastWindowedTBMDQRCSForecaster,
    )

    x = np.linspace(-1.0, 1.0, 4)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    pattern = xx + 0.5 * yy
    scales = np.array(
        [
            [1.0, 1.1, 1.21, 1.331],
            [0.8, 0.88, 0.968, 1.0648],
            [1.3, 1.43, 1.573, 1.7303],
            [0.6, 0.66, 0.726, 0.7986],
        ],
        dtype=np.float64,
    )
    states = np.asarray([[scale * pattern for scale in traj] for traj in scales])

    model = FastWindowedTBMDQRCSForecaster(
        FastWindowedTBMDQRCSConfig(
            history_length=2,
            ranks=[3, 2, 2, 2],
            n_spatial_sensors=4,
            max_train_segments=None,
            correction_head_type="patch_residual_svd",
            correction_patch_size=2,
            correction_patch_residual_rank=1,
            correction_alpha=1e-8,
            sensor_rcond=1e-10,
            random_state=0,
        )
    )
    model.fit(states)
    history = states[:1, :2]
    pred_before = model.predict_next(history)

    path = tmp_path / "fast_patch_predictor.npz"
    model.save(path)
    loaded = FastWindowedTBMDQRCSForecaster.load(path)
    pred_after = loaded.predict_next(history)

    np.testing.assert_allclose(pred_before, pred_after, atol=1e-6)
    assert loaded.get_config()["correction_head_type"] == "patch_residual_svd"
    assert loaded.get_config()["correction_patch_size"] == 2
    assert loaded.get_config()["correction_patch_residual_rank"] == 1


def test_fast_windowed_tbmd_qr_cs_composite_residual_save_load_roundtrip(tmp_path):
    from TBMD.experiments.navier_stokes_fast_tplus1 import (
        FastWindowedTBMDQRCSConfig,
        FastWindowedTBMDQRCSForecaster,
    )

    x = np.linspace(-1.0, 1.0, 4)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    smooth = xx + 0.5 * yy
    texture = 0.1 * ((np.indices((4, 4)).sum(axis=0) % 2) * 2.0 - 1.0)
    pattern = smooth + texture
    scales = np.array(
        [
            [1.0, 1.1, 1.21, 1.331],
            [0.8, 0.88, 0.968, 1.0648],
            [1.3, 1.43, 1.573, 1.7303],
            [0.6, 0.66, 0.726, 0.7986],
        ],
        dtype=np.float64,
    )
    states = np.asarray([[scale * pattern for scale in traj] for traj in scales])

    model = FastWindowedTBMDQRCSForecaster(
        FastWindowedTBMDQRCSConfig(
            history_length=2,
            ranks=[3, 2, 2, 2],
            n_spatial_sensors=4,
            max_train_segments=None,
            sensor_decoder="ridge",
            decoder_ridge_lambda=1e-8,
            correction_head_type="composite_patch_hf_svd",
            correction_patch_size=2,
            correction_patch_residual_rank=1,
            correction_residual_rank=1,
            correction_scale=1.3,
            correction_hf_scale=0.4,
            correction_highpass_cutoff_fraction=0.45,
            correction_alpha=1e-8,
            sensor_rcond=1e-10,
            random_state=0,
        )
    )
    model.fit(states)
    history = states[:1, :2]
    pred_before = model.predict_next(history)

    path = tmp_path / "fast_composite_predictor.npz"
    model.save(path)
    loaded = FastWindowedTBMDQRCSForecaster.load(path)
    pred_after = loaded.predict_next(history)

    np.testing.assert_allclose(pred_before, pred_after, atol=1e-6)
    assert loaded.get_config()["correction_head_type"] == "composite_patch_hf_svd"
    assert loaded.get_config()["correction_patch_residual_rank"] == 1
    assert loaded.get_config()["correction_residual_rank"] == 1
    assert loaded.get_config()["correction_hf_scale"] == pytest.approx(0.4)
    assert loaded._coefficient_corrector["mode"] == "composite_patch_hf_svd"
    assert [component["name"] for component in loaded._coefficient_corrector["components"]] == [
        "patch",
        "hfweighted",
    ]


def test_fast_windowed_tbmd_qr_cs_none_corrector_uses_base_prediction(tmp_path):
    from TBMD.experiments.navier_stokes_fast_tplus1 import (
        FastWindowedTBMDQRCSConfig,
        FastWindowedTBMDQRCSForecaster,
    )

    pattern = np.array([[1.0, -0.5], [0.25, 0.75]], dtype=np.float64)
    scales = np.array(
        [
            [1.0, 1.2, 1.44, 1.728],
            [0.8, 0.96, 1.152, 1.3824],
            [1.4, 1.68, 2.016, 2.4192],
        ],
        dtype=np.float64,
    )
    states = np.asarray([[scale * pattern for scale in traj] for traj in scales])

    model = FastWindowedTBMDQRCSForecaster(
        FastWindowedTBMDQRCSConfig(
            history_length=2,
            ranks=[3, 2, 2, 1],
            n_spatial_sensors=2,
            max_train_segments=None,
            coefficient_calibration_type="ridge",
            coefficient_calibration_blend=1.0,
            correction_head_type="none",
            sensor_rcond=1e-10,
            random_state=0,
        )
    )
    model.fit(states)
    history = states[:1, :2]
    pred_before = model.predict_next(history)

    path = tmp_path / "fast_none_predictor.npz"
    model.save(path)
    loaded = FastWindowedTBMDQRCSForecaster.load(path)
    pred_after = loaded.predict_next(history)

    np.testing.assert_allclose(pred_before, pred_after, atol=1e-6)
    assert loaded.get_config()["correction_head_type"] == "none"
    assert loaded.get_metrics()["fit"]["train_base"]["r2"] == pytest.approx(
        loaded.get_metrics()["fit"]["train_corrected"]["r2"]
    )


def test_fast_tplus1_registry_exposes_practical_and_quality_presets():
    from TBMD.experiments.navier_stokes_model_registry import get_fast_tplus1_model_specs
    from TBMD.experiments.navier_stokes_fast_tplus1 import FastWindowedTBMDQRCSForecaster

    specs = {spec.slug: spec for spec in get_fast_tplus1_model_specs()}

    assert set(specs) >= {
        "fast_tplus1_r300_s300",
        "fast_tplus1_r300_s600",
        "fast_tplus1_r300_s600_residual_svd256",
        "fast_tplus1_r300_s600_residual_svd256_scale11",
        "fast_tplus1_r300_s600_residual_svd256_scale165",
        "fast_tplus1_r300_s1000_residual_svd256_scale11",
        "fast_tplus1_r300_s1000_residual_svd256_scale13",
        "fast_tplus1_r300_s1000_patch16_svd32_scale13",
        "fast_tplus1_r300_s1000_tempsmooth_patch16_svd32_a01_scale13",
        "fast_tplus1_r300_s1000_composite_patch24_hf256_scale13_hf04",
        "fast_tplus1_r300_s1000_composite_patch24_hf256_scale13_hf04_seg2048",
        "fast_tplus1_r300_s1000_residual_svd256_scale15",
    }
    assert specs["fast_tplus1_r300_s300"].notes["label"] == "practical"
    assert specs["fast_tplus1_r300_s600"].notes["label"] == "quality-max"
    assert specs["fast_tplus1_r300_s600_residual_svd256"].notes["label"] == "residual-svd-dev-candidate"
    assert specs["fast_tplus1_r300_s600_residual_svd256_scale11"].notes["label"] == "residual-svd-scale-dev-candidate"
    assert specs["fast_tplus1_r300_s600_residual_svd256_scale165"].notes["label"] == "residual-svd-scale-peak-dev-candidate"
    assert specs["fast_tplus1_r300_s1000_residual_svd256_scale11"].notes["label"] == "sensor-overbudget-dev-candidate"
    assert specs["fast_tplus1_r300_s1000_residual_svd256_scale13"].notes["label"] == "sensor-overbudget-cached-multidev-candidate"
    assert specs["fast_tplus1_r300_s1000_patch16_svd32_scale13"].notes["label"] == "sensor-overbudget-patch-residual-multidev-candidate"
    assert (
        specs["fast_tplus1_r300_s1000_tempsmooth_patch16_svd32_a01_scale13"].notes["label"]
        == "sensor-overbudget-temporal-patch-residual-multidev-candidate"
    )
    assert (
        specs[
            "fast_tplus1_r300_s1000_composite_patch24_hf256_scale13_hf04"
        ].notes["label"]
        == "sensor-overbudget-composite-patch-hf-confirmed-candidate"
    )
    assert (
        specs[
            "fast_tplus1_r300_s1000_composite_patch24_hf256_scale13_hf04_seg2048"
        ].notes["label"]
        == "sensor-overbudget-composite-patch-hf-selected-seg2048-candidate"
    )
    assert specs["fast_tplus1_r300_s1000_residual_svd256_scale15"].notes["label"] == "sensor-overbudget-scale-dev-candidate"
    assert specs["fast_tplus1_r300_s300"].factory().config.ranks[-1] == 300
    assert specs["fast_tplus1_r300_s600"].factory().config.n_spatial_sensors == 600
    assert (
        specs["fast_tplus1_r300_s600_residual_svd256"].factory().config.correction_residual_rank
        == 256
    )
    assert (
        specs["fast_tplus1_r300_s600_residual_svd256_scale11"].factory().config.correction_scale
        == pytest.approx(1.1)
    )
    assert (
        specs["fast_tplus1_r300_s600_residual_svd256_scale165"].factory().config.correction_scale
        == pytest.approx(1.65)
    )
    assert (
        specs["fast_tplus1_r300_s1000_residual_svd256_scale11"].factory().config.n_spatial_sensors
        == 1000
    )
    assert (
        specs["fast_tplus1_r300_s1000_residual_svd256_scale11"].factory().config.sensor_decoder
        == "ridge"
    )
    assert (
        specs["fast_tplus1_r300_s1000_residual_svd256_scale13"].factory().config.correction_scale
        == pytest.approx(1.3)
    )
    patch_config = specs["fast_tplus1_r300_s1000_patch16_svd32_scale13"].factory().config
    assert patch_config.correction_head_type == "patch_residual_svd"
    assert patch_config.correction_patch_size == 16
    assert patch_config.correction_patch_residual_rank == 32
    temporal_patch_config = specs[
        "fast_tplus1_r300_s1000_tempsmooth_patch16_svd32_a01_scale13"
    ].factory().config
    assert temporal_patch_config.correction_head_type == "patch_residual_svd"
    assert temporal_patch_config.correction_patch_residual_rank == 32
    assert temporal_patch_config.coefficient_temporal_smoothing_alpha == pytest.approx(0.1)
    composite_config = specs[
        "fast_tplus1_r300_s1000_composite_patch24_hf256_scale13_hf04"
    ].factory().config
    assert composite_config.correction_head_type == "composite_patch_hf_svd"
    assert composite_config.correction_patch_size == 16
    assert composite_config.correction_patch_residual_rank == 24
    assert composite_config.correction_residual_rank == 256
    assert composite_config.correction_scale == pytest.approx(1.3)
    assert composite_config.correction_hf_scale == pytest.approx(0.4)
    assert composite_config.correction_highpass_cutoff_fraction == pytest.approx(0.45)
    composite_seg_config = specs[
        "fast_tplus1_r300_s1000_composite_patch24_hf256_scale13_hf04_seg2048"
    ].factory().config
    assert composite_seg_config.correction_head_type == "composite_patch_hf_svd"
    assert composite_seg_config.max_train_segments == 2048
    assert composite_seg_config.correction_patch_residual_rank == 24
    assert composite_seg_config.correction_residual_rank == 256
    assert (
        specs["fast_tplus1_r300_s1000_residual_svd256_scale15"].factory().config.correction_scale
        == pytest.approx(1.5)
    )
    assert isinstance(specs["fast_tplus1_r300_s300"].factory(), FastWindowedTBMDQRCSForecaster)


def test_fast_tplus1_accuracy_sweep_builds_candidates_and_selects_by_dev():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "tuning"
        / "tune_fast_tplus1_accuracy.py"
    )
    spec = importlib.util.spec_from_file_location(
        "tune_fast_tplus1_accuracy",
        script_path,
    )
    sweep_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep_module)

    candidates = sweep_module.build_candidates(groups=("quick",))
    candidate_names = [candidate.name for candidate in candidates]

    assert "baseline_h7_rt8_r300_s300" in candidate_names
    assert "history10_rt10_r300_s300" in candidate_names
    assert len(candidate_names) == len(set(candidate_names))

    s600_refine_names = [
        candidate.name for candidate in sweep_module.build_candidates(groups=("s600_refine",))
    ]
    assert "quality_s600_reference" in s600_refine_names
    assert "spatial40_r300_s600" in s600_refine_names
    assert "history10_rt10_r300_s600" in s600_refine_names
    assert "rsegment350_s600" in s600_refine_names

    residual_head_candidates = sweep_module.build_candidates(groups=("residual_head",))
    residual_head_names = [candidate.name for candidate in residual_head_candidates]
    assert "residual_svd64_r300_s600" in residual_head_names
    assert any(candidate.correction_residual_rank == 64 for candidate in residual_head_candidates)
    assert any(candidate.correction_alpha == pytest.approx(1e-6) for candidate in residual_head_candidates)
    residual_fine_names = [
        candidate.name for candidate in sweep_module.build_candidates(groups=("residual_head_fine",))
    ]
    assert "residual_svd192_r300_s600" in residual_fine_names
    assert "residual_svd256_r300_s600" in residual_fine_names
    mlp_head_candidates = sweep_module.build_candidates(groups=("mlp_head",))
    mlp_head_names = [candidate.name for candidate in mlp_head_candidates]
    assert "mlp_residual_svd128_h64_e80_r300_s600" in mlp_head_names
    assert any(candidate.correction_head_type == "mlp_residual_svd" for candidate in mlp_head_candidates)
    scale_names = [
        candidate.name for candidate in sweep_module.build_candidates(groups=("correction_scale",))
    ]
    assert "quality_s600_scale0.9" in scale_names
    assert "residual_svd256_scale0.9_r300_s600" in scale_names
    weighted_names = [
        candidate.name for candidate in sweep_module.build_candidates(groups=("weighted_residual",))
    ]
    assert "residual_energy_svd256_scale1.1_r300_s600" in weighted_names
    assert any(candidate.correction_residual_weighting == "residual_energy" for candidate in sweep_module.build_candidates(groups=("weighted_residual",)))
    fine_scale_names = [
        candidate.name for candidate in sweep_module.build_candidates(groups=("correction_scale_fine",))
    ]
    assert "residual_svd256_scale1.15_r300_s600" in fine_scale_names
    assert "residual_svd256_scale1.25_r300_s600" in fine_scale_names
    upper_scale_names = [
        candidate.name for candidate in sweep_module.build_candidates(groups=("correction_scale_upper",))
    ]
    assert "residual_svd256_scale1.4_r300_s600" in upper_scale_names
    assert "residual_svd256_scale1.6_r300_s600" in upper_scale_names
    high_scale_names = [
        candidate.name for candidate in sweep_module.build_candidates(groups=("correction_scale_high",))
    ]
    assert "residual_svd256_scale2.0_r300_s600" in high_scale_names
    assert "residual_svd256_scale2.5_r300_s600" in high_scale_names
    peak_scale_names = [
        candidate.name for candidate in sweep_module.build_candidates(groups=("correction_scale_peak",))
    ]
    assert "residual_svd256_scale1.65_r300_s600" in peak_scale_names
    assert "residual_svd256_scale1.75_r300_s600" in peak_scale_names
    decoder_recovery_candidates = sweep_module.build_candidates(groups=("decoder_recovery",))
    decoder_names = [candidate.name for candidate in decoder_recovery_candidates]
    assert "residual_svd256_scale1.1_fista_i25_r300_s600" in decoder_names
    assert "residual_svd256_scale1.1_fista_i25_r300_s300" in decoder_names
    assert any(candidate.sensor_decoder == "fista" for candidate in decoder_recovery_candidates)
    assert any(candidate.sensor_decoder == "ridge" for candidate in decoder_recovery_candidates)
    decoder_regularization_names = [
        candidate.name for candidate in sweep_module.build_candidates(groups=("decoder_regularization",))
    ]
    assert "decoder_reg_ridge_lam1_r300_s600" in decoder_regularization_names
    assert "decoder_reg_fista_l1_1e-4_i50_r300_s600" in decoder_regularization_names
    sensor_overbudget_names = [
        candidate.name for candidate in sweep_module.build_candidates(groups=("sensor_overbudget",))
    ]
    assert "sensor_overbudget_r300_s800_residual_svd256_scale1.1" in sensor_overbudget_names
    assert "sensor_overbudget_r300_s1000_residual_svd256_scale1.1" in sensor_overbudget_names
    sensor_overbudget_upper_names = [
        candidate.name for candidate in sweep_module.build_candidates(groups=("sensor_overbudget_upper",))
    ]
    assert "sensor_overbudget_r300_s1200_residual_svd256_scale1.1" in sensor_overbudget_upper_names
    assert "sensor_overbudget_r300_s1500_residual_svd256_scale1.1" in sensor_overbudget_upper_names
    sensor_overbudget_scale_names = [
        candidate.name for candidate in sweep_module.build_candidates(groups=("sensor_overbudget_scale",))
    ]
    assert "s1000_residual_svd256_scale1.05" in sensor_overbudget_scale_names
    assert "s1000_residual_svd256_scale1.25" in sensor_overbudget_scale_names
    sensor_overbudget_scale_upper_names = [
        candidate.name for candidate in sweep_module.build_candidates(groups=("sensor_overbudget_scale_upper",))
    ]
    assert "s1000_residual_svd256_scale1.3" in sensor_overbudget_scale_upper_names
    assert "s1000_residual_svd256_scale1.5" in sensor_overbudget_scale_upper_names

    selected = sweep_module.select_best_result(
        [
            {"candidate": "baseline_h7_rt8_r300_s300", "dev_spatial_r2": 0.80},
            {"candidate": "history10_rt10_r300_s300", "dev_spatial_r2": 0.83},
            {"candidate": "spatial40_r300_s300", "dev_spatial_r2": 0.81},
        ]
    )

    assert selected["candidate"] == "history10_rt10_r300_s300"


def test_tbmd_qr_cs_next_step_script_exposes_fast_decoders_and_hybrid_plan():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "tuning"
        / "tune_tbmd_qr_cs_next_step_forecasting.py"
    )
    spec = importlib.util.spec_from_file_location(
        "tune_tbmd_qr_cs_next_step_forecasting",
        script_path,
    )
    sweep_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep_module)

    plan = sweep_module.build_experiment_plan("smoke")
    decoder_names = {candidate["decoder"] for candidate in plan["decoder_candidates"]}
    hybrid_families = {candidate["family"] for candidate in plan["hybrid_candidates"]}

    assert {"current_admm", "ridge_sensor_decoder", "fista_decoder"} <= decoder_names
    assert "hybrid_sensor_conditioned" in hybrid_families
    assert not plan["limits"]["evaluate_official_test"]


def test_tbmd_qr_cs_next_step_ridge_and_fista_decoders_are_batched():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "tuning"
        / "tune_tbmd_qr_cs_next_step_forecasting.py"
    )
    spec = importlib.util.spec_from_file_location(
        "tune_tbmd_qr_cs_next_step_forecasting",
        script_path,
    )
    sweep_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep_module)

    sensing_matrix = np.eye(3, dtype=np.float64)
    measurements = np.array(
        [
            [1.0, -2.0, 0.5],
            [0.25, 0.0, -0.75],
        ],
        dtype=np.float64,
    )

    ridge = sweep_module.RidgeSensorDecoder(ridge_lambda=1e-10).fit(sensing_matrix)
    ridge_coeffs = ridge.decode(measurements)
    np.testing.assert_allclose(ridge_coeffs, measurements, atol=1e-8)

    fista = sweep_module.FistaSensorDecoder(l1_lambda=1e-8, max_iter=25, tol=0.0).fit(
        sensing_matrix
    )
    fista_coeffs = fista.decode(measurements)
    assert fista_coeffs.shape == measurements.shape
    np.testing.assert_allclose(fista_coeffs, measurements, atol=1e-5)


def test_tbmd_qr_cs_next_step_selection_uses_dev_and_tie_breakers():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "tuning"
        / "tune_tbmd_qr_cs_next_step_forecasting.py"
    )
    spec = importlib.util.spec_from_file_location(
        "tune_tbmd_qr_cs_next_step_forecasting",
        script_path,
    )
    sweep_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep_module)

    selected = sweep_module.select_best_result(
        [
            {
                "candidate": "higher_rmse",
                "dev_one_step_r2": 0.8,
                "dev_rmse": 0.5,
                "inference_ms_per_step": 1.0,
                "n_sensors": 30,
            },
            {
                "candidate": "lower_rmse",
                "dev_one_step_r2": 0.8,
                "dev_rmse": 0.4,
                "inference_ms_per_step": 1.0,
                "n_sensors": 30,
            },
            {
                "candidate": "lower_r2",
                "dev_one_step_r2": 0.79,
                "dev_rmse": 0.1,
                "inference_ms_per_step": 0.1,
                "n_sensors": 10,
            },
        ]
    )

    assert selected["candidate"] == "lower_rmse"


def test_fast_tplus1_cached_residual_dev_blocks_hold_out_train_only():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "tuning"
        / "tune_fast_tplus1_cached_residual.py"
    )
    spec = importlib.util.spec_from_file_location(
        "tune_fast_tplus1_cached_residual",
        script_path,
    )
    sweep_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep_module)

    blocks = sweep_module.build_dev_blocks(10, dev_count=2, n_splits=3)

    assert len(blocks) == 3
    for train_idx, dev_idx in blocks:
        assert train_idx.shape == (8,)
        assert dev_idx.shape == (2,)
        assert set(train_idx).isdisjoint(set(dev_idx))


def test_fast_tplus1_cached_residual_candidates_include_robust_variants():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "tuning"
        / "tune_fast_tplus1_cached_residual.py"
    )
    spec = importlib.util.spec_from_file_location(
        "tune_fast_tplus1_cached_residual",
        script_path,
    )
    sweep_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep_module)

    candidates = {candidate.name: candidate for candidate in sweep_module.build_residual_candidates("fast")}

    assert "uniform_svd256_scale1.1" in candidates
    assert "energy_svd256_scale1.1" in candidates
    assert "patch16_rank16_scale1.1" in candidates
    assert "gated_svd256_scale1.3_thr1_str1" in candidates
    assert "innovation_svd256_ir16_scale1.1" in candidates
    assert "coeffcal_svd256_blend0.5_scale1.1" in candidates
    assert "coeffcal_base_blend1" in candidates
    assert "coeffcal_svd128_blend1_scale0.25" in candidates
    assert "hard_svd256_pow1_clip4_scale1.3" in candidates
    assert "hard_patch16_rank32_pow1_clip4_scale1.3" in candidates
    assert "innovgate_svd256_scale1.3_thr0_str1" in candidates
    assert "innovgate_patch16_rank32_scale1.3_thr0_str1" in candidates
    assert "highpass_svd256_cut0.35_scale0.75" in candidates
    assert "highpass_patch16_rank32_cut0.35_scale0.75" in candidates
    assert "hfweight_svd256_cut0.35_scale1.1" in candidates
    assert "hfweight_patch16_rank32_cut0.35_scale1.1" in candidates
    assert "composite_patch16_rank32_hf256_cut0.45_p1.3_hf0.25" in candidates
    assert "coeffdelta_base_blend0.25" in candidates
    assert "coeffdelta_base_ir16_blend0.25" in candidates
    assert "coeffdelta_svd64_ir16_blend0.25_scale0.25" in candidates
    assert "tempsmooth_base_a0.2" in candidates
    assert "tempsmooth_svd256_a0.2_scale1.1" in candidates
    assert "tempsmooth_coeffcal_svd128_a0.2_blend1_scale0.25" in candidates
    assert "tempsmooth_patch16_rank32_a0.1_scale1.3" in candidates
    assert candidates["uniform_svd256_scale1.1"].residual_rank == 256
    assert candidates["energy_svd256_scale1.1"].residual_weighting == "residual_energy"
    assert candidates["patch16_rank16_scale1.1"].head_type == "patch_residual_svd"
    assert candidates["patch16_rank16_scale1.1"].patch_size == 16
    assert candidates["gated_svd256_scale1.3_thr1_str1"].gate_type == "coefficient_rms"
    assert candidates["innovation_svd256_ir16_scale1.1"].innovation_rank == 16
    assert candidates["innovation_svd256_ir16_scale1.1"].innovation_include_norms
    assert candidates["coeffcal_svd256_blend0.5_scale1.1"].coefficient_calibration_type == "ridge"
    assert candidates["coeffcal_svd256_blend0.5_scale1.1"].coefficient_calibration_blend == pytest.approx(0.5)
    assert candidates["coeffcal_base_blend1"].head_type == "none"
    assert candidates["coeffcal_base_blend1"].coefficient_calibration_type == "ridge"
    assert candidates["coeffcal_svd128_blend1_scale0.25"].scale == pytest.approx(0.25)
    assert candidates["hard_svd256_pow1_clip4_scale1.3"].sample_weighting == "hard_frame_rmse"
    assert candidates["hard_patch16_rank32_pow1_clip4_scale1.3"].head_type == "patch_residual_svd"
    assert candidates["innovgate_svd256_scale1.3_thr0_str1"].gate_type == "sensor_innovation_rms"
    assert candidates["highpass_svd256_cut0.35_scale0.75"].residual_target == "highpass"
    assert candidates["highpass_patch16_rank32_cut0.35_scale0.75"].head_type == "patch_residual_svd"
    assert candidates["hfweight_svd256_cut0.35_scale1.1"].residual_weighting == "highpass_energy"
    assert candidates["hfweight_patch16_rank32_cut0.35_scale1.1"].residual_weighting == "highpass_energy"
    assert candidates["composite_patch16_rank32_hf256_cut0.45_p1.3_hf0.25"].head_type == "composite_patch_hf_svd"
    assert candidates["composite_patch16_rank32_hf256_cut0.45_p1.3_hf0.25"].hf_scale == pytest.approx(0.25)
    assert candidates["coeffdelta_base_blend0.25"].coefficient_calibration_type == "delta_ridge"
    assert candidates["coeffdelta_base_blend0.25"].head_type == "none"
    assert candidates["coeffdelta_base_ir16_blend0.25"].coefficient_calibration_innovation_rank == 16
    assert candidates["coeffdelta_base_ir16_blend0.25"].coefficient_calibration_include_norms
    assert candidates["coeffdelta_svd64_ir16_blend0.25_scale0.25"].residual_rank == 64
    assert candidates["tempsmooth_base_a0.2"].coefficient_temporal_smoothing_alpha == pytest.approx(0.2)
    assert candidates["tempsmooth_base_a0.2"].head_type == "none"
    assert candidates["tempsmooth_svd256_a0.2_scale1.1"].residual_rank == 256
    assert candidates["tempsmooth_coeffcal_svd128_a0.2_blend1_scale0.25"].coefficient_calibration_type == "ridge"
    assert candidates["tempsmooth_patch16_rank32_a0.1_scale1.3"].head_type == "patch_residual_svd"
    assert candidates["tempsmooth_patch16_rank32_a0.1_scale1.3"].patch_residual_rank == 32


def test_fast_tplus1_segment_refs_match_selected_window_order():
    from TBMD.experiments.navier_stokes_fast_tplus1 import (
        build_forecast_segment_tensor_with_refs,
    )

    states = np.arange(2 * 5, dtype=np.float64).reshape(2, 5, 1, 1)

    segments, refs = build_forecast_segment_tensor_with_refs(
        states,
        history_length=2,
        stride=1,
        max_segments=4,
    )

    np.testing.assert_array_equal(refs, np.array([[0, 0], [0, 1], [1, 0], [1, 2]]))
    assert segments.shape == (3, 1, 1, 4)
    np.testing.assert_array_equal(segments[:, 0, 0, 0], states[0, 0:3, 0, 0])
    np.testing.assert_array_equal(segments[:, 0, 0, 1], states[0, 1:4, 0, 0])
    np.testing.assert_array_equal(segments[:, 0, 0, 2], states[1, 0:3, 0, 0])
    np.testing.assert_array_equal(segments[:, 0, 0, 3], states[1, 2:5, 0, 0])


def test_fast_tplus1_temporal_coefficient_smoothing_is_causal_and_resets_on_gaps():
    from TBMD.experiments.navier_stokes_fast_tplus1 import (
        smooth_coefficients_by_segment_refs,
    )

    coeffs = np.array([[0.0], [10.0], [20.0], [100.0], [300.0]], dtype=np.float64)
    refs = np.array([[0, 0], [0, 1], [0, 3], [1, 0], [1, 1]], dtype=np.int64)

    smoothed = smooth_coefficients_by_segment_refs(
        coeffs,
        refs,
        alpha=0.5,
        reset_on_gap=True,
    )

    np.testing.assert_allclose(smoothed[:, 0], [0.0, 5.0, 20.0, 100.0, 200.0])


def test_fast_tplus1_cached_residual_filter_can_target_candidate_names():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "tuning"
        / "tune_fast_tplus1_cached_residual.py"
    )
    spec = importlib.util.spec_from_file_location(
        "tune_fast_tplus1_cached_residual",
        script_path,
    )
    sweep_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep_module)

    selected = sweep_module.filter_candidates(
        sweep_module.build_residual_candidates("fast"),
        family="temporal",
        candidate_names=[
            "tempsmooth_base_a0.2",
            "tempsmooth_svd256_a0.2_scale1.1",
        ],
    )

    assert [candidate.name for candidate in selected] == [
        "tempsmooth_base_a0.2",
        "tempsmooth_svd256_a0.2_scale1.1",
    ]


def test_fast_tplus1_cached_residual_selection_uses_multi_dev_robust_score():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "tuning"
        / "tune_fast_tplus1_cached_residual.py"
    )
    spec = importlib.util.spec_from_file_location(
        "tune_fast_tplus1_cached_residual",
        script_path,
    )
    sweep_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep_module)

    selected = sweep_module.select_best_robust_result(
        [
            {
                "candidate": "same_mean_lower_worst",
                "mean_dev_r2": 0.8,
                "worst_dev_r2": 0.72,
                "mean_dev_rmse": 0.4,
            },
            {
                "candidate": "same_mean_higher_worst",
                "mean_dev_r2": 0.8,
                "worst_dev_r2": 0.74,
                "mean_dev_rmse": 0.5,
            },
            {
                "candidate": "lower_mean",
                "mean_dev_r2": 0.79,
                "worst_dev_r2": 0.78,
                "mean_dev_rmse": 0.1,
            },
        ]
    )

    assert selected["candidate"] == "same_mean_higher_worst"


def test_fast_tplus1_cached_residual_computes_hard_bucket_metrics_from_base_errors():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "tuning"
        / "tune_fast_tplus1_cached_residual.py"
    )
    spec = importlib.util.spec_from_file_location(
        "tune_fast_tplus1_cached_residual",
        script_path,
    )
    sweep_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep_module)

    target = np.arange(10, dtype=np.float64).reshape(10, 1, 1)
    base = target.copy()
    base[-2:] += 10.0
    pred = target.copy()
    pred[-2:] += 1.0

    metrics = sweep_module.compute_hard_bucket_metrics(
        target,
        pred,
        base,
        hard_fraction=0.2,
    )

    assert metrics["dev_hard_count"] == 2
    assert metrics["dev_easy_count"] == 2
    assert metrics["dev_hard_rmse"] == pytest.approx(1.0)
    assert metrics["dev_hard_base_rmse"] == pytest.approx(10.0)
    assert metrics["dev_hard_delta_rmse"] == pytest.approx(-9.0)


def test_fast_tplus1_cached_residual_hard_bucket_selection_uses_hard_cases_with_tolerance():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "tuning"
        / "tune_fast_tplus1_cached_residual.py"
    )
    spec = importlib.util.spec_from_file_location(
        "tune_fast_tplus1_cached_residual",
        script_path,
    )
    sweep_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep_module)

    selected = sweep_module.select_best_robust_result(
        [
            {
                "candidate": "best_mean_weak_hard",
                "mean_dev_r2": 0.800,
                "worst_dev_r2": 0.790,
                "mean_dev_rmse": 0.4,
                "mean_dev_hard_r2": 0.40,
                "worst_dev_hard_r2": 0.35,
            },
            {
                "candidate": "near_mean_strong_hard",
                "mean_dev_r2": 0.797,
                "worst_dev_r2": 0.785,
                "mean_dev_rmse": 0.41,
                "mean_dev_hard_r2": 0.55,
                "worst_dev_hard_r2": 0.50,
            },
            {
                "candidate": "too_low_mean_strong_hard",
                "mean_dev_r2": 0.760,
                "worst_dev_r2": 0.755,
                "mean_dev_rmse": 0.5,
                "mean_dev_hard_r2": 0.90,
                "worst_dev_hard_r2": 0.90,
            },
        ],
        objective="hard_bucket",
        mean_r2_tolerance=0.005,
    )

    assert selected["candidate"] == "near_mean_strong_hard"


def test_fast_tplus1_cached_residual_aggregates_optional_structure_metrics():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "tuning"
        / "tune_fast_tplus1_cached_residual.py"
    )
    spec = importlib.util.spec_from_file_location(
        "tune_fast_tplus1_cached_residual",
        script_path,
    )
    sweep_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep_module)

    rows = [
        {
            "candidate": "a",
            "residual_rank": 1,
            "actual_residual_rank": 1,
            "head_type": "global_residual_svd",
            "patch_size": None,
            "patch_residual_rank": None,
            "actual_patch_residual_rank": None,
            "gate_type": "none",
            "gate_threshold": 0.0,
            "gate_strength": 0.0,
            "gate_min": 1.0,
            "innovation_rank": 0,
            "actual_innovation_rank": 0,
            "innovation_include_norms": False,
            "coefficient_calibration_type": "none",
            "coefficient_calibration_blend": 1.0,
            "coefficient_calibration_alpha": 1e-6,
            "base_dev_r2": 0.5,
            "base_dev_rmse": 1.0,
            "scale": 1.0,
            "alpha": 1e-8,
            "residual_target": "highpass",
            "highpass_cutoff_fraction": 0.35,
            "residual_weighting": "uniform",
            "sample_weighting": "uniform",
            "sample_weight_power": 1.0,
            "sample_weight_floor": 0.25,
            "sample_weight_clip": 4.0,
            "sample_weight_min": 1.0,
            "sample_weight_max": 1.0,
            "train_r2": 0.8,
            "dev_r2": 0.7,
            "dev_rmse": 0.4,
            "dev_mae": 0.3,
            "dev_structure_score": 0.6,
            "dev_gradient_rel_frob_err": 0.8,
            "dev_high_frequency_energy_rel_err": 0.9,
            "correction_fit_eval_time_seconds": 0.1,
        },
        {
            "candidate": "a",
            "residual_rank": 1,
            "actual_residual_rank": 1,
            "head_type": "global_residual_svd",
            "patch_size": None,
            "patch_residual_rank": None,
            "actual_patch_residual_rank": None,
            "gate_type": "none",
            "gate_threshold": 0.0,
            "gate_strength": 0.0,
            "gate_min": 1.0,
            "innovation_rank": 0,
            "actual_innovation_rank": 0,
            "innovation_include_norms": False,
            "coefficient_calibration_type": "none",
            "coefficient_calibration_blend": 1.0,
            "coefficient_calibration_alpha": 1e-6,
            "base_dev_r2": 0.5,
            "base_dev_rmse": 1.0,
            "scale": 1.0,
            "alpha": 1e-8,
            "residual_target": "highpass",
            "highpass_cutoff_fraction": 0.35,
            "residual_weighting": "uniform",
            "sample_weighting": "uniform",
            "sample_weight_power": 1.0,
            "sample_weight_floor": 0.25,
            "sample_weight_clip": 4.0,
            "sample_weight_min": 1.0,
            "sample_weight_max": 1.0,
            "train_r2": 0.9,
            "dev_r2": 0.75,
            "dev_rmse": 0.35,
            "dev_mae": 0.25,
            "dev_structure_score": 0.4,
            "dev_gradient_rel_frob_err": 0.6,
            "dev_high_frequency_energy_rel_err": 0.7,
            "correction_fit_eval_time_seconds": 0.2,
        },
    ]

    aggregated = sweep_module.aggregate_candidate_results(rows)

    assert aggregated[0]["residual_target"] == "highpass"
    assert aggregated[0]["mean_dev_structure_score"] == pytest.approx(0.5)
    assert aggregated[0]["mean_dev_gradient_rel_frob_err"] == pytest.approx(0.7)
    assert aggregated[0]["mean_dev_high_frequency_energy_rel_err"] == pytest.approx(0.8)


def test_fast_tplus1_cached_residual_highpass_filter_keeps_controls_and_highpass():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "tuning"
        / "tune_fast_tplus1_cached_residual.py"
    )
    spec = importlib.util.spec_from_file_location(
        "tune_fast_tplus1_cached_residual",
        script_path,
    )
    sweep_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep_module)

    candidates = sweep_module.filter_candidates(
        sweep_module.build_residual_candidates("fast"),
        family="highpass",
    )
    names = {candidate.name for candidate in candidates}

    assert "uniform_svd256_scale1.1" in names
    assert "patch16_rank32_scale1.3" in names
    assert "highpass_svd256_cut0.35_scale0.75" in names
    assert "highpass_patch16_rank32_cut0.35_scale0.75" in names
    assert "hfweight_svd256_cut0.35_scale1.1" in names
    assert "hfweight_patch16_rank32_cut0.35_scale1.1" in names
    assert all(
        "highpass" in name
        or "hfweight" in name
        or name in {"uniform_svd256_scale1.1", "patch16_rank32_scale1.3", "coeffcal_base_blend1"}
        for name in names
    )

    hfweight_names = {
        candidate.name
        for candidate in sweep_module.filter_candidates(
            sweep_module.build_residual_candidates("fast"),
            family="hfweight",
        )
    }
    assert "hfweight_svd256_cut0.35_scale1.1" in hfweight_names
    assert "highpass_svd256_cut0.35_scale0.75" not in hfweight_names

    composite_names = {
        candidate.name
        for candidate in sweep_module.filter_candidates(
            sweep_module.build_residual_candidates("fast"),
            family="composite",
        )
    }
    assert "patch16_rank32_scale1.3" in composite_names
    assert "hfweight_svd256_cut0.45_scale1.2" in composite_names
    assert "composite_patch16_rank32_hf256_cut0.45_p1.3_hf0.25" in composite_names
    assert "highpass_svd256_cut0.35_scale0.75" not in composite_names


def test_fast_tplus1_visualization_metrics_are_single_frame_safe():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "evaluation"
        / "visualize_fast_tplus1_prediction.py"
    )
    spec = importlib.util.spec_from_file_location(
        "visualize_fast_tplus1_prediction",
        script_path,
    )
    viz_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(viz_module)

    target = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    pred = np.array([[1.0, 1.0], [3.0, 5.0]], dtype=np.float64)
    metrics = viz_module.compute_frame_metrics(target, pred)

    assert metrics["rmse"] == pytest.approx(np.sqrt(0.5))
    assert metrics["mae"] == pytest.approx(0.5)
    assert metrics["r2"] == pytest.approx(0.6)


def test_structure_metrics_capture_gradient_spectrum_and_high_frequency_loss():
    from TBMD.experiments.navier_stokes_structure_metrics import (
        compute_structure_metrics,
        radial_power_spectrum,
    )

    checker = np.indices((8, 8)).sum(axis=0) % 2
    target = (2.0 * checker - 1.0).astype(np.float64)[None]
    smooth_pred = np.zeros_like(target)
    perfect_metrics = compute_structure_metrics(target, target)
    smooth_metrics = compute_structure_metrics(target, smooth_pred)
    spectrum = radial_power_spectrum(target)

    assert perfect_metrics["gradient_rel_frob_err"] == pytest.approx(0.0)
    assert perfect_metrics["laplacian_rel_frob_err"] == pytest.approx(0.0)
    assert perfect_metrics["radial_spectrum_rel_err"] == pytest.approx(0.0)
    assert smooth_metrics["gradient_rel_frob_err"] > 0.9
    assert smooth_metrics["high_frequency_energy_rel_err"] > 0.9
    assert smooth_metrics["structure_score"] > perfect_metrics["structure_score"]
    assert spectrum.ndim == 1
    assert np.sum(spectrum) > 0.0


def test_structure_metrics_script_aggregates_worst_frames_by_structure_score():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "evaluation"
        / "evaluate_fast_tplus1_structure_metrics.py"
    )
    spec = importlib.util.spec_from_file_location(
        "evaluate_fast_tplus1_structure_metrics",
        script_path,
    )
    structure_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(structure_module)

    rows = [
        {"trajectory_index": 0, "start_index": 0, "structure_score": 0.2, "r2": 0.9},
        {"trajectory_index": 1, "start_index": 2, "structure_score": 0.8, "r2": 0.5},
        {"trajectory_index": 2, "start_index": 1, "structure_score": 0.4, "r2": 0.7},
    ]
    selected = structure_module.select_representative_frames(rows)

    assert selected["worst"]["trajectory_index"] == 1
    assert selected["best"]["trajectory_index"] == 0
    assert selected["median"]["trajectory_index"] == 2


def test_structure_metrics_aggregate_worst_uses_largest_error_values():
    from TBMD.experiments.navier_stokes_structure_metrics import aggregate_structure_rows

    rows = [
        {
            "trajectory_index": 0,
            "start_index": 0,
            "target_index": 7,
            "mse": 0.1,
            "gradient_rmse": 0.2,
            "laplacian_rmse": 0.3,
            "r2": 0.9,
            "structure_score": 0.4,
        },
        {
            "trajectory_index": 1,
            "start_index": 1,
            "target_index": 8,
            "mse": 0.5,
            "gradient_rmse": 0.7,
            "laplacian_rmse": 0.8,
            "r2": 0.6,
            "structure_score": 0.9,
        },
    ]

    aggregate = aggregate_structure_rows(rows)

    assert aggregate["mse_worst"] == pytest.approx(0.5)
    assert aggregate["gradient_rmse_worst"] == pytest.approx(0.7)
    assert aggregate["laplacian_rmse_worst"] == pytest.approx(0.8)
    assert aggregate["r2_worst"] == pytest.approx(0.6)


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


def test_persistence_forecaster_predicts_current_state_as_t_plus_one_baseline():
    test_states = np.array(
        [
            [[1.0], [2.0], [4.0], [8.0]],
        ],
        dtype=np.float32,
    ).reshape(1, 4, 1, 1)

    forecaster = TrajectoryAwarePersistenceForecaster()
    forecaster.fit(test_states)

    one_step = forecaster.evaluate_one_step(test_states)
    rollout = forecaster.evaluate_rollout(test_states)

    np.testing.assert_array_equal(one_step["pred_spatial"][:, 0, 0], [1.0, 2.0, 4.0])
    np.testing.assert_array_equal(one_step["target_spatial"][:, 0, 0], [2.0, 4.0, 8.0])
    np.testing.assert_array_equal(rollout["pred_spatial"][:, 0, 0], [1.0, 1.0, 1.0])
    assert one_step["n_eval_samples"] == 3
    assert rollout["n_rollout_steps"] == 3


def test_dmd_forecaster_learns_scalar_geometric_transition():
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

    forecaster = TrajectoryAwareDMDForecaster(rank=1, spatial_mean_centering=False)
    forecaster.fit(train_states)

    one_step = forecaster.evaluate_one_step(test_states)
    rollout = forecaster.evaluate_rollout(test_states)

    assert one_step["spatial_r2"] > 0.999999
    assert rollout["spatial_r2"] > 0.999999


def test_stable_dmd_forecaster_clips_operator_spectral_radius():
    train_states = np.array(
        [
            [[1.0], [2.0], [4.0], [8.0]],
            [[3.0], [6.0], [12.0], [24.0]],
        ],
        dtype=np.float32,
    ).reshape(2, 4, 1, 1)

    forecaster = TrajectoryAwareStableDMDForecaster(
        rank=1,
        spatial_mean_centering=False,
        max_spectral_radius=1.0,
    )
    forecaster.fit(train_states)

    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(forecaster._operator))))

    assert spectral_radius <= 1.0 + 1e-8


def test_stable_dmd_forecaster_bounds_unstable_scalar_rollout():
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

    stable = TrajectoryAwareStableDMDForecaster(
        rank=1,
        spatial_mean_centering=False,
        max_spectral_radius=1.0,
    )
    stable.fit(train_states)
    rollout = stable.evaluate_rollout(test_states)

    np.testing.assert_allclose(rollout["pred_spatial"][:, 0, 0], [5.0, 5.0, 5.0], atol=1e-6)


def test_eigenvalue_projected_dmd_clips_unstable_modes_only():
    train_states = np.array(
        [
            [[1.0, 1.0], [0.5, 2.0], [0.25, 4.0], [0.125, 8.0]],
        ],
        dtype=np.float32,
    ).reshape(1, 4, 1, 2)

    forecaster = TrajectoryAwareEigenvalueProjectedDMDForecaster(
        rank=2,
        spatial_mean_centering=False,
        max_spectral_radius=1.0,
    )
    forecaster.fit(train_states)

    eigen_magnitudes = np.sort(np.abs(np.linalg.eigvals(forecaster._operator)))

    assert eigen_magnitudes[-1] <= 1.0 + 1e-8
    assert forecaster._n_projected_modes == 1
    assert np.any(np.isclose(eigen_magnitudes, 0.5, atol=1e-6))


def test_eigenvalue_projected_dmd_bounds_unstable_scalar_rollout():
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

    projected = TrajectoryAwareEigenvalueProjectedDMDForecaster(
        rank=1,
        spatial_mean_centering=False,
        max_spectral_radius=1.0,
    )
    projected.fit(train_states)
    rollout = projected.evaluate_rollout(test_states)

    np.testing.assert_allclose(rollout["pred_spatial"][:, 0, 0], [5.0, 5.0, 5.0], atol=1e-6)


def test_cs_forecaster_runs_qr_cs_lstm_pipeline_on_small_trajectories():
    pattern = np.array([[1.0, -0.5], [0.25, 0.75]], dtype=np.float32)

    def make_states(scales):
        return np.asarray(
            [[scale * pattern for scale in trajectory] for trajectory in scales],
            dtype=np.float32,
        )

    train_states = make_states(
        [
            [1.0, 1.2, 1.44, 1.728, 2.0736],
            [0.7, 0.84, 1.008, 1.2096, 1.45152],
            [1.5, 1.8, 2.16, 2.592, 3.1104],
        ]
    )
    test_states = make_states([[1.1, 1.32, 1.584, 1.9008, 2.28096]])

    forecaster = TrajectoryAwareCSForecaster(
        rank=1,
        n_sensors=2,
        coefficient_source="sensor_cs",
        feature_mode="coeff",
        spatial_mean_centering=False,
        lstm_hidden_size=8,
        lstm_num_layers=1,
        lstm_seq_length=2,
        lstm_num_epochs=3,
        lstm_batch_size=4,
        lstm_val_split=0.34,
        cs_max_iter=20,
        cs_tol=1e-5,
    )
    forecaster.fit(train_states)
    one_step = forecaster.evaluate_one_step(test_states)
    rollout = forecaster.evaluate_rollout(test_states)
    sensor_summary = forecaster.get_sensor_summary()

    assert sensor_summary["requested_sensors"] == 2
    assert sensor_summary["actual_sensors"] == 2
    assert sensor_summary["sensor_selection_method"] == "qr_plus_leverage"
    assert sensor_summary["coefficient_source"] == "sensor_cs"
    assert sensor_summary["cs_initialization"] == "zero"
    assert sensor_summary["fit_cs_mean_iterations"] is not None
    assert sensor_summary["fit_reconstruction"]["rel_frob_err"] >= 0.0
    assert one_step["n_eval_samples"] == 3
    assert rollout["n_rollout_steps"] == 3
    assert np.isfinite(one_step["spatial_r2"])
    assert np.isfinite(rollout["spatial_r2"])


def test_cs_forecaster_can_train_optional_correction_head():
    pattern = np.array([[1.0, -0.5], [0.25, 0.75]], dtype=np.float32)

    def make_states(scales):
        return np.asarray(
            [[scale * pattern for scale in trajectory] for trajectory in scales],
            dtype=np.float32,
        )

    train_states = make_states(
        [
            [1.0, 1.1, 1.21, 1.331, 1.4641],
            [0.8, 0.88, 0.968, 1.0648, 1.17128],
            [1.4, 1.54, 1.694, 1.8634, 2.04974],
        ]
    )
    test_states = make_states([[1.2, 1.32, 1.452, 1.5972, 1.75692]])

    forecaster = TrajectoryAwareCSForecaster(
        rank=1,
        n_sensors=1,
        coefficient_source="sensor_lstsq",
        feature_mode="coeff",
        spatial_mean_centering=False,
        lstm_hidden_size=8,
        lstm_num_layers=1,
        lstm_seq_length=2,
        lstm_num_epochs=3,
        lstm_batch_size=4,
        lstm_val_split=0.34,
        correction_hidden_size=8,
        correction_num_layers=1,
        correction_num_epochs=3,
        correction_batch_size=4,
        correction_val_split=0.34,
        correction_feature_mode="window",
    )
    forecaster.fit(train_states)
    one_step = forecaster.evaluate_one_step(test_states)
    rollout = forecaster.evaluate_rollout(test_states)
    sensor_summary = forecaster.get_sensor_summary()

    assert sensor_summary["correction_enabled"] is True
    assert sensor_summary["correction_feature_mode"] == "window"
    assert sensor_summary["correction_scale"] == pytest.approx(1.0)
    assert sensor_summary["correction_training_history"] is not None
    assert "baseline_spatial_r2" in one_step
    assert "baseline_spatial_r2" in rollout
    assert np.isfinite(one_step["spatial_r2"])
    assert np.isfinite(rollout["spatial_r2"])


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
        "Persistence Forecaster",
        "DMD Forecaster",
        "Linear Forecaster",
        "MLP Forecaster",
        "LSTM Forecaster",
        "LSTM + T+1 Residual Corrector",
        "Multi-Resolution Linear",
    ]
    assert [spec.slug for spec in specs] == [
        "persistence_forecaster",
        "dmd_forecaster",
        "linear_forecaster",
        "mlp_forecaster",
        "lstm_forecaster",
        "lstm_t_plus_1_residual_corrected",
        "multi_resolution_linear",
    ]


def test_model_registry_promotes_stage4_rank_sweep_winner():
    assert DEFAULT_NAVIER_STOKES_RANKS == [64, 64, 15]


def test_multi_resolution_fit_rejects_test_states_to_prevent_leakage_like_api():
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

    forecaster = TrajectoryAwareMultiResolutionForecaster()

    with pytest.raises(TypeError):
        forecaster.fit(train_states, test_states)


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


def test_compute_spatial_metrics_reports_error_diagnostics():
    from TBMD.experiments.navier_stokes_examples import compute_spatial_metrics

    target = np.array(
        [
            [[1.0, 2.0]],
            [[3.0, 4.0]],
        ],
        dtype=np.float32,
    )
    pred = np.array(
        [
            [[1.0, 1.0]],
            [[5.0, 4.0]],
        ],
        dtype=np.float32,
    )

    metrics = compute_spatial_metrics(target, pred)

    assert metrics["mae"] == pytest.approx(0.75)
    assert metrics["max_abs_err"] == pytest.approx(2.0)
    assert metrics["bias"] == pytest.approx(0.25)
    assert metrics["per_sample_rmse"] == pytest.approx([np.sqrt(0.5), np.sqrt(2.0)])


def test_compute_common_horizon_diagnostics_reports_step_and_trajectory_errors():
    test_states = np.zeros((2, 4, 1, 1), dtype=np.float32)
    target = np.array(
        [
            [[[1.0]], [[2.0]], [[3.0]]],
            [[[10.0]], [[20.0]], [[30.0]]],
        ],
        dtype=np.float32,
    )
    pred = np.array(
        [
            [[[1.0]], [[1.0]], [[1.0]]],
            [[[8.0]], [[18.0]], [[25.0]]],
        ],
        dtype=np.float32,
    )
    eval_result = {
        "n_eval_samples": 6,
        "target_spatial": target.reshape(-1, 1, 1),
        "pred_spatial": pred.reshape(-1, 1, 1),
    }

    diagnostics = compute_common_horizon_diagnostics(
        eval_result,
        test_states,
        common_warmup_steps=1,
        worst_count=1,
    )

    assert diagnostics["per_step_rmse"] == pytest.approx([np.sqrt(2.0), np.sqrt(2.5), np.sqrt(14.5)])
    assert diagnostics["per_step_mae"] == pytest.approx([1.0, 1.5, 3.5])
    assert diagnostics["per_trajectory_rmse"] == pytest.approx([np.sqrt(5.0 / 3.0), np.sqrt(11.0)])
    assert diagnostics["worst_trajectory_indices"] == [1]


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
