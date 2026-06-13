#!/usr/bin/env python3
"""Evaluate the experimental QR/CS-based Navier-Stokes forecasting pipeline."""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MPL_CACHE_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from TBMD.experiments import (
    TrajectoryAwareCSForecaster,
    compute_common_horizon_diagnostics,
    compute_common_horizon_metrics,
    load_navier_stokes_trajectory_dataset,
    split_train_dev_trajectories,
)
from TBMD.experiments.navier_stokes_model_registry import (
    DEFAULT_COMMON_WARMUP_STEPS,
    DEFAULT_N_TRAIN_TRAJECTORIES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("cs_forecasting")

DATA_ROOT = PROJECT_ROOT / "data" / "navier_stokes"
OUTPUT_PATH = PROJECT_ROOT / "scripts" / "plots" / "models_eval" / "cs_forecasting_summary.json"
TUNING_DEV_SPLIT = 0.2


def load_data(n_train_trajectories: int, n_test_trajectories: int | None):
    dataset = load_navier_stokes_trajectory_dataset(DATA_ROOT)
    train_states = dataset.train_states[:n_train_trajectories]
    test_states = dataset.test_states
    if n_test_trajectories is not None:
        test_states = test_states[:n_test_trajectories]
    return train_states, test_states


def evaluate_model(model, states, common_warmup_steps):
    one_step = model.evaluate_one_step(states)
    rollout = model.evaluate_rollout(states)
    one_step_common = compute_common_horizon_metrics(
        one_step,
        states,
        common_warmup_steps,
    )
    rollout_common = compute_common_horizon_metrics(
        rollout,
        states,
        common_warmup_steps,
    )
    rollout_diagnostics = compute_common_horizon_diagnostics(
        rollout,
        states,
        common_warmup_steps,
    )
    metrics = {
        "one_step_r2_common": one_step_common["r2"],
        "one_step_rmse_common": one_step_common["rmse"],
        "rollout_r2_common": rollout_common["r2"],
        "rollout_rmse_common": rollout_common["rmse"],
        "rollout_mae_common": rollout_common["mae"],
        "rollout_rel_frob_common": rollout_common["rel_frob"],
        "stability_gap_r2_common": one_step_common["r2"] - rollout_common["r2"],
        "first_step_rmse": rollout_diagnostics["per_step_rmse"][0],
        "last_step_rmse": rollout_diagnostics["per_step_rmse"][-1],
        "worst_trajectory_indices": rollout_diagnostics["worst_trajectory_indices"],
        "worst_trajectory_rmse": rollout_diagnostics["worst_trajectory_rmse"],
        "n_eval_steps_per_trajectory": rollout_common["n_eval_steps_per_trajectory"],
    }
    if "baseline_pred_spatial" in one_step:
        baseline_one_step = dict(one_step)
        baseline_one_step["pred_spatial"] = one_step["baseline_pred_spatial"]
        baseline_one_step_common = compute_common_horizon_metrics(
            baseline_one_step,
            states,
            common_warmup_steps,
        )
        metrics["baseline_one_step_r2_common"] = baseline_one_step_common["r2"]
        metrics["one_step_correction_gain_r2_common"] = (
            one_step_common["r2"] - baseline_one_step_common["r2"]
        )
    if "baseline_pred_spatial" in rollout:
        baseline_rollout = dict(rollout)
        baseline_rollout["pred_spatial"] = rollout["baseline_pred_spatial"]
        baseline_rollout_common = compute_common_horizon_metrics(
            baseline_rollout,
            states,
            common_warmup_steps,
        )
        metrics["baseline_rollout_r2_common"] = baseline_rollout_common["r2"]
        metrics["rollout_correction_gain_r2_common"] = (
            rollout_common["r2"] - baseline_rollout_common["r2"]
        )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, default=45)
    parser.add_argument("--n-sensors", type=int, default=45)
    parser.add_argument(
        "--coefficient-source",
        choices=["sensor_cs", "sensor_lstsq", "full_projection"],
        default="sensor_cs",
        help="sensor_cs is the faithful QR+ADMM path; sensor_lstsq is a fast ablation.",
    )
    parser.add_argument("--n-train-trajectories", type=int, default=640)
    parser.add_argument("--n-test-trajectories", type=int, default=200)
    parser.add_argument("--lstm-hidden-size", type=int, default=256)
    parser.add_argument("--lstm-num-layers", type=int, default=2)
    parser.add_argument("--lstm-seq-length", type=int, default=7)
    parser.add_argument("--lstm-num-epochs", type=int, default=50)
    parser.add_argument("--lstm-batch-size", type=int, default=32)
    parser.add_argument("--cs-max-iter", type=int, default=100)
    parser.add_argument("--cs-tol", type=float, default=1e-4)
    parser.add_argument("--cs-epsilon-l1", type=float, default=1e-3)
    parser.add_argument(
        "--cs-initialization",
        choices=["zero", "sensor_lstsq"],
        default="zero",
        help="Initialization for the ADMM coefficient solve.",
    )
    parser.add_argument("--correction-hidden-size", type=int, default=512)
    parser.add_argument("--correction-num-layers", type=int, default=2)
    parser.add_argument("--correction-num-epochs", type=int, default=120)
    parser.add_argument("--correction-batch-size", type=int, default=32)
    parser.add_argument("--correction-spatial-loss-weight", type=float, default=0.0)
    parser.add_argument("--correction-rel-frob-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--correction-feature-mode",
        choices=["last", "window"],
        default="last",
        help="Inputs for the residual correction head.",
    )
    parser.add_argument("--correction-scale", type=float, default=1.0)
    parser.add_argument("--common-warmup-steps", type=int, default=DEFAULT_COMMON_WARMUP_STEPS)
    parser.add_argument("--skip-official-test", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_train_trajectories > DEFAULT_N_TRAIN_TRAJECTORIES:
        raise ValueError(
            f"n_train_trajectories cannot exceed {DEFAULT_N_TRAIN_TRAJECTORIES}"
        )

    all_train_states, official_test_states = load_data(
        args.n_train_trajectories,
        args.n_test_trajectories,
    )
    tuning_train_states, tuning_dev_states = split_train_dev_trajectories(
        all_train_states,
        dev_split=TUNING_DEV_SPLIT,
    )
    logger.info("CS train trajectories: %s", tuning_train_states.shape)
    logger.info("CS dev trajectories: %s", tuning_dev_states.shape)
    logger.info("CS official-test subset: %s", official_test_states.shape)

    model = TrajectoryAwareCSForecaster(
        rank=args.rank,
        n_sensors=args.n_sensors,
        coefficient_source=args.coefficient_source,
        feature_mode="coeff_plus_delta",
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_num_layers=args.lstm_num_layers,
        lstm_seq_length=args.lstm_seq_length,
        lstm_num_epochs=args.lstm_num_epochs,
        lstm_batch_size=args.lstm_batch_size,
        cs_max_iter=args.cs_max_iter,
        cs_tol=args.cs_tol,
        cs_epsilon_l1=args.cs_epsilon_l1,
        cs_initialization=args.cs_initialization,
        correction_hidden_size=args.correction_hidden_size,
        correction_num_layers=args.correction_num_layers,
        correction_num_epochs=args.correction_num_epochs,
        correction_batch_size=args.correction_batch_size,
        correction_spatial_loss_weight=args.correction_spatial_loss_weight,
        correction_rel_frob_loss_weight=args.correction_rel_frob_loss_weight,
        correction_feature_mode=args.correction_feature_mode,
        correction_scale=args.correction_scale,
        verbose=args.verbose,
    )

    start = time.time()
    model.fit(tuning_train_states)
    fit_time = time.time() - start
    logger.info("Fit complete in %.1fs", fit_time)

    dev_start = time.time()
    dev_result = evaluate_model(model, tuning_dev_states, args.common_warmup_steps)
    dev_result["time"] = time.time() - dev_start
    logger.info(
        "Dev rollout_r2_common=%.4f rmse=%.4f",
        dev_result["rollout_r2_common"],
        dev_result["rollout_rmse_common"],
    )

    final_test_result = None
    if not args.skip_official_test:
        final_start = time.time()
        final_test_result = evaluate_model(
            model,
            official_test_states,
            args.common_warmup_steps,
        )
        final_test_result["time"] = time.time() - final_start
        logger.info(
            "Test rollout_r2_common=%.4f rmse=%.4f",
            final_test_result["rollout_r2_common"],
            final_test_result["rollout_rmse_common"],
        )

    config_payload = vars(args).copy()
    config_payload["output"] = str(config_payload["output"])
    payload = {
        "protocol": (
            "Experimental CS-based forecasting: QR sensor placement, sparse sensor "
            "measurements, coefficient recovery, LSTM coefficient rollout, dictionary reconstruction."
        ),
        "coefficient_source": args.coefficient_source,
        "train_shape": list(all_train_states.shape),
        "tuning_train_shape": list(tuning_train_states.shape),
        "tuning_dev_shape": list(tuning_dev_states.shape),
        "official_test_shape": list(official_test_states.shape),
        "common_warmup_steps": args.common_warmup_steps,
        "config": config_payload,
        "sensor_summary": model.get_sensor_summary(),
        "fit_time": fit_time,
        "dev_result": dev_result,
        "final_test_result": final_test_result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("CS forecasting summary saved to: %s", args.output)


if __name__ == "__main__":
    main()
