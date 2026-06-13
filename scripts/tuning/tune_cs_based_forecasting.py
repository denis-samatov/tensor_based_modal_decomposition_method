#!/usr/bin/env python3
"""Controlled sweep for the experimental QR/CS-based forecasting pipeline."""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
logger = logging.getLogger("tune_cs_forecasting")

DATA_ROOT = PROJECT_ROOT / "data" / "navier_stokes"
OUTPUT_PATH = PROJECT_ROOT / "scripts" / "plots" / "models_eval" / "cs_forecasting_sweep_summary.json"
TUNING_DEV_SPLIT = 0.2
SELECTION_METRIC = "rollout_r2_common"


@dataclass(frozen=True)
class CSCandidate:
    name: str
    rank: int
    n_sensors: int
    coefficient_source: str
    cs_max_iter: int = 50
    cs_epsilon_l1: float = 1e-2
    cs_initialization: str = "zero"
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_num_epochs: int = 50
    correction_hidden_size: int = 64
    correction_num_layers: int = 2
    correction_num_epochs: int = 0
    correction_spatial_loss_weight: float = 0.0
    correction_rel_frob_loss_weight: float = 0.0
    correction_feature_mode: str = "last"
    correction_scale: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def factory(self) -> TrajectoryAwareCSForecaster:
        return TrajectoryAwareCSForecaster(
            rank=self.rank,
            n_sensors=self.n_sensors,
            coefficient_source=self.coefficient_source,
            feature_mode="coeff_plus_delta",
            lstm_hidden_size=self.lstm_hidden_size,
            lstm_num_layers=self.lstm_num_layers,
            lstm_seq_length=7,
            lstm_num_epochs=self.lstm_num_epochs,
            lstm_batch_size=32,
            cs_max_iter=self.cs_max_iter,
            cs_epsilon_l1=self.cs_epsilon_l1,
            cs_initialization=self.cs_initialization,
            correction_hidden_size=self.correction_hidden_size,
            correction_num_layers=self.correction_num_layers,
            correction_num_epochs=self.correction_num_epochs,
            correction_spatial_loss_weight=self.correction_spatial_loss_weight,
            correction_rel_frob_loss_weight=self.correction_rel_frob_loss_weight,
            correction_feature_mode=self.correction_feature_mode,
            correction_scale=self.correction_scale,
            verbose=False,
        )

    def as_metadata(self) -> dict[str, Any]:
        payload = dict(self.metadata)
        payload.update(
            {
                "rank": self.rank,
                "n_sensors": self.n_sensors,
                "compression_ratio": self.n_sensors / self.rank,
                "coefficient_source": self.coefficient_source,
                "cs_max_iter": self.cs_max_iter,
                "cs_epsilon_l1": self.cs_epsilon_l1,
                "cs_initialization": self.cs_initialization,
                "lstm_hidden_size": self.lstm_hidden_size,
                "lstm_num_layers": self.lstm_num_layers,
                "lstm_num_epochs": self.lstm_num_epochs,
                "correction_hidden_size": self.correction_hidden_size,
                "correction_num_layers": self.correction_num_layers,
                "correction_num_epochs": self.correction_num_epochs,
                "correction_spatial_loss_weight": self.correction_spatial_loss_weight,
                "correction_rel_frob_loss_weight": self.correction_rel_frob_loss_weight,
                "correction_feature_mode": self.correction_feature_mode,
                "correction_scale": self.correction_scale,
            }
        )
        return payload


def build_candidates(groups=("oracle", "lstsq", "cs")) -> list[CSCandidate]:
    requested = set(groups)
    candidates = [
        CSCandidate(
            name="oracle_full_projection_r30",
            rank=30,
            n_sensors=15,
            coefficient_source="full_projection",
            metadata={"group": "oracle"},
        ),
        CSCandidate(
            name="lstsq_r30_s15",
            rank=30,
            n_sensors=15,
            coefficient_source="sensor_lstsq",
            metadata={"group": "lstsq"},
        ),
        CSCandidate(
            name="lstsq_r30_s25",
            rank=30,
            n_sensors=25,
            coefficient_source="sensor_lstsq",
            metadata={"group": "lstsq"},
        ),
        CSCandidate(
            name="lstsq_r45_s20",
            rank=45,
            n_sensors=20,
            coefficient_source="sensor_lstsq",
            metadata={"group": "lstsq"},
        ),
        CSCandidate(
            name="cs_r30_s15_i50",
            rank=30,
            n_sensors=15,
            coefficient_source="sensor_cs",
            cs_max_iter=50,
            metadata={"group": "cs"},
        ),
        CSCandidate(
            name="cs_r30_s15_i200",
            rank=30,
            n_sensors=15,
            coefficient_source="sensor_cs",
            cs_max_iter=200,
            metadata={"group": "cs"},
        ),
        CSCandidate(
            name="cs_r30_s25_i100",
            rank=30,
            n_sensors=25,
            coefficient_source="sensor_cs",
            cs_max_iter=100,
            metadata={"group": "cs"},
        ),
        CSCandidate(
            name="cs_r30_s28_i100",
            rank=30,
            n_sensors=28,
            coefficient_source="sensor_cs",
            cs_max_iter=100,
            metadata={"group": "cs"},
        ),
        CSCandidate(
            name="cs_r30_s30_i100",
            rank=30,
            n_sensors=30,
            coefficient_source="sensor_cs",
            cs_max_iter=100,
            metadata={"group": "cs"},
        ),
        CSCandidate(
            name="cs_r30_s30_i100_eps001",
            rank=30,
            n_sensors=30,
            coefficient_source="sensor_cs",
            cs_max_iter=100,
            cs_epsilon_l1=1e-3,
            metadata={"group": "cs"},
        ),
        CSCandidate(
            name="cs_r30_s30_i100_eps001_corr_h64_l2_e80",
            rank=30,
            n_sensors=30,
            coefficient_source="sensor_cs",
            cs_max_iter=100,
            cs_epsilon_l1=1e-3,
            correction_hidden_size=64,
            correction_num_layers=2,
            correction_num_epochs=80,
            metadata={"group": "cs"},
        ),
        CSCandidate(
            name="cs_r30_s30_i100_eps001_corr_h128_l2_e80",
            rank=30,
            n_sensors=30,
            coefficient_source="sensor_cs",
            cs_max_iter=100,
            cs_epsilon_l1=1e-3,
            correction_hidden_size=128,
            correction_num_layers=2,
            correction_num_epochs=80,
            metadata={"group": "cs"},
        ),
        CSCandidate(
            name="cs_r30_s30_i100_eps001_lstsq_init",
            rank=30,
            n_sensors=30,
            coefficient_source="sensor_cs",
            cs_max_iter=100,
            cs_epsilon_l1=1e-3,
            cs_initialization="sensor_lstsq",
            metadata={"group": "cs"},
        ),
        CSCandidate(
            name="cs_r45_s20_i100",
            rank=45,
            n_sensors=20,
            coefficient_source="sensor_cs",
            cs_max_iter=100,
            metadata={"group": "cs"},
        ),
        CSCandidate(
            name="cs_r45_s45_i100",
            rank=45,
            n_sensors=45,
            coefficient_source="sensor_cs",
            cs_max_iter=100,
            metadata={"group": "cs"},
        ),
        CSCandidate(
            name="cs_r45_s45_i100_eps001_corr_h128_l2_e80_spatial025_rel01",
            rank=45,
            n_sensors=45,
            coefficient_source="sensor_cs",
            cs_max_iter=100,
            cs_epsilon_l1=1e-3,
            correction_hidden_size=128,
            correction_num_layers=2,
            correction_num_epochs=80,
            correction_spatial_loss_weight=0.25,
            correction_rel_frob_loss_weight=0.1,
            metadata={"group": "cs"},
        ),
        CSCandidate(
            name="cs_r45_s45_i100_eps001_corr_h256_l2_e120_spatial025_rel01",
            rank=45,
            n_sensors=45,
            coefficient_source="sensor_cs",
            cs_max_iter=100,
            cs_epsilon_l1=1e-3,
            correction_hidden_size=256,
            correction_num_layers=2,
            correction_num_epochs=120,
            correction_spatial_loss_weight=0.25,
            correction_rel_frob_loss_weight=0.1,
            metadata={"group": "cs"},
        ),
        CSCandidate(
            name="cs_r45_s45_i100_eps001_lstm_h256_l2_corr_h256_l2_e120_spatial025_rel01",
            rank=45,
            n_sensors=45,
            coefficient_source="sensor_cs",
            cs_max_iter=100,
            cs_epsilon_l1=1e-3,
            lstm_hidden_size=256,
            lstm_num_layers=2,
            correction_hidden_size=256,
            correction_num_layers=2,
            correction_num_epochs=120,
            correction_spatial_loss_weight=0.25,
            correction_rel_frob_loss_weight=0.1,
            metadata={"group": "cs"},
        ),
        CSCandidate(
            name="cs_r45_s45_i100_eps001_lstm_h256_l2_corr_h512_l2_e120_spatial025_rel01",
            rank=45,
            n_sensors=45,
            coefficient_source="sensor_cs",
            cs_max_iter=100,
            cs_epsilon_l1=1e-3,
            lstm_hidden_size=256,
            lstm_num_layers=2,
            correction_hidden_size=512,
            correction_num_layers=2,
            correction_num_epochs=120,
            correction_spatial_loss_weight=0.25,
            correction_rel_frob_loss_weight=0.1,
            metadata={"group": "cs"},
        ),
        CSCandidate(
            name="cs_r45_s45_i100_eps001_lstm_h256_l2_corr_h512_l2_e120_latent",
            rank=45,
            n_sensors=45,
            coefficient_source="sensor_cs",
            cs_max_iter=100,
            cs_epsilon_l1=1e-3,
            lstm_hidden_size=256,
            lstm_num_layers=2,
            correction_hidden_size=512,
            correction_num_layers=2,
            correction_num_epochs=120,
            correction_spatial_loss_weight=0.0,
            correction_rel_frob_loss_weight=0.0,
            metadata={"group": "cs"},
        ),
        CSCandidate(
            name="cs_r45_s60_i100_eps001_lstm_h256_l2_corr_h512_l2_e120_latent",
            rank=45,
            n_sensors=60,
            coefficient_source="sensor_cs",
            cs_max_iter=100,
            cs_epsilon_l1=1e-3,
            lstm_hidden_size=256,
            lstm_num_layers=2,
            correction_hidden_size=512,
            correction_num_layers=2,
            correction_num_epochs=120,
            correction_spatial_loss_weight=0.0,
            correction_rel_frob_loss_weight=0.0,
            metadata={"group": "cs"},
        ),
        CSCandidate(
            name="cs_r45_s45_i100_eps001_lstm_h256_l2_corr_h512_l2_e120_latent_scale125",
            rank=45,
            n_sensors=45,
            coefficient_source="sensor_cs",
            cs_max_iter=100,
            cs_epsilon_l1=1e-3,
            lstm_hidden_size=256,
            lstm_num_layers=2,
            correction_hidden_size=512,
            correction_num_layers=2,
            correction_num_epochs=120,
            correction_spatial_loss_weight=0.0,
            correction_rel_frob_loss_weight=0.0,
            correction_scale=1.25,
            metadata={"group": "cs"},
        ),
        CSCandidate(
            name="cs_r45_s75_i100_eps001_lstm_h256_l2_corr_h512_l2_e120_latent",
            rank=45,
            n_sensors=75,
            coefficient_source="sensor_cs",
            cs_max_iter=100,
            cs_epsilon_l1=1e-3,
            lstm_hidden_size=256,
            lstm_num_layers=2,
            correction_hidden_size=512,
            correction_num_layers=2,
            correction_num_epochs=120,
            correction_spatial_loss_weight=0.0,
            correction_rel_frob_loss_weight=0.0,
            metadata={"group": "cs"},
        ),
        CSCandidate(
            name="cs_r45_s45_i100_eps001_corr_h128_l2_e80_window_spatial025_rel01",
            rank=45,
            n_sensors=45,
            coefficient_source="sensor_cs",
            cs_max_iter=100,
            cs_epsilon_l1=1e-3,
            correction_hidden_size=128,
            correction_num_layers=2,
            correction_num_epochs=80,
            correction_spatial_loss_weight=0.25,
            correction_rel_frob_loss_weight=0.1,
            correction_feature_mode="window",
            metadata={"group": "cs"},
        ),
        CSCandidate(
            name="cs_r45_s45_i100_eps001_corr_h256_l2_e120_window_spatial025_rel01",
            rank=45,
            n_sensors=45,
            coefficient_source="sensor_cs",
            cs_max_iter=100,
            cs_epsilon_l1=1e-3,
            correction_hidden_size=256,
            correction_num_layers=2,
            correction_num_epochs=120,
            correction_spatial_loss_weight=0.25,
            correction_rel_frob_loss_weight=0.1,
            correction_feature_mode="window",
            metadata={"group": "cs"},
        ),
        CSCandidate(
            name="cs_r45_s45_i100_eps001_corr_h256_l3_e120_spatial025_rel01",
            rank=45,
            n_sensors=45,
            coefficient_source="sensor_cs",
            cs_max_iter=100,
            cs_epsilon_l1=1e-3,
            correction_hidden_size=256,
            correction_num_layers=3,
            correction_num_epochs=120,
            correction_spatial_loss_weight=0.25,
            correction_rel_frob_loss_weight=0.1,
            metadata={"group": "cs"},
        ),
        CSCandidate(
            name="cs_r45_s45_i100_eps0003_corr_h128_l2_e80_spatial025_rel01",
            rank=45,
            n_sensors=45,
            coefficient_source="sensor_cs",
            cs_max_iter=100,
            cs_epsilon_l1=3e-4,
            correction_hidden_size=128,
            correction_num_layers=2,
            correction_num_epochs=80,
            correction_spatial_loss_weight=0.25,
            correction_rel_frob_loss_weight=0.1,
            metadata={"group": "cs"},
        ),
    ]
    return [candidate for candidate in candidates if candidate.metadata["group"] in requested]


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


def evaluate_candidate(candidate, train_states, eval_states, common_warmup_steps):
    logger.info("Running %s", candidate.name)
    start = time.time()
    model = candidate.factory()
    model.fit(train_states)
    fit_time = time.time() - start
    metrics = evaluate_model(model, eval_states, common_warmup_steps)
    elapsed = time.time() - start
    result = {
        "candidate": candidate.name,
        "rank": candidate.rank,
        "n_sensors": candidate.n_sensors,
        "coefficient_source": candidate.coefficient_source,
        "fit_time": fit_time,
        "time": elapsed,
        "metadata": candidate.as_metadata(),
        "sensor_summary": model.get_sensor_summary(),
        **metrics,
    }
    logger.info(
        "%s -> rollout_r2_common=%.4f rmse=%.4f time=%.1fs",
        candidate.name,
        result["rollout_r2_common"],
        result["rollout_rmse_common"],
        elapsed,
    )
    return result


def sort_results_for_selection(results):
    return sorted(results, key=lambda item: item[SELECTION_METRIC], reverse=True)


def select_best_result(results):
    if not results:
        raise ValueError("Cannot select from an empty result list")
    return sort_results_for_selection(results)[0]


def best_results_by_group(results):
    by_group = {}
    for result in sort_results_for_selection(results):
        by_group.setdefault(result["metadata"]["group"], result)
    return by_group


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["oracle", "lstsq", "cs"],
        choices=["oracle", "lstsq", "cs"],
    )
    parser.add_argument("--n-train-trajectories", type=int, default=160)
    parser.add_argument("--n-test-trajectories", type=int, default=40)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--skip-official-test", action="store_true")
    parser.add_argument("--common-warmup-steps", type=int, default=DEFAULT_COMMON_WARMUP_STEPS)
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
    candidates = build_candidates(groups=tuple(args.groups))
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]

    logger.info("CS sweep train trajectories: %s", tuning_train_states.shape)
    logger.info("CS sweep dev trajectories: %s", tuning_dev_states.shape)
    logger.info("CS sweep official-test subset: %s", official_test_states.shape)

    dev_results = sort_results_for_selection(
        [
            evaluate_candidate(
                candidate,
                tuning_train_states,
                tuning_dev_states,
                args.common_warmup_steps,
            )
            for candidate in candidates
        ]
    )
    selected_dev_result = select_best_result(dev_results)
    selected_candidate = next(
        candidate for candidate in candidates if candidate.name == selected_dev_result["candidate"]
    )
    logger.info(
        "Selected %s by dev %s=%.6f",
        selected_dev_result["candidate"],
        SELECTION_METRIC,
        selected_dev_result[SELECTION_METRIC],
    )

    final_test_result = None
    if not args.skip_official_test:
        final_test_result = evaluate_candidate(
            selected_candidate,
            tuning_train_states,
            official_test_states,
            args.common_warmup_steps,
        )

    config_payload = vars(args).copy()
    config_payload["output"] = str(config_payload["output"])
    payload = {
        "protocol": (
            "Select an experimental CS-based forecasting candidate on a train/dev "
            "trajectory split, then evaluate the selected candidate once on a "
            "held-out official-test subset."
        ),
        "selection_metric": SELECTION_METRIC,
        "dev_split": TUNING_DEV_SPLIT,
        "train_shape": list(all_train_states.shape),
        "tuning_train_shape": list(tuning_train_states.shape),
        "tuning_dev_shape": list(tuning_dev_states.shape),
        "official_test_shape": list(official_test_states.shape),
        "common_warmup_steps": args.common_warmup_steps,
        "config": config_payload,
        "candidate_grid": [
            {"candidate": candidate.name, "metadata": candidate.as_metadata()}
            for candidate in candidates
        ],
        "dev_results": dev_results,
        "best_by_group": best_results_by_group(dev_results),
        "selected_candidate": selected_dev_result["candidate"],
        "selected_dev_result": selected_dev_result,
        "final_test_result": final_test_result,
        "results": dev_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("CS forecasting sweep summary saved to: %s", args.output)


if __name__ == "__main__":
    main()
