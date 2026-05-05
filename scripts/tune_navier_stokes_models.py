#!/usr/bin/env python3
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

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from TBMD.config import LatentModalForecasterConfig
from TBMD.experiments import (
    TrajectoryAwareLatentForecaster,
    compute_common_horizon_metrics,
    load_navier_stokes_trajectory_dataset,
)
from TBMD.experiments.navier_stokes_model_registry import (
    DEFAULT_COMMON_WARMUP_STEPS,
    DEFAULT_N_TRAIN_TRAJECTORIES,
    DEFAULT_NAVIER_STOKES_RANKS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("tune_models")

DATA_ROOT = PROJECT_ROOT / "data" / "navier_stokes"
OUTPUT_PATH = PROJECT_ROOT / "scripts" / "plots" / "models_eval" / "tuning_summary.json"


def load_data():
    dataset = load_navier_stokes_trajectory_dataset(DATA_ROOT)
    return dataset.train_states[:DEFAULT_N_TRAIN_TRAJECTORIES], dataset.test_states


def build_candidates():
    return [
        (
            "mlp_h128_l2_e150",
            "mlp",
            LatentModalForecasterConfig(
                ranks=DEFAULT_NAVIER_STOKES_RANKS,
                forecaster_type="mlp",
                delta_forecast=False,
                verbose=False,
                mlp_hidden_size=128,
                mlp_num_layers=2,
                mlp_num_epochs=150,
            ),
        ),
        (
            "mlp_h128_l3_e150",
            "mlp",
            LatentModalForecasterConfig(
                ranks=DEFAULT_NAVIER_STOKES_RANKS,
                forecaster_type="mlp",
                delta_forecast=False,
                verbose=False,
                mlp_hidden_size=128,
                mlp_num_layers=3,
                mlp_num_epochs=150,
            ),
        ),
        (
            "lstm_h128_l2_s6_e150",
            "lstm",
            LatentModalForecasterConfig(
                ranks=DEFAULT_NAVIER_STOKES_RANKS,
                forecaster_type="lstm",
                verbose=False,
                lstm_hidden_size=128,
                lstm_num_layers=2,
                lstm_seq_length=6,
                lstm_num_epochs=150,
            ),
        ),
        (
            "lstm_h128_l2_s7_e150",
            "lstm",
            LatentModalForecasterConfig(
                ranks=DEFAULT_NAVIER_STOKES_RANKS,
                forecaster_type="lstm",
                verbose=False,
                lstm_hidden_size=128,
                lstm_num_layers=2,
                lstm_seq_length=7,
                lstm_num_epochs=150,
            ),
        ),
        (
            "lstm_h128_l2_s8_e150",
            "lstm",
            LatentModalForecasterConfig(
                ranks=DEFAULT_NAVIER_STOKES_RANKS,
                forecaster_type="lstm",
                verbose=False,
                lstm_hidden_size=128,
                lstm_num_layers=2,
                lstm_seq_length=8,
                lstm_num_epochs=150,
            ),
        ),
        (
            "lstm_h128_l2_s6_e200_lr5e4",
            "lstm",
            LatentModalForecasterConfig(
                ranks=DEFAULT_NAVIER_STOKES_RANKS,
                forecaster_type="lstm",
                verbose=False,
                lstm_hidden_size=128,
                lstm_num_layers=2,
                lstm_seq_length=6,
                lstm_num_epochs=200,
                lstm_learning_rate=5e-4,
            ),
        ),
    ]


def main():
    train_states, test_states = load_data()
    logger.info("Train trajectories: %s", train_states.shape)
    logger.info("Test trajectories: %s", test_states.shape)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for candidate_name, family, config in build_candidates():
        logger.info("Running %s", candidate_name)
        t0 = time.time()
        forecaster = TrajectoryAwareLatentForecaster(config=config)
        forecaster.fit(train_states)
        rollout = forecaster.evaluate_rollout(test_states)
        metrics = compute_common_horizon_metrics(
            rollout,
            test_states,
            DEFAULT_COMMON_WARMUP_STEPS,
        )
        elapsed = time.time() - t0
        result = {
            "candidate": candidate_name,
            "family": family,
            "rollout_r2_common": metrics["r2"],
            "rollout_rmse_common": metrics["rmse"],
            "rollout_rel_frob_common": metrics["rel_frob"],
            "n_eval_steps_per_trajectory": metrics["n_eval_steps_per_trajectory"],
            "time": elapsed,
            "config": config.__dict__,
        }
        results.append(result)
        logger.info(
            "%s -> rollout_r2_common=%.4f rmse=%.4f time=%.1fs",
            candidate_name,
            result["rollout_r2_common"],
            result["rollout_rmse_common"],
            elapsed,
        )

    results.sort(key=lambda item: item["rollout_r2_common"], reverse=True)
    best_by_family = {}
    for result in results:
        best_by_family.setdefault(result["family"], result)

    payload = {
        "train_shape": list(train_states.shape),
        "test_shape": list(test_states.shape),
        "common_warmup_steps": DEFAULT_COMMON_WARMUP_STEPS,
        "results": results,
        "best_by_family": best_by_family,
    }
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    logger.info("Tuning summary saved to: %s", OUTPUT_PATH)
    for family, result in best_by_family.items():
        logger.info("Best %s: %s (R2=%.4f)", family, result["candidate"], result["rollout_r2_common"])


if __name__ == "__main__":
    main()
