#!/usr/bin/env python3
"""Stage 6b: Scheduled Sampling tuning.

Tests the impact of Scheduled Sampling on the Stage 6 optimal configuration.
"""

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

import numpy as np

from TBMD.config import LatentModalForecasterConfig
from TBMD.experiments import (
    TrajectoryAwareLatentForecaster,
    TrajectoryAwareResidualCorrectedForecaster,
    compute_common_horizon_diagnostics,
    compute_common_horizon_metrics,
    load_navier_stokes_trajectory_dataset,
    split_train_dev_trajectories,
)
from TBMD.experiments.navier_stokes_model_registry import (
    DEFAULT_COMMON_WARMUP_STEPS,
    DEFAULT_N_TRAIN_TRAJECTORIES,
    DEFAULT_NAVIER_STOKES_RANKS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("stage6b_scheduled_sampling")

DATA_ROOT = PROJECT_ROOT / "data" / "navier_stokes"
OUTPUT_PATH = PROJECT_ROOT / "scripts" / "plots" / "models_eval" / "stage6b_summary.json"
TUNING_DEV_SPLIT = 0.2
SELECTION_METRIC = "rollout_r2_common"


@dataclass(frozen=True)
class Stage6bCandidate:
    name: str
    ranks: list[int]
    delta_forecast: bool = False
    feature_mode: str = "latent_plus_delta"
    lstm_seq_length: int = 10
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_num_epochs: int = 150
    lstm_use_scheduled_sampling: bool = False
    lstm_ss_unroll_steps: int = 5
    lstm_ss_decay_rate: float = 0.01
    lstm_ss_min_prob: float = 0.0
    correction_hidden_size: int = 64
    correction_num_layers: int = 2
    correction_num_epochs: int = 120
    use_correction: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def r3(self) -> int:
        return int(self.ranks[2])

    def config(self) -> LatentModalForecasterConfig:
        return LatentModalForecasterConfig(
            ranks=list(self.ranks),
            forecaster_type="lstm",
            verbose=False,
            delta_forecast=self.delta_forecast,
            lstm_hidden_size=self.lstm_hidden_size,
            lstm_num_layers=self.lstm_num_layers,
            lstm_seq_length=self.lstm_seq_length,
            lstm_num_epochs=self.lstm_num_epochs,
            lstm_use_scheduled_sampling=self.lstm_use_scheduled_sampling,
            lstm_ss_unroll_steps=self.lstm_ss_unroll_steps,
            lstm_ss_decay_rate=self.lstm_ss_decay_rate,
            lstm_ss_min_prob=self.lstm_ss_min_prob,
        )

    def as_metadata(self) -> dict[str, Any]:
        payload = dict(self.metadata)
        payload.update(
            {
                "ranks": list(self.ranks),
                "r3": self.r3,
                "delta_forecast": self.delta_forecast,
                "feature_mode": self.feature_mode,
                "lstm": {
                    "hidden_size": self.lstm_hidden_size,
                    "num_layers": self.lstm_num_layers,
                    "seq_length": self.lstm_seq_length,
                    "num_epochs": self.lstm_num_epochs,
                    "use_scheduled_sampling": self.lstm_use_scheduled_sampling,
                    "ss_unroll_steps": self.lstm_ss_unroll_steps,
                    "ss_decay_rate": self.lstm_ss_decay_rate,
                },
            }
        )
        return payload


def build_candidates() -> list[Stage6bCandidate]:
    candidates: list[Stage6bCandidate] = []

    # Baseline (Stage 6 winner)
    candidates.append(
        Stage6bCandidate(
            name="baseline_seq10",
            ranks=list(DEFAULT_NAVIER_STOKES_RANKS),
            delta_forecast=False,
            feature_mode="latent_plus_delta",
            lstm_seq_length=10,
            lstm_use_scheduled_sampling=False,
        )
    )

    # SS Unroll 5
    candidates.append(
        Stage6bCandidate(
            name="seq10_ss_unroll5",
            ranks=list(DEFAULT_NAVIER_STOKES_RANKS),
            delta_forecast=False,
            feature_mode="latent_plus_delta",
            lstm_seq_length=10,
            lstm_use_scheduled_sampling=True,
            lstm_ss_unroll_steps=5,
            lstm_ss_decay_rate=0.01,
        )
    )

    # SS Unroll 10
    candidates.append(
        Stage6bCandidate(
            name="seq10_ss_unroll10",
            ranks=list(DEFAULT_NAVIER_STOKES_RANKS),
            delta_forecast=False,
            feature_mode="latent_plus_delta",
            lstm_seq_length=10,
            lstm_use_scheduled_sampling=True,
            lstm_ss_unroll_steps=10,
            lstm_ss_decay_rate=0.01,
        )
    )

    return candidates


def _build_forecaster(candidate: Stage6bCandidate):
    if candidate.use_correction:
        return TrajectoryAwareResidualCorrectedForecaster(
            config=candidate.config(),
            feature_mode=candidate.feature_mode,
            correction_hidden_size=candidate.correction_hidden_size,
            correction_num_layers=candidate.correction_num_layers,
            correction_dropout=0.0,
            correction_learning_rate=1e-3,
            correction_weight_decay=1e-5,
            correction_num_epochs=candidate.correction_num_epochs,
            correction_batch_size=32,
            correction_val_split=0.2,
            correction_early_stopping_patience=20,
            correction_latent_loss_weight=1.0,
            correction_spatial_loss_weight=0.0,
            correction_rel_frob_loss_weight=0.0,
        )

    return TrajectoryAwareLatentForecaster(
        config=candidate.config(),
        feature_mode=candidate.feature_mode,
    )


def load_data(n_train_trajectories: int = DEFAULT_N_TRAIN_TRAJECTORIES):
    dataset = load_navier_stokes_trajectory_dataset(DATA_ROOT)
    return dataset.train_states[:n_train_trajectories], dataset.test_states


def sort_results_for_selection(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(results, key=lambda item: item[SELECTION_METRIC], reverse=True)


def select_best_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("Cannot select from an empty result list")
    return sort_results_for_selection(results)[0]


def evaluate_candidate(
    candidate: Stage6bCandidate,
    train_states: np.ndarray,
    test_states: np.ndarray,
) -> dict[str, Any]:
    logger.info("Running %s", candidate.name)
    start = time.time()

    model = _build_forecaster(candidate)
    model.fit(train_states)

    one_step = model.evaluate_one_step(test_states)
    rollout = model.evaluate_rollout(test_states)

    one_step_common = compute_common_horizon_metrics(
        one_step, test_states, DEFAULT_COMMON_WARMUP_STEPS
    )
    rollout_common = compute_common_horizon_metrics(
        rollout, test_states, DEFAULT_COMMON_WARMUP_STEPS
    )
    rollout_diagnostics = compute_common_horizon_diagnostics(
        rollout, test_states, DEFAULT_COMMON_WARMUP_STEPS
    )

    elapsed = time.time() - start

    result = {
        "candidate": candidate.name,
        "ranks": list(candidate.ranks),
        "r3": candidate.r3,
        "delta_forecast": candidate.delta_forecast,
        "feature_mode": candidate.feature_mode,
        "one_step_r2_common": one_step_common["r2"],
        "one_step_rmse_common": one_step_common["rmse"],
        "rollout_r2_common": rollout_common["r2"],
        "rollout_rmse_common": rollout_common["rmse"],
        "rollout_mae_common": rollout_common["mae"],
        "rollout_rel_frob_common": rollout_common["rel_frob"],
        "stability_gap_r2_common": one_step_common["r2"] - rollout_common["r2"],
        "first_step_rmse": rollout_diagnostics["per_step_rmse"][0],
        "first_step_autoregressive_rmse": rollout_diagnostics["per_step_rmse"][0],
        "last_step_rmse": rollout_diagnostics["per_step_rmse"][-1],
        "time": elapsed,
        "metadata": candidate.as_metadata(),
    }
    logger.info(
        "%s -> rollout_r2_common=%.4f rmse=%.4f first_step_rmse=%.4f gap=%.4f time=%.1fs",
        candidate.name,
        result["rollout_r2_common"],
        result["rollout_rmse_common"],
        result["first_step_rmse"],
        result["stability_gap_r2_common"],
        elapsed,
    )
    return result


def main() -> None:
    argparse.ArgumentParser().parse_args()
    all_train_states, official_test_states = load_data(DEFAULT_N_TRAIN_TRAJECTORIES)
    tuning_train_states, tuning_dev_states = split_train_dev_trajectories(
        all_train_states,
        dev_split=TUNING_DEV_SPLIT,
    )

    candidates = build_candidates()
    logger.info("Running %d candidates", len(candidates))

    dev_results = sort_results_for_selection(
        [evaluate_candidate(c, tuning_train_states, tuning_dev_states) for c in candidates]
    )

    selected_dev_result = select_best_result(dev_results)
    selected_candidate = next(c for c in candidates if c.name == selected_dev_result["candidate"])

    logger.info("Selected %s", selected_dev_result["candidate"])

    final_test_result = evaluate_candidate(
        selected_candidate,
        all_train_states,
        official_test_states,
    )

    payload = {
        "stage": "stage6b_scheduled_sampling",
        "dev_results": dev_results,
        "selected_dev_result": selected_dev_result,
        "final_test_result": final_test_result,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    logger.info("Stage 6b summary saved to: %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
