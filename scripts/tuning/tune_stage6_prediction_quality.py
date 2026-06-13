#!/usr/bin/env python3
"""Stage 6 controlled prediction quality improvement sweep.

Tests delta-forecast target formulation, temporal window ablation,
rank extension, and AR-linear baseline against the Stage 4 winner.

Candidates:
  Group A — delta_forecast ablation (3 candidates)
  Group B — seq_length ablation (2 candidates)
  Group C — rank extension (1 candidate)
  Group D — AR-linear baseline (1 candidate)
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

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
logger = logging.getLogger("stage6_prediction_quality")

DATA_ROOT = PROJECT_ROOT / "data" / "navier_stokes"
OUTPUT_PATH = PROJECT_ROOT / "scripts" / "plots" / "models_eval" / "stage6_summary.json"
TUNING_DEV_SPLIT = 0.2
SELECTION_METRIC = "rollout_r2_common"

# Previous best from Stage 4 official test
PREVIOUS_BASELINE_TEST_R2 = 0.7443


@dataclass(frozen=True)
class Stage6Candidate:
    """Single candidate in the Stage 6 sweep."""

    name: str
    group: str
    ranks: list[int]
    delta_forecast: bool = False
    feature_mode: str = "latent_plus_delta"
    lstm_seq_length: int = 7
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_num_epochs: int = 150
    correction_hidden_size: int = 64
    correction_num_layers: int = 2
    correction_num_epochs: int = 120
    use_correction: bool = True
    use_ar_linear: bool = False
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
                },
                "correction_head": {
                    "hidden_size": self.correction_hidden_size,
                    "num_layers": self.correction_num_layers,
                    "num_epochs": self.correction_num_epochs,
                    "enabled": self.use_correction,
                },
                "use_ar_linear": self.use_ar_linear,
            }
        )
        return payload


def build_stage6_candidates(
    *,
    groups: tuple[str, ...] = (
        "delta_ablation",
        "seq_length_ablation",
        "rank_extension",
        "ar_baseline",
    ),
) -> list[Stage6Candidate]:
    """Build the full candidate grid for Stage 6."""

    requested = set(groups)
    candidates: list[Stage6Candidate] = []

    # ---- Group A: delta_forecast ablation ----
    if "delta_ablation" in requested:
        # A1: current best (control)
        candidates.append(
            Stage6Candidate(
                name="baseline_absolute",
                group="delta_ablation",
                ranks=list(DEFAULT_NAVIER_STOKES_RANKS),
                delta_forecast=False,
                feature_mode="latent_plus_delta",
                metadata={"is_stage6_control": True},
            )
        )
        # A2: delta prediction, vanilla features
        candidates.append(
            Stage6Candidate(
                name="delta_latent_only",
                group="delta_ablation",
                ranks=list(DEFAULT_NAVIER_STOKES_RANKS),
                delta_forecast=True,
                feature_mode="latent",
            )
        )
        # A3: delta prediction + velocity features
        candidates.append(
            Stage6Candidate(
                name="delta_plus_delta_features",
                group="delta_ablation",
                ranks=list(DEFAULT_NAVIER_STOKES_RANKS),
                delta_forecast=True,
                feature_mode="latent_plus_delta",
            )
        )

    # ---- Group B: seq_length ablation ----
    if "seq_length_ablation" in requested:
        candidates.append(
            Stage6Candidate(
                name="seq_length_10",
                group="seq_length_ablation",
                ranks=list(DEFAULT_NAVIER_STOKES_RANKS),
                delta_forecast=False,
                feature_mode="latent_plus_delta",
                lstm_seq_length=10,
            )
        )
        candidates.append(
            Stage6Candidate(
                name="seq_length_14",
                group="seq_length_ablation",
                ranks=list(DEFAULT_NAVIER_STOKES_RANKS),
                delta_forecast=False,
                feature_mode="latent_plus_delta",
                lstm_seq_length=14,
            )
        )

    # ---- Group C: rank extension ----
    if "rank_extension" in requested:
        candidates.append(
            Stage6Candidate(
                name="rank_r3_20",
                group="rank_extension",
                ranks=[64, 64, 20],
                delta_forecast=False,
                feature_mode="latent_plus_delta",
            )
        )

    # ---- Group D: AR-linear baseline ----
    if "ar_baseline" in requested:
        candidates.append(
            Stage6Candidate(
                name="ar_linear_baseline",
                group="ar_baseline",
                ranks=list(DEFAULT_NAVIER_STOKES_RANKS),
                delta_forecast=False,
                feature_mode="latent",
                use_correction=False,
                use_ar_linear=True,
            )
        )

    return candidates


def _build_ar_linear_forecaster(candidate: Stage6Candidate):
    """Build a linear autoregression-only forecaster (OLS baseline)."""
    return TrajectoryAwareLatentForecaster(
        config=LatentModalForecasterConfig(
            ranks=list(candidate.ranks),
            forecaster_type="linear",
            verbose=False,
            delta_forecast=False,
        ),
        feature_mode="latent",
    )


def _build_forecaster(candidate: Stage6Candidate):
    """Build the appropriate forecaster for a Stage 6 candidate."""
    if candidate.use_ar_linear:
        return _build_ar_linear_forecaster(candidate)

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
    candidate: Stage6Candidate,
    train_states: np.ndarray,
    test_states: np.ndarray,
) -> dict[str, Any]:
    logger.info("Running %s (group=%s)", candidate.name, candidate.group)
    start = time.time()

    model = _build_forecaster(candidate)
    model.fit(train_states)

    one_step = model.evaluate_one_step(test_states)
    rollout = model.evaluate_rollout(test_states)

    one_step_common = compute_common_horizon_metrics(
        one_step,
        test_states,
        DEFAULT_COMMON_WARMUP_STEPS,
    )
    rollout_common = compute_common_horizon_metrics(
        rollout,
        test_states,
        DEFAULT_COMMON_WARMUP_STEPS,
    )
    rollout_diagnostics = compute_common_horizon_diagnostics(
        rollout,
        test_states,
        DEFAULT_COMMON_WARMUP_STEPS,
    )
    elapsed = time.time() - start

    result = {
        "candidate": candidate.name,
        "group": candidate.group,
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
        "worst_trajectory_indices": rollout_diagnostics["worst_trajectory_indices"],
        "worst_trajectory_rmse": rollout_diagnostics["worst_trajectory_rmse"],
        "n_eval_steps_per_trajectory": rollout_common["n_eval_steps_per_trajectory"],
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


def build_payload(
    *,
    all_train_states,
    tuning_train_states,
    tuning_dev_states,
    official_test_states,
    candidates: list[Stage6Candidate],
    dev_results: list[dict[str, Any]],
    selected_dev_result: dict[str, Any],
    final_test_result: Optional[dict[str, Any]],
    skipped_official_test: bool,
) -> dict[str, Any]:
    return {
        "stage": "stage6_prediction_quality",
        "protocol": (
            "Select delta/seq_length/rank candidate on a train/dev trajectory split, "
            "then evaluate the selected candidate once on the official test split."
        ),
        "selection_metric": SELECTION_METRIC,
        "dev_split": TUNING_DEV_SPLIT,
        "skipped_official_test": skipped_official_test,
        "train_shape": list(all_train_states.shape),
        "tuning_train_shape": list(tuning_train_states.shape),
        "tuning_dev_shape": list(tuning_dev_states.shape),
        "official_test_shape": list(official_test_states.shape),
        "common_warmup_steps": DEFAULT_COMMON_WARMUP_STEPS,
        "previous_official_test_baseline": {
            "candidate": "rank_r3_15 (Stage 4 winner)",
            "ranks": list(DEFAULT_NAVIER_STOKES_RANKS),
            "rollout_r2_common": PREVIOUS_BASELINE_TEST_R2,
        },
        "candidate_grid": [
            {
                "candidate": c.name,
                "group": c.group,
                "metadata": c.as_metadata(),
            }
            for c in candidates
        ],
        "dev_results": dev_results,
        "selected_candidate": selected_dev_result["candidate"],
        "selected_dev_result": selected_dev_result,
        "final_test_result": final_test_result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["delta_ablation", "seq_length_ablation", "rank_extension", "ar_baseline"],
        choices=["delta_ablation", "seq_length_ablation", "rank_extension", "ar_baseline"],
        help="Candidate groups to run.",
    )
    parser.add_argument(
        "--n-train-trajectories",
        type=int,
        default=DEFAULT_N_TRAIN_TRAJECTORIES,
        help="Number of train trajectories to load.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Optional smoke-test limit.",
    )
    parser.add_argument(
        "--skip-official-test",
        action="store_true",
        help="Only run the train/dev sweep.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="JSON summary path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_train_states, official_test_states = load_data(args.n_train_trajectories)
    tuning_train_states, tuning_dev_states = split_train_dev_trajectories(
        all_train_states,
        dev_split=TUNING_DEV_SPLIT,
    )
    logger.info("Stage 6 tuning train: %s", tuning_train_states.shape)
    logger.info("Stage 6 tuning dev: %s", tuning_dev_states.shape)
    logger.info("Official test held out: %s", official_test_states.shape)

    candidates = build_stage6_candidates(groups=tuple(args.groups))
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]
    if not candidates:
        raise ValueError("No Stage 6 candidates selected")

    logger.info("Running %d candidates", len(candidates))
    dev_results = sort_results_for_selection(
        [
            evaluate_candidate(candidate, tuning_train_states, tuning_dev_states)
            for candidate in candidates
        ]
    )
    selected_dev_result = select_best_result(dev_results)
    selected_candidate = next(c for c in candidates if c.name == selected_dev_result["candidate"])
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
            all_train_states,
            official_test_states,
        )

    payload = build_payload(
        all_train_states=all_train_states,
        tuning_train_states=tuning_train_states,
        tuning_dev_states=tuning_dev_states,
        official_test_states=official_test_states,
        candidates=candidates,
        dev_results=dev_results,
        selected_dev_result=selected_dev_result,
        final_test_result=final_test_result,
        skipped_official_test=args.skip_official_test,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Stage 6 summary saved to: %s", args.output)

    # Print dev leaderboard
    logger.info("\n%s Dev Leaderboard %s", "=" * 25, "=" * 25)
    for r in dev_results:
        logger.info(
            "  %-30s  group=%-20s  rollout_r2=%.4f  1st_step_rmse=%.4f  gap=%.4f  delta=%s",
            r["candidate"],
            r["group"],
            r["rollout_r2_common"],
            r["first_step_rmse"],
            r["stability_gap_r2_common"],
            r["delta_forecast"],
        )


if __name__ == "__main__":
    main()
