#!/usr/bin/env python3
"""Stage 4 controlled Navier-Stokes rank and architecture sweep."""

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

from TBMD.config import LatentModalForecasterConfig
from TBMD.experiments import (
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
logger = logging.getLogger("stage4_rank_sweep")

DATA_ROOT = PROJECT_ROOT / "data" / "navier_stokes"
OUTPUT_PATH = PROJECT_ROOT / "scripts" / "plots" / "models_eval" / "stage4_summary.json"
TUNING_DEV_SPLIT = 0.2
SELECTION_METRIC = "rollout_r2_common"
PREVIOUS_BASELINE_TEST_R2 = 0.5477365147680742

RANK_SWEEP_R3 = (3, 5, 8, 10, 15)
CORRECTION_HEAD_SWEEP = (
    ("corr_h64_l2_e120", 64, 2, 120),
    ("corr_h128_l2_e120", 128, 2, 120),
    ("corr_h64_l3_e150", 64, 3, 150),
    ("corr_h128_l2_e200", 128, 2, 200),
)
LSTM_BACKBONE_SWEEP = (
    ("lstm_h128_l2", 128, 2),
    ("lstm_h256_l2", 256, 2),
    ("lstm_h128_l3", 128, 3),
)
SPATIAL_RANK_SWEEP = (
    ("ranks_32_32_5", [32, 32, 5]),
    ("ranks_48_48_5", [48, 48, 5]),
)

BASELINE_RANKS = [64, 64, 5]
BASELINE_CORRECTION_LABEL = "corr_h64_l2_e120"
BASELINE_LSTM_LABEL = "lstm_h128_l2"


@dataclass(frozen=True)
class Stage4Candidate:
    name: str
    groups: tuple[str, ...]
    ranks: list[int]
    correction_label: str = BASELINE_CORRECTION_LABEL
    correction_hidden_size: int = 64
    correction_num_layers: int = 2
    correction_num_epochs: int = 120
    lstm_label: str = BASELINE_LSTM_LABEL
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_seq_length: int = 7
    lstm_num_epochs: int = 150
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def r3(self) -> int:
        return int(self.ranks[2])

    def config(self) -> LatentModalForecasterConfig:
        return LatentModalForecasterConfig(
            ranks=list(self.ranks),
            forecaster_type="lstm",
            verbose=False,
            delta_forecast=False,
            lstm_hidden_size=self.lstm_hidden_size,
            lstm_num_layers=self.lstm_num_layers,
            lstm_seq_length=self.lstm_seq_length,
            lstm_num_epochs=self.lstm_num_epochs,
        )

    def factory(self) -> TrajectoryAwareResidualCorrectedForecaster:
        return TrajectoryAwareResidualCorrectedForecaster(
            config=self.config(),
            feature_mode="latent_plus_delta",
            correction_hidden_size=self.correction_hidden_size,
            correction_num_layers=self.correction_num_layers,
            correction_dropout=0.0,
            correction_learning_rate=1e-3,
            correction_weight_decay=1e-5,
            correction_num_epochs=self.correction_num_epochs,
            correction_batch_size=32,
            correction_val_split=0.2,
            correction_early_stopping_patience=20,
            correction_latent_loss_weight=1.0,
            correction_spatial_loss_weight=0.0,
            correction_rel_frob_loss_weight=0.0,
        )

    def as_metadata(self) -> dict[str, Any]:
        payload = dict(self.metadata)
        payload.update(
            {
                "ranks": list(self.ranks),
                "r3": self.r3,
                "feature_mode": "latent_plus_delta",
                "delta_forecast": False,
                "lstm": {
                    "label": self.lstm_label,
                    "hidden_size": self.lstm_hidden_size,
                    "num_layers": self.lstm_num_layers,
                    "seq_length": self.lstm_seq_length,
                    "num_epochs": self.lstm_num_epochs,
                },
                "correction_head": {
                    "label": self.correction_label,
                    "hidden_size": self.correction_hidden_size,
                    "num_layers": self.correction_num_layers,
                    "num_epochs": self.correction_num_epochs,
                    "loss": {
                        "latent": 1.0,
                        "spatial": 0.0,
                        "rel_frob": 0.0,
                    },
                },
            }
        )
        return payload


def _baseline_candidate() -> Stage4Candidate:
    return Stage4Candidate(
        name="baseline_r3_5_corr_h64_l2_e120_lstm_h128_l2",
        groups=("rank_sweep", "correction_head_sweep", "lstm_backbone_sweep"),
        ranks=list(BASELINE_RANKS),
        metadata={
            "aliases": ["rank_r3_5", BASELINE_CORRECTION_LABEL, BASELINE_LSTM_LABEL],
            "is_stage4_baseline": True,
        },
    )


def build_stage4_candidates(
    *,
    groups: tuple[str, ...] = ("rank_sweep", "correction_head_sweep", "lstm_backbone_sweep"),
    include_spatial: bool = False,
) -> list[Stage4Candidate]:
    requested = set(groups)
    if include_spatial:
        requested.add("spatial_rank_sweep")

    candidates = [_baseline_candidate()]

    for r3 in RANK_SWEEP_R3:
        if r3 == BASELINE_RANKS[2]:
            continue
        candidates.append(
            Stage4Candidate(
                name=f"rank_r3_{r3}",
                groups=("rank_sweep",),
                ranks=[64, 64, int(r3)],
            )
        )

    for label, hidden_size, num_layers, num_epochs in CORRECTION_HEAD_SWEEP:
        if label == BASELINE_CORRECTION_LABEL:
            continue
        candidates.append(
            Stage4Candidate(
                name=label,
                groups=("correction_head_sweep",),
                ranks=list(BASELINE_RANKS),
                correction_label=label,
                correction_hidden_size=int(hidden_size),
                correction_num_layers=int(num_layers),
                correction_num_epochs=int(num_epochs),
            )
        )

    for label, hidden_size, num_layers in LSTM_BACKBONE_SWEEP:
        if label == BASELINE_LSTM_LABEL:
            continue
        candidates.append(
            Stage4Candidate(
                name=label,
                groups=("lstm_backbone_sweep",),
                ranks=list(BASELINE_RANKS),
                lstm_label=label,
                lstm_hidden_size=int(hidden_size),
                lstm_num_layers=int(num_layers),
            )
        )

    for label, ranks in SPATIAL_RANK_SWEEP:
        candidates.append(
            Stage4Candidate(
                name=label,
                groups=("spatial_rank_sweep",),
                ranks=list(ranks),
            )
        )

    return [
        candidate
        for candidate in candidates
        if requested.intersection(candidate.groups)
    ]


def load_data(n_train_trajectories: int = DEFAULT_N_TRAIN_TRAJECTORIES):
    dataset = load_navier_stokes_trajectory_dataset(DATA_ROOT)
    return dataset.train_states[:n_train_trajectories], dataset.test_states


def sort_results_for_selection(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(results, key=lambda item: item[SELECTION_METRIC], reverse=True)


def select_best_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("Cannot select from an empty result list")
    return sort_results_for_selection(results)[0]


def best_results_by_group(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_group: dict[str, dict[str, Any]] = {}
    for result in sort_results_for_selection(results):
        for group in result["groups"]:
            by_group.setdefault(group, result)
    return by_group


def evaluate_candidate(
    candidate: Stage4Candidate,
    train_states,
    test_states,
) -> dict[str, Any]:
    logger.info("Running %s", candidate.name)
    start = time.time()
    model = candidate.factory()
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
        "groups": list(candidate.groups),
        "ranks": list(candidate.ranks),
        "r3": candidate.r3,
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
        "time": elapsed,
        "metadata": candidate.as_metadata(),
    }
    logger.info(
        "%s -> dev/test rollout_r2_common=%.4f rmse=%.4f time=%.1fs",
        candidate.name,
        result["rollout_r2_common"],
        result["rollout_rmse_common"],
        elapsed,
    )
    return result


def build_payload(
    *,
    all_train_states,
    tuning_train_states,
    tuning_dev_states,
    official_test_states,
    candidates: list[Stage4Candidate],
    dev_results: list[dict[str, Any]],
    selected_dev_result: dict[str, Any],
    final_test_result: dict[str, Any] | None,
    include_spatial: bool,
    skipped_official_test: bool,
) -> dict[str, Any]:
    selected_ranks = selected_dev_result["ranks"]
    return {
        "protocol": (
            "Select rank/capacity candidate on a train/dev trajectory split, then "
            "evaluate the selected candidate once on the official test split."
        ),
        "architectural_clarification": {
            "forecasting_pipeline": "SVD latent coefficients -> LSTM -> residual head -> SVD reconstruction",
            "qr_used_in_forecasting": False,
            "cs_used_in_forecasting": False,
        },
        "selection_metric": SELECTION_METRIC,
        "dev_split": TUNING_DEV_SPLIT,
        "include_spatial_rank_sweep": include_spatial,
        "skipped_official_test": skipped_official_test,
        "train_shape": list(all_train_states.shape),
        "tuning_train_shape": list(tuning_train_states.shape),
        "tuning_dev_shape": list(tuning_dev_states.shape),
        "official_test_shape": list(official_test_states.shape),
        "common_warmup_steps": DEFAULT_COMMON_WARMUP_STEPS,
        "previous_official_test_baseline": {
            "candidate": "LSTM + T+1 Residual Corrector",
            "ranks": list(BASELINE_RANKS),
            "rollout_r2_common": PREVIOUS_BASELINE_TEST_R2,
        },
        "candidate_grid": [
            {
                "candidate": candidate.name,
                "groups": list(candidate.groups),
                "metadata": candidate.as_metadata(),
            }
            for candidate in candidates
        ],
        "dev_results": dev_results,
        "results": dev_results,
        "best_by_group": best_results_by_group(dev_results),
        "selected_candidate": selected_dev_result["candidate"],
        "selected_dev_result": selected_dev_result,
        "final_test_result": final_test_result,
        "best_candidate": selected_dev_result,
        "registry_update": {
            "current_default_ranks": list(DEFAULT_NAVIER_STOKES_RANKS),
            "dev_selected_ranks": list(selected_ranks),
            "update_default_ranks_if_final_test_confirms": list(selected_ranks),
            "default_ranks_change_needed": list(selected_ranks) != list(DEFAULT_NAVIER_STOKES_RANKS),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-spatial",
        action="store_true",
        help="Include optional [32,32,5] and [48,48,5] spatial compression candidates.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["rank_sweep", "correction_head_sweep", "lstm_backbone_sweep"],
        choices=["rank_sweep", "correction_head_sweep", "lstm_backbone_sweep", "spatial_rank_sweep"],
        help="Candidate groups to run. Defaults to the three primary Stage 4 sweeps.",
    )
    parser.add_argument(
        "--n-train-trajectories",
        type=int,
        default=DEFAULT_N_TRAIN_TRAJECTORIES,
        help="Number of train trajectories to load before the train/dev split.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Optional smoke-test limit applied after candidate construction.",
    )
    parser.add_argument(
        "--skip-official-test",
        action="store_true",
        help="Only run the train/dev sweep; useful for smoke checks.",
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
    logger.info("Stage 4 tuning train trajectories: %s", tuning_train_states.shape)
    logger.info("Stage 4 tuning dev trajectories: %s", tuning_dev_states.shape)
    logger.info("Official test trajectories held out from selection: %s", official_test_states.shape)

    candidates = build_stage4_candidates(
        groups=tuple(args.groups),
        include_spatial=args.include_spatial or "spatial_rank_sweep" in args.groups,
    )
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]
    if not candidates:
        raise ValueError("No Stage 4 candidates selected")

    dev_results = sort_results_for_selection(
        [
            evaluate_candidate(candidate, tuning_train_states, tuning_dev_states)
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
        include_spatial=args.include_spatial or "spatial_rank_sweep" in args.groups,
        skipped_official_test=args.skip_official_test,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Stage 4 summary saved to: %s", args.output)


if __name__ == "__main__":
    main()
