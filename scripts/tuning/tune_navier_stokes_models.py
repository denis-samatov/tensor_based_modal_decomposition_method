#!/usr/bin/env python3
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any

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
    TrajectoryAwareResidualCorrectedForecaster,
    compute_common_horizon_diagnostics,
    compute_common_horizon_metrics,
    load_navier_stokes_trajectory_dataset,
    split_train_dev_trajectories as _split_train_dev_trajectories,
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
TUNING_DEV_SPLIT = 0.2


@dataclass(frozen=True)
class TuningCandidate:
    name: str
    family: str
    factory: Callable[[], Any]
    metadata: dict[str, Any] = field(default_factory=dict)


def load_data():
    dataset = load_navier_stokes_trajectory_dataset(DATA_ROOT)
    return dataset.train_states[:DEFAULT_N_TRAIN_TRAJECTORIES], dataset.test_states


def split_train_dev_trajectories(train_states, dev_split=TUNING_DEV_SPLIT):
    return _split_train_dev_trajectories(train_states, dev_split=dev_split)


def _latent_candidate(
    name,
    family,
    config,
    *,
    feature_mode="latent",
    metadata=None,
):
    payload = dict(metadata or {})
    payload.update(
        {
            "config": config.__dict__.copy(),
            "feature_mode": feature_mode,
        }
    )
    return TuningCandidate(
        name=name,
        family=family,
        factory=lambda: TrajectoryAwareLatentForecaster(
            config=config,
            feature_mode=feature_mode,
        ),
        metadata=payload,
    )


def _residual_candidate(
    name,
    *,
    spatial_loss_weight,
    rel_frob_loss_weight,
    metadata=None,
):
    config = LatentModalForecasterConfig(
        ranks=DEFAULT_NAVIER_STOKES_RANKS,
        forecaster_type="lstm",
        verbose=False,
        delta_forecast=False,
        lstm_hidden_size=128,
        lstm_num_layers=2,
        lstm_seq_length=7,
        lstm_num_epochs=150,
    )
    correction_loss = {
        "latent": 1.0,
        "spatial": spatial_loss_weight,
        "rel_frob": rel_frob_loss_weight,
    }
    payload = dict(metadata or {})
    payload.update(
        {
            "config": config.__dict__.copy(),
            "feature_mode": "latent_plus_delta",
            "correction_head": "mlp_h64_l2_e120",
            "correction_loss": correction_loss,
        }
    )
    return TuningCandidate(
        name=name,
        family="residual",
        factory=lambda: TrajectoryAwareResidualCorrectedForecaster(
            config=config,
            feature_mode="latent_plus_delta",
            correction_hidden_size=64,
            correction_num_layers=2,
            correction_dropout=0.0,
            correction_learning_rate=1e-3,
            correction_weight_decay=1e-5,
            correction_num_epochs=120,
            correction_batch_size=32,
            correction_val_split=0.2,
            correction_early_stopping_patience=20,
            correction_latent_loss_weight=1.0,
            correction_spatial_loss_weight=spatial_loss_weight,
            correction_rel_frob_loss_weight=rel_frob_loss_weight,
        ),
        metadata=payload,
    )


def build_candidates(groups=("mlp", "lstm", "residual")):
    requested = set(groups)
    candidates = [
        _latent_candidate(
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
        _latent_candidate(
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
        _latent_candidate(
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
        _latent_candidate(
            "lstm_h128_l2_s7_e150_plus_delta_features",
            "lstm",
            LatentModalForecasterConfig(
                ranks=DEFAULT_NAVIER_STOKES_RANKS,
                forecaster_type="lstm",
                verbose=False,
                delta_forecast=False,
                lstm_hidden_size=128,
                lstm_num_layers=2,
                lstm_seq_length=7,
                lstm_num_epochs=150,
            ),
            feature_mode="latent_plus_delta",
        ),
        _latent_candidate(
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
        _latent_candidate(
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
        _residual_candidate(
            "lstm_residual_latent_only",
            spatial_loss_weight=0.0,
            rel_frob_loss_weight=0.0,
        ),
        _residual_candidate(
            "lstm_residual_mixed_spatial_rel",
            spatial_loss_weight=0.1,
            rel_frob_loss_weight=0.05,
        ),
    ]
    return [candidate for candidate in candidates if candidate.family in requested]


def main():
    all_train_states, official_test_states = load_data()
    train_states, dev_states = split_train_dev_trajectories(all_train_states)
    logger.info("Tuning train trajectories: %s", train_states.shape)
    logger.info("Tuning dev trajectories: %s", dev_states.shape)
    logger.info("Official test trajectories held out from tuning: %s", official_test_states.shape)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for candidate in build_candidates():
        logger.info("Running %s", candidate.name)
        t0 = time.time()
        forecaster = candidate.factory()
        forecaster.fit(train_states)
        rollout = forecaster.evaluate_rollout(dev_states)
        metrics = compute_common_horizon_metrics(
            rollout,
            dev_states,
            DEFAULT_COMMON_WARMUP_STEPS,
        )
        diagnostics = compute_common_horizon_diagnostics(
            rollout,
            dev_states,
            DEFAULT_COMMON_WARMUP_STEPS,
        )
        elapsed = time.time() - t0
        result = {
            "candidate": candidate.name,
            "family": candidate.family,
            "rollout_r2_common": metrics["r2"],
            "rollout_rmse_common": metrics["rmse"],
            "rollout_mae_common": metrics["mae"],
            "rollout_rel_frob_common": metrics["rel_frob"],
            "stability_proxy_per_step_rmse": diagnostics["per_step_rmse"],
            "worst_trajectory_indices": diagnostics["worst_trajectory_indices"],
            "worst_trajectory_rmse": diagnostics["worst_trajectory_rmse"],
            "n_eval_steps_per_trajectory": metrics["n_eval_steps_per_trajectory"],
            "time": elapsed,
            "metadata": candidate.metadata,
        }
        results.append(result)
        logger.info(
            "%s -> rollout_r2_common=%.4f rmse=%.4f time=%.1fs",
            candidate.name,
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
        "dev_shape": list(dev_states.shape),
        "official_test_shape": list(official_test_states.shape),
        "selection_split": "train_trajectory_holdout",
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
