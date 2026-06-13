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

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from TBMD.experiments import (
    TrajectoryAwareDMDForecaster,
    TrajectoryAwareEigenvalueProjectedDMDForecaster,
    TrajectoryAwareStableDMDForecaster,
    compute_common_horizon_diagnostics,
    compute_common_horizon_metrics,
    load_navier_stokes_trajectory_dataset,
    split_train_dev_trajectories,
)
from TBMD.experiments.navier_stokes_model_registry import (
    DEFAULT_COMMON_WARMUP_STEPS,
    DEFAULT_DMD_RANK,
    DEFAULT_N_TRAIN_TRAJECTORIES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("stage3_stable_dmd")

DATA_ROOT = PROJECT_ROOT / "data" / "navier_stokes"
OUTPUT_PATH = PROJECT_ROOT / "scripts" / "plots" / "models_eval" / "stage3_stable_dmd_summary.json"
DEFAULT_RHOS = (1.0, 0.995, 0.98, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5, 0.4)
TUNING_DEV_SPLIT = 0.2
SELECTION_METRIC = "rollout_r2_common"


def _rho_slug(rho: float) -> str:
    return str(rho).replace(".", "_")


def build_stable_dmd_candidates(*, rank: int = DEFAULT_DMD_RANK, rhos=DEFAULT_RHOS):
    candidates = [
        {
            "name": "dmd_unconstrained",
            "max_spectral_radius": None,
            "stabilization": "none",
            "factory": lambda: TrajectoryAwareDMDForecaster(rank=rank),
        }
    ]
    for rho in rhos:
        candidates.append(
            {
                "name": f"stable_dmd_rho_{_rho_slug(float(rho))}",
                "max_spectral_radius": float(rho),
                "stabilization": "uniform_scale",
                "factory": lambda rho=float(rho): TrajectoryAwareStableDMDForecaster(
                    rank=rank,
                    max_spectral_radius=rho,
                ),
            }
        )
        candidates.append(
            {
                "name": f"projected_dmd_rho_{_rho_slug(float(rho))}",
                "max_spectral_radius": float(rho),
                "stabilization": "eigenvalue_projection",
                "factory": lambda rho=float(rho): TrajectoryAwareEigenvalueProjectedDMDForecaster(
                    rank=rank,
                    max_spectral_radius=rho,
                ),
            }
        )
    return candidates


def load_data():
    dataset = load_navier_stokes_trajectory_dataset(DATA_ROOT)
    return dataset.train_states[:DEFAULT_N_TRAIN_TRAJECTORIES], dataset.test_states


def sort_results_for_selection(results):
    return sorted(results, key=lambda item: item[SELECTION_METRIC], reverse=True)


def select_best_result(results):
    if not results:
        raise ValueError("Cannot select from an empty result list")
    return sort_results_for_selection(results)[0]


def evaluate_candidate(candidate, train_states, test_states):
    logger.info("Running %s", candidate["name"])
    start = time.time()
    model = candidate["factory"]()
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

    return {
        "candidate": candidate["name"],
        "max_spectral_radius": candidate["max_spectral_radius"],
        "stabilization": candidate["stabilization"],
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
        "unconstrained_spectral_radius": getattr(model, "_unconstrained_spectral_radius", None),
        "operator_scale": getattr(model, "_operator_scale", 1.0),
        "n_projected_modes": getattr(model, "_n_projected_modes", 0),
        "projection_imag_max": getattr(model, "_projection_imag_max", 0.0),
        "time": time.time() - start,
    }


def main():
    all_train_states, official_test_states = load_data()
    tuning_train_states, tuning_dev_states = split_train_dev_trajectories(
        all_train_states,
        dev_split=TUNING_DEV_SPLIT,
    )
    logger.info("Stable-DMD tuning train trajectories: %s", tuning_train_states.shape)
    logger.info("Stable-DMD tuning dev trajectories: %s", tuning_dev_states.shape)
    logger.info("Official test trajectories held out from rho selection: %s", official_test_states.shape)

    candidates = build_stable_dmd_candidates()
    dev_results = sort_results_for_selection(
        [
            evaluate_candidate(candidate, tuning_train_states, tuning_dev_states)
            for candidate in candidates
        ]
    )
    selected_dev_result = select_best_result(dev_results)
    selected_candidate = next(
        candidate for candidate in candidates if candidate["name"] == selected_dev_result["candidate"]
    )
    logger.info(
        "Selected %s by dev %s=%.6f",
        selected_dev_result["candidate"],
        SELECTION_METRIC,
        selected_dev_result[SELECTION_METRIC],
    )
    final_test_result = evaluate_candidate(
        selected_candidate,
        all_train_states,
        official_test_states,
    )
    payload = {
        "protocol": "Select max_spectral_radius on a train/dev trajectory split, then evaluate the selected candidate once on the official test split.",
        "selection_metric": SELECTION_METRIC,
        "dev_split": TUNING_DEV_SPLIT,
        "train_shape": list(all_train_states.shape),
        "tuning_train_shape": list(tuning_train_states.shape),
        "tuning_dev_shape": list(tuning_dev_states.shape),
        "official_test_shape": list(official_test_states.shape),
        "common_warmup_steps": DEFAULT_COMMON_WARMUP_STEPS,
        "rank": DEFAULT_DMD_RANK,
        "dev_results": dev_results,
        "selected_candidate": selected_dev_result["candidate"],
        "selected_dev_result": selected_dev_result,
        "final_test_result": final_test_result,
        "results": dev_results,
        "best_candidate": selected_dev_result,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Stable DMD summary saved to: %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
