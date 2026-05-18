#!/usr/bin/env python3
"""Train/dev accuracy sweep for Fast TBMD+QR+CS t+1 predictors."""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MPL_CACHE_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from TBMD.experiments import load_navier_stokes_trajectory_dataset, split_train_dev_trajectories
from TBMD.experiments.navier_stokes_fast_tplus1 import (
    FastWindowedTBMDQRCSConfig,
    FastWindowedTBMDQRCSForecaster,
)
from TBMD.experiments.navier_stokes_model_registry import DEFAULT_N_TRAIN_TRAJECTORIES

DATA_ROOT = PROJECT_ROOT / "data" / "navier_stokes"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "plots"
    / "models_eval"
    / "stage5_fast_tplus1_accuracy_sweep.json"
)
DEV_SPLIT = 0.2


@dataclass(frozen=True)
class FastTPlus1Candidate:
    name: str
    groups: tuple[str, ...]
    history_length: int
    r_tau: int
    r_x: int
    r_y: int
    r_segment: int
    n_spatial_sensors: int
    notes: dict[str, object]
    correction_alpha: float = 1e-8
    correction_scale: float = 1.0
    correction_residual_rank: int | None = None

    def config(self, *, max_train_segments: int | None, random_state: int) -> FastWindowedTBMDQRCSConfig:
        return FastWindowedTBMDQRCSConfig(
            history_length=self.history_length,
            ranks=[self.r_tau, self.r_x, self.r_y, self.r_segment],
            n_spatial_sensors=self.n_spatial_sensors,
            max_train_segments=max_train_segments,
            correction_alpha=self.correction_alpha,
            correction_scale=self.correction_scale,
            correction_residual_rank=self.correction_residual_rank,
            sensor_rcond=1e-6,
            random_state=random_state,
        )


def build_candidates(groups: tuple[str, ...] = ("quick",)) -> list[FastTPlus1Candidate]:
    """Build focused train/dev candidates for the next t+1 accuracy stage."""
    include = set(groups)
    candidates = [
        FastTPlus1Candidate(
            name="baseline_h7_rt8_r300_s300",
            groups=("quick", "baseline"),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=300,
            notes={"purpose": "current practical reference"},
        ),
        FastTPlus1Candidate(
            name="history10_rt10_r300_s300",
            groups=("quick", "history_length"),
            history_length=10,
            r_tau=10,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=300,
            notes={"hypothesis": "longer temporal context improves t+1"},
        ),
        FastTPlus1Candidate(
            name="history12_rt12_r300_s300",
            groups=("history_length",),
            history_length=12,
            r_tau=12,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=300,
            notes={"hypothesis": "even longer temporal context"},
        ),
        FastTPlus1Candidate(
            name="spatial40_r300_s300",
            groups=("quick", "spatial_rank"),
            history_length=7,
            r_tau=8,
            r_x=40,
            r_y=40,
            r_segment=300,
            n_spatial_sensors=300,
            notes={"hypothesis": "higher spatial rank improves reconstruction"},
        ),
        FastTPlus1Candidate(
            name="rsegment350_s350",
            groups=("quick", "segment_rank", "sensor_budget"),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=350,
            n_spatial_sensors=350,
            notes={"hypothesis": "higher latent rank with matched sensors"},
        ),
        FastTPlus1Candidate(
            name="rsegment400_s400",
            groups=("segment_rank", "sensor_budget"),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=400,
            n_spatial_sensors=400,
            notes={"hypothesis": "larger latent rank may cross 0.85"},
        ),
        FastTPlus1Candidate(
            name="quality_s600_reference",
            groups=("sensor_budget", "quality_reference", "s600_refine"),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"purpose": "current quality-max reference"},
        ),
        FastTPlus1Candidate(
            name="spatial40_r300_s600",
            groups=("s600_refine", "spatial_rank", "sensor_budget"),
            history_length=7,
            r_tau=8,
            r_x=40,
            r_y=40,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "higher spatial rank with quality sensor budget"},
        ),
        FastTPlus1Candidate(
            name="history10_rt10_r300_s600",
            groups=("s600_refine", "history_length", "sensor_budget"),
            history_length=10,
            r_tau=10,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "longer temporal context with quality sensor budget"},
        ),
        FastTPlus1Candidate(
            name="rsegment350_s600",
            groups=("s600_refine", "segment_rank", "sensor_budget"),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=350,
            n_spatial_sensors=600,
            notes={"hypothesis": "higher latent rank while keeping overdetermined sensors"},
        ),
        FastTPlus1Candidate(
            name="residual_svd32_r300_s600",
            groups=("residual_head",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "low-rank residual basis regularizes correction head"},
            correction_residual_rank=32,
        ),
        FastTPlus1Candidate(
            name="residual_svd64_r300_s600",
            groups=("residual_head",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "medium low-rank residual basis"},
            correction_residual_rank=64,
        ),
        FastTPlus1Candidate(
            name="residual_svd128_r300_s600",
            groups=("residual_head", "residual_head_fine"),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "larger residual basis retains more fine-scale error"},
            correction_residual_rank=128,
        ),
        FastTPlus1Candidate(
            name="residual_svd160_r300_s600",
            groups=("residual_head_fine",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "fine residual basis just above rank 128"},
            correction_residual_rank=160,
        ),
        FastTPlus1Candidate(
            name="residual_svd192_r300_s600",
            groups=("residual_head_fine",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "fine residual basis around mid-rank"},
            correction_residual_rank=192,
        ),
        FastTPlus1Candidate(
            name="residual_svd256_r300_s600",
            groups=("residual_head_fine",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "high residual rank approaches full residual head"},
            correction_residual_rank=256,
        ),
        FastTPlus1Candidate(
            name="residual_svd64_alpha1e-6_r300_s600",
            groups=("residual_head",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "stronger ridge regularization for residual SVD head"},
            correction_alpha=1e-6,
            correction_residual_rank=64,
        ),
    ]
    if "all" in include:
        return candidates
    return [candidate for candidate in candidates if include.intersection(candidate.groups)]


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _compact_eval(eval_result: dict[str, object]) -> dict[str, object]:
    target = np.asarray(eval_result["target_spatial"], dtype=np.float64)
    pred = np.asarray(eval_result["pred_spatial"], dtype=np.float64)
    return {
        "spatial_r2": float(eval_result["spatial_r2"]),
        "spatial_rmse": float(eval_result["spatial_rmse"]),
        "spatial_mae": float(np.mean(np.abs(target - pred))),
        "spatial_rel_frob_err": float(eval_result["spatial_rel_frob_err"]),
        "spatial_mse": float(eval_result["spatial_mse"]),
        "n_eval_samples": int(eval_result["n_eval_samples"]),
    }


def select_best_result(results: list[dict[str, object]]) -> dict[str, object]:
    if not results:
        raise ValueError("results must not be empty")
    return max(results, key=lambda item: item["dev_spatial_r2"])


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train-trajectories", type=int, default=640)
    parser.add_argument("--n-dev-trajectories", type=int, default=None)
    parser.add_argument("--dev-split", type=float, default=DEV_SPLIT)
    parser.add_argument("--groups", nargs="*", default=["quick"])
    parser.add_argument("--max-train-segments", type=int, default=2048)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_train_trajectories > DEFAULT_N_TRAIN_TRAJECTORIES:
        raise ValueError(
            f"n_train_trajectories cannot exceed {DEFAULT_N_TRAIN_TRAJECTORIES}"
        )
    dataset = load_navier_stokes_trajectory_dataset(DATA_ROOT)
    all_train_states = dataset.train_states[: args.n_train_trajectories]
    tuning_train_states, tuning_dev_states = split_train_dev_trajectories(
        all_train_states,
        dev_split=args.dev_split,
    )
    if args.n_dev_trajectories is not None:
        tuning_dev_states = tuning_dev_states[: args.n_dev_trajectories]

    candidates = build_candidates(tuple(args.groups))
    results = []
    details = []
    for candidate in candidates:
        model = FastWindowedTBMDQRCSForecaster(
            candidate.config(
                max_train_segments=args.max_train_segments,
                random_state=args.random_state,
            )
        )
        start = time.perf_counter()
        model.fit(tuning_train_states)
        fit_time = time.perf_counter() - start
        eval_start = time.perf_counter()
        eval_result = model.evaluate_one_step(tuning_dev_states)
        eval_time = time.perf_counter() - eval_start
        compact = _compact_eval(eval_result)
        row = {
            "candidate": candidate.name,
            "groups": ",".join(candidate.groups),
            "history_length": candidate.history_length,
            "r_tau": candidate.r_tau,
            "r_x": candidate.r_x,
            "r_y": candidate.r_y,
            "r_segment": candidate.r_segment,
            "n_spatial_sensors": candidate.n_spatial_sensors,
            "correction_alpha": candidate.correction_alpha,
            "correction_scale": candidate.correction_scale,
            "correction_residual_rank": candidate.correction_residual_rank,
            "dev_spatial_r2": compact["spatial_r2"],
            "dev_spatial_rmse": compact["spatial_rmse"],
            "dev_spatial_mae": compact["spatial_mae"],
            "dev_spatial_rel_frob_err": compact["spatial_rel_frob_err"],
            "fit_time_seconds": float(fit_time),
            "dev_eval_time_seconds": float(eval_time),
            "n_dev_eval_samples": compact["n_eval_samples"],
        }
        results.append(row)
        details.append(
            {
                "candidate": candidate.__dict__,
                "config": model.get_config(),
                "fit_metrics": model.get_metrics().get("fit", {}),
                "dev_metrics": compact,
            }
        )

    selected = select_best_result(results)
    payload = {
        "stage": "stage5_fast_tplus1_accuracy_sweep",
        "protocol": (
            "Train/dev sweep only. The official test split is not loaded or evaluated. "
            "Use this output to choose a future final-refit candidate."
        ),
        "groups": list(args.groups),
        "train_shape": list(all_train_states.shape),
        "tuning_train_shape": list(tuning_train_states.shape),
        "tuning_dev_shape": list(tuning_dev_states.shape),
        "max_train_segments": args.max_train_segments,
        "random_state": args.random_state,
        "results": results,
        "selected_by_dev": selected,
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    _write_csv(args.output.with_suffix(".csv"), results)
    print(f"Saved fast t+1 accuracy sweep to {args.output}")


if __name__ == "__main__":
    main()
