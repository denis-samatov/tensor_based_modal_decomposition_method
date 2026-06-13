#!/usr/bin/env python3
"""Final refit for selected Fast TBMD+QR+CS t+1 Navier-Stokes predictors."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MPL_CACHE_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from TBMD.experiments import load_navier_stokes_trajectory_dataset
from TBMD.experiments.navier_stokes_model_registry import (
    DEFAULT_N_TRAIN_TRAJECTORIES,
    get_fast_tplus1_model_specs,
)

DATA_ROOT = PROJECT_ROOT / "data" / "navier_stokes"
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "plots" / "models_eval" / "stage5_fast_tplus1_final"


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


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_report(path: Path, payload: dict[str, object]) -> None:
    rows = payload["results"]
    lines = [
        "# Stage 5 Final Fast TBMD+QR+CS T+1 Refit",
        "",
        "Protocol: selected configs are refit on the requested train trajectories with no dev selection. "
        "The official test split is evaluated once per selected preset.",
        "",
        "| Preset | Label | Rank | Sensors | Test R² | RMSE | MAE | Rel. Frob. | Inference s/sample | Model Size MB |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {slug} | {label} | {rank} | {sensors} | {r2:.4f} | {rmse:.4f} | "
            "{mae:.4f} | {rel:.4f} | {sec:.6f} | {size:.2f} |".format(
                slug=row["slug"],
                label=row["label"],
                rank=row["rank"],
                sensors=row["sensors"],
                r2=row["test_spatial_r2"],
                rmse=row["test_spatial_rmse"],
                mae=row["test_spatial_mae"],
                rel=row["test_spatial_rel_frob_err"],
                sec=row["inference_seconds_per_sample"],
                size=row["model_size_mb"],
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: this stage is a one-step-ahead sparse-sensing predictor, not a rollout forecaster. "
            "Unlike heavy neural-operator approaches, the proposed Fast TBMD+QR+CS pipeline constructs a compact "
            "TBMD hidden state from sparse sensing and learns only a lightweight correction head for one-step-ahead prediction.",
            "",
            "FNO/PINN comparison status: local FNO/PINN baselines are not present in this run and should not be "
            "treated as directly comparable until trained and evaluated on the same split and metrics.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slugs",
        nargs="*",
        default=["fast_tplus1_r300_s300", "fast_tplus1_r300_s600"],
        help="Registry slugs to refit. Defaults to practical and quality-max presets.",
    )
    parser.add_argument("--n-train-trajectories", type=int, default=DEFAULT_N_TRAIN_TRAJECTORIES)
    parser.add_argument("--n-test-trajectories", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_train_trajectories > DEFAULT_N_TRAIN_TRAJECTORIES:
        raise ValueError(
            f"n_train_trajectories cannot exceed {DEFAULT_N_TRAIN_TRAJECTORIES}"
        )

    specs_by_slug = {spec.slug: spec for spec in get_fast_tplus1_model_specs()}
    missing = [slug for slug in args.slugs if slug not in specs_by_slug]
    if missing:
        raise ValueError(f"Unknown fast t+1 preset slug(s): {missing}")

    dataset = load_navier_stokes_trajectory_dataset(DATA_ROOT)
    train_states = dataset.train_states[: args.n_train_trajectories]
    test_states = dataset.test_states[: args.n_test_trajectories]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    detailed = []
    for slug in args.slugs:
        spec = specs_by_slug[slug]
        model = spec.factory()
        model.config.random_state = args.random_state

        fit_start = time.perf_counter()
        model.fit(train_states)
        fit_time = time.perf_counter() - fit_start

        eval_start = time.perf_counter()
        eval_result = model.evaluate_one_step(test_states)
        eval_time = time.perf_counter() - eval_start
        compact_eval = _compact_eval(eval_result)

        model_path = args.output_dir / f"{slug}_predictor.npz"
        model.save(model_path)
        model_size_mb = model_path.stat().st_size / (1024 * 1024)
        inference_seconds_per_sample = eval_time / max(compact_eval["n_eval_samples"], 1)

        row = {
            "slug": slug,
            "name": spec.name,
            "label": spec.notes["label"],
            "purpose": spec.notes["purpose"],
            "rank": spec.notes["rank"],
            "sensors": spec.notes["sensors"],
            "history_length": spec.notes["history_length"],
            "history_measurements": spec.notes["history_measurements"],
            "train_trajectories": int(train_states.shape[0]),
            "test_trajectories": int(test_states.shape[0]),
            "test_eval_samples": compact_eval["n_eval_samples"],
            "test_spatial_r2": compact_eval["spatial_r2"],
            "test_spatial_rmse": compact_eval["spatial_rmse"],
            "test_spatial_mae": compact_eval["spatial_mae"],
            "test_spatial_rel_frob_err": compact_eval["spatial_rel_frob_err"],
            "fit_time_seconds": float(fit_time),
            "test_eval_time_seconds": float(eval_time),
            "inference_seconds_per_sample": float(inference_seconds_per_sample),
            "samples_per_second": float(1.0 / max(inference_seconds_per_sample, 1e-12)),
            "model_size_mb": float(model_size_mb),
            "model_path": str(model_path),
        }
        results.append(row)
        detailed.append(
            {
                "row": row,
                "config": model.get_config(),
                "fit_metrics": model.get_metrics().get("fit", {}),
                "official_test_metrics": compact_eval,
            }
        )

    payload = {
        "stage": "stage5_fast_tplus1_final_refit",
        "protocol": (
            "Final refit for preselected fast t+1 TBMD+QR+CS configs. "
            "No dev split or official-test tuning is performed in this script."
        ),
        "selection_source": (
            "Configs are supplied by CLI slugs and must be selected before this final run "
            "from train/dev artifacts. This script does not perform dev selection."
        ),
        "train_shape": list(train_states.shape),
        "official_test_shape": list(test_states.shape),
        "random_state": int(args.random_state),
        "slugs": list(args.slugs),
        "results": results,
        "details": detailed,
        "risk_register": [
            "Official test is evaluated once per preselected preset; do not use this output for further tuning.",
            "Saved .npz predictors contain large dense dictionary/head arrays and may be tens or hundreds of MB.",
            "Quality-max uses more sensors and is slower at inference than the practical preset.",
            "FNO/PINN comparisons require local same-split baselines before making performance claims.",
        ],
        "accuracy_roadmap": {
            "history_length": [5, 7, 10, 12],
            "r_tau": [6, 8, 10, 12],
            "r_segment": [250, 300, 350, 400],
            "r_x_r_y": [32, 40, 48],
            "sensor_placement": [
                "current QR over history dictionary",
                "QR over history+target modes",
                "QR over residual/error-sensitive basis",
                "QR plus leverage oversampling",
            ],
            "correction_heads": [
                "current coefficient ridge",
                "small MLP on coefficients only",
                "low-rank spatial residual head",
                "avoid full coeff-to-4096 head unless metrics and speed justify it",
            ],
        },
    }

    json_path = args.output_dir / "stage5_fast_tplus1_final_summary.json"
    csv_path = args.output_dir / "stage5_fast_tplus1_final_metrics.csv"
    md_path = args.output_dir / "stage5_fast_tplus1_final_report.md"
    json_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    _write_csv(csv_path, results)
    _write_markdown_report(md_path, payload)
    print(f"Saved final Stage 5 fast t+1 summary to {json_path}")


if __name__ == "__main__":
    main()
