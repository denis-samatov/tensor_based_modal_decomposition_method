#!/usr/bin/env python3
"""Cached multi-dev residual sweep for Fast TBMD+QR+CS t+1 forecasting.

This script keeps the expensive TBMD dictionary, QR sensors, and sensor decoder
fixed per train-only dev split. It then sweeps residual correction heads from
cached base predictions and coefficient features. The official test split is
not loaded or evaluated.
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.utils.extmath import randomized_svd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MPL_CACHE_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from TBMD.experiments import load_navier_stokes_trajectory_dataset
from TBMD.experiments.navier_stokes_fast_tplus1 import (
    apply_coefficient_calibrator,
    apply_ridge_residual_corrector,
    attach_coefficient_gate,
    build_correction_feature_matrix,
    build_forecast_segment_tensor_with_refs,
    fft_highpass_frames,
    fit_coefficient_calibrator,
    fit_composite_patch_hf_residual_corrector,
    fit_noop_residual_corrector,
    fit_patch_residual_svd_corrector,
    fit_ridge_residual_corrector,
    fit_segment_dictionary,
    fit_sensor_coefficient_decoder,
    fit_sensor_innovation_encoder,
    history_and_target,
    history_sensor_matrix,
    place_fixed_spatial_sensors,
    predict_next_sensor_decoder_with_measurements,
    reconstruct_target_from_coefficients,
    residual_target_frames,
    smooth_coefficients_by_segment_refs,
    target_frames_from_segments,
)
from TBMD.experiments.navier_stokes_forecasting import _compute_regression_metrics
from TBMD.experiments.navier_stokes_model_registry import DEFAULT_N_TRAIN_TRAJECTORIES
from TBMD.experiments.navier_stokes_structure_metrics import compute_structure_metrics

DATA_ROOT = PROJECT_ROOT / "data" / "navier_stokes"
OUTPUT_PATH = (
    PROJECT_ROOT / "scripts" / "plots" / "models_eval" / "stage5_fast_tplus1_cached_residual.json"
)


@dataclass(frozen=True)
class CachedBaseConfig:
    history_length: int
    ranks: tuple[int, int, int, int]
    n_spatial_sensors: int
    max_train_segments: int | None
    sensor_decoder: str = "ridge"
    decoder_ridge_lambda: float = 1e-8
    sensor_rcond: float = 1e-6
    random_state: int = 0
    dtype: str = "float32"


@dataclass(frozen=True)
class ResidualCandidate:
    name: str
    residual_rank: int | None
    scale: float
    alpha: float = 1e-8
    residual_target: str = "field"
    highpass_cutoff_fraction: float = 0.35
    residual_weighting: str = "uniform"
    residual_weight_floor: float = 0.1
    hf_scale: float = 0.0
    sample_weighting: str = "uniform"
    sample_weight_power: float = 1.0
    sample_weight_floor: float = 0.25
    sample_weight_clip: float = 4.0
    head_type: str = "global_residual_svd"
    patch_size: int | None = None
    patch_residual_rank: int | None = None
    gate_type: str = "none"
    gate_threshold: float = 1.25
    gate_strength: float = 1.0
    gate_min: float = 0.5
    innovation_rank: int = 0
    innovation_include_norms: bool = False
    coefficient_calibration_type: str = "none"
    coefficient_calibration_alpha: float = 1e-6
    coefficient_calibration_blend: float = 1.0
    coefficient_calibration_innovation_rank: int = 0
    coefficient_calibration_include_norms: bool = False
    coefficient_temporal_smoothing_alpha: float = 0.0
    coefficient_temporal_reset_on_gap: bool = True


def build_base_config(mode: str, args: argparse.Namespace) -> CachedBaseConfig:
    if mode == "smoke":
        ranks = (6, 16, 16, 48)
        sensors = 96
        max_segments = 96
    else:
        ranks = (8, 32, 32, 300)
        sensors = 1000
        max_segments = 2048
    return CachedBaseConfig(
        history_length=args.history_length,
        ranks=tuple(args.ranks) if args.ranks else ranks,
        n_spatial_sensors=args.n_spatial_sensors or sensors,
        max_train_segments=args.max_train_segments
        if args.max_train_segments is not None
        else max_segments,
        sensor_decoder=args.sensor_decoder,
        decoder_ridge_lambda=args.decoder_ridge_lambda,
        sensor_rcond=args.sensor_rcond,
        random_state=args.random_state,
    )


def build_residual_candidates(mode: str) -> list[ResidualCandidate]:
    if mode == "smoke":
        return [
            ResidualCandidate("smoke_svd16_scale1.0", residual_rank=16, scale=1.0),
            ResidualCandidate("smoke_svd32_scale1.1", residual_rank=32, scale=1.1),
            ResidualCandidate(
                "smoke_energy_svd32_scale1.1",
                residual_rank=32,
                scale=1.1,
                residual_weighting="residual_energy",
            ),
            ResidualCandidate(
                "smoke_patch16_rank4_scale1.0",
                residual_rank=None,
                scale=1.0,
                head_type="patch_residual_svd",
                patch_size=16,
                patch_residual_rank=4,
            ),
            ResidualCandidate(
                "smoke_innovation_svd16_rank4_scale1.0",
                residual_rank=16,
                scale=1.0,
                innovation_rank=4,
                innovation_include_norms=True,
            ),
            ResidualCandidate(
                "smoke_coeffcal_svd16_blend1.0",
                residual_rank=16,
                scale=1.0,
                coefficient_calibration_type="ridge",
                coefficient_calibration_blend=1.0,
            ),
            ResidualCandidate(
                "smoke_coeffcal_base_blend1.0",
                residual_rank=None,
                scale=0.0,
                head_type="none",
                coefficient_calibration_type="ridge",
                coefficient_calibration_blend=1.0,
            ),
            ResidualCandidate(
                "smoke_coeffdelta_base_blend0.25",
                residual_rank=None,
                scale=0.0,
                head_type="none",
                coefficient_calibration_type="delta_ridge",
                coefficient_calibration_blend=0.25,
            ),
            ResidualCandidate(
                "smoke_coeffdelta_base_ir8_blend0.25",
                residual_rank=None,
                scale=0.0,
                head_type="none",
                coefficient_calibration_type="delta_ridge",
                coefficient_calibration_blend=0.25,
                coefficient_calibration_innovation_rank=8,
                coefficient_calibration_include_norms=True,
            ),
            ResidualCandidate(
                "smoke_tempsmooth_base_a0.2",
                residual_rank=None,
                scale=0.0,
                head_type="none",
                coefficient_temporal_smoothing_alpha=0.2,
            ),
            ResidualCandidate(
                "smoke_tempsmooth_svd16_a0.2_scale1.0",
                residual_rank=16,
                scale=1.0,
                coefficient_temporal_smoothing_alpha=0.2,
            ),
            ResidualCandidate(
                "smoke_hard_svd16_pow1_clip4_scale1.0",
                residual_rank=16,
                scale=1.0,
                sample_weighting="hard_frame_rmse",
                sample_weight_power=1.0,
                sample_weight_clip=4.0,
            ),
            ResidualCandidate(
                "smoke_innovgate_svd16_scale1.0",
                residual_rank=16,
                scale=1.0,
                gate_type="sensor_innovation_rms",
                gate_threshold=0.0,
                gate_strength=1.0,
                gate_min=0.35,
            ),
            ResidualCandidate(
                "smoke_highpass_svd16_cut0.35_scale0.75",
                residual_rank=16,
                scale=0.75,
                residual_target="highpass",
                highpass_cutoff_fraction=0.35,
            ),
            ResidualCandidate(
                "smoke_highpass_patch16_rank4_cut0.35_scale0.75",
                residual_rank=None,
                scale=0.75,
                head_type="patch_residual_svd",
                patch_size=16,
                patch_residual_rank=4,
                residual_target="highpass",
                highpass_cutoff_fraction=0.35,
            ),
            ResidualCandidate(
                "smoke_hfweight_svd16_cut0.35_scale1.0",
                residual_rank=16,
                scale=1.0,
                residual_weighting="highpass_energy",
                highpass_cutoff_fraction=0.35,
            ),
            ResidualCandidate(
                "smoke_hfweight_patch16_rank4_cut0.35_scale1.0",
                residual_rank=None,
                scale=1.0,
                head_type="patch_residual_svd",
                patch_size=16,
                patch_residual_rank=4,
                residual_weighting="highpass_energy",
                highpass_cutoff_fraction=0.35,
            ),
            ResidualCandidate(
                "smoke_composite_patch16_rank4_hf16_cut0.35_p1_hf0.25",
                residual_rank=16,
                scale=1.0,
                hf_scale=0.25,
                head_type="composite_patch_hf_svd",
                patch_size=16,
                patch_residual_rank=4,
                highpass_cutoff_fraction=0.35,
            ),
        ]
    ranks = [128, 192, 256, 320]
    scales = [1.0, 1.05, 1.1, 1.15, 1.2, 1.3]
    candidates: list[ResidualCandidate] = []
    for rank in ranks:
        for scale in scales:
            candidates.append(
                ResidualCandidate(
                    f"uniform_svd{rank}_scale{scale:g}",
                    residual_rank=rank,
                    scale=scale,
                    residual_weighting="uniform",
                )
            )
            if rank in {192, 256, 320} and scale in {1.0, 1.1, 1.2}:
                candidates.append(
                    ResidualCandidate(
                        f"energy_svd{rank}_scale{scale:g}",
                        residual_rank=rank,
                        scale=scale,
                        residual_weighting="residual_energy",
                    )
                )
    for rank in [192, 256]:
        candidates.append(
            ResidualCandidate(
                f"uniform_svd{rank}_scale1.1_alpha1e-6",
                residual_rank=rank,
                scale=1.1,
                alpha=1e-6,
            )
        )
    for power in [0.5, 1.0, 1.5]:
        for clip in [2.5, 4.0]:
            for scale in [1.1, 1.2, 1.3]:
                candidates.append(
                    ResidualCandidate(
                        f"hard_svd256_pow{power:g}_clip{clip:g}_scale{scale:g}",
                        residual_rank=256,
                        scale=scale,
                        sample_weighting="hard_frame_rmse",
                        sample_weight_power=power,
                        sample_weight_clip=clip,
                    )
                )
            for patch_rank in [24, 32]:
                candidates.append(
                    ResidualCandidate(
                        f"hard_patch16_rank{patch_rank}_pow{power:g}_clip{clip:g}_scale1.3",
                        residual_rank=None,
                        scale=1.3,
                        head_type="patch_residual_svd",
                        patch_size=16,
                        patch_residual_rank=patch_rank,
                        sample_weighting="hard_frame_rmse",
                        sample_weight_power=power,
                        sample_weight_clip=clip,
                    )
                )
    for scale in [1.3, 1.5]:
        for threshold in [0.85, 1.0, 1.15]:
            for strength in [0.5, 1.0, 2.0]:
                candidates.append(
                    ResidualCandidate(
                        f"gated_svd256_scale{scale:g}_thr{threshold:g}_str{strength:g}",
                        residual_rank=256,
                        scale=scale,
                        gate_type="coefficient_rms",
                        gate_threshold=threshold,
                        gate_strength=strength,
                        gate_min=0.5,
                    )
                )
    for threshold in [0.0, 0.5, 1.0]:
        for strength in [0.5, 1.0, 2.0]:
            candidates.append(
                ResidualCandidate(
                    f"innovgate_svd256_scale1.3_thr{threshold:g}_str{strength:g}",
                    residual_rank=256,
                    scale=1.3,
                    gate_type="sensor_innovation_rms",
                    gate_threshold=threshold,
                    gate_strength=strength,
                    gate_min=0.5,
                )
            )
            candidates.append(
                ResidualCandidate(
                    f"innovgate_patch16_rank32_scale1.3_thr{threshold:g}_str{strength:g}",
                    residual_rank=None,
                    scale=1.3,
                    head_type="patch_residual_svd",
                    patch_size=16,
                    patch_residual_rank=32,
                    gate_type="sensor_innovation_rms",
                    gate_threshold=threshold,
                    gate_strength=strength,
                    gate_min=0.5,
                )
            )
    for innovation_rank in [8, 16, 32, 64]:
        for scale in [1.0, 1.1, 1.2, 1.3]:
            candidates.append(
                ResidualCandidate(
                    f"innovation_svd256_ir{innovation_rank}_scale{scale:g}",
                    residual_rank=256,
                    scale=scale,
                    innovation_rank=innovation_rank,
                    innovation_include_norms=True,
                )
            )
    for blend in [0.25, 0.5, 0.75, 1.0]:
        for scale in [1.0, 1.1, 1.2]:
            candidates.append(
                ResidualCandidate(
                    f"coeffcal_svd256_blend{blend:g}_scale{scale:g}",
                    residual_rank=256,
                    scale=scale,
                    coefficient_calibration_type="ridge",
                    coefficient_calibration_blend=blend,
                )
            )
    for blend in [0.75, 1.0]:
        candidates.append(
            ResidualCandidate(
                f"coeffcal_base_blend{blend:g}",
                residual_rank=None,
                scale=0.0,
                head_type="none",
                coefficient_calibration_type="ridge",
                coefficient_calibration_blend=blend,
            )
        )
        for rank in [64, 128]:
            for scale in [0.1, 0.25, 0.5, 0.75]:
                candidates.append(
                    ResidualCandidate(
                        f"coeffcal_svd{rank}_blend{blend:g}_scale{scale:g}",
                        residual_rank=rank,
                        scale=scale,
                        coefficient_calibration_type="ridge",
                        coefficient_calibration_blend=blend,
                    )
                )
        for patch_rank in [8, 16]:
            for scale in [0.1, 0.25, 0.5]:
                candidates.append(
                    ResidualCandidate(
                        f"coeffcal_patch16_rank{patch_rank}_blend{blend:g}_scale{scale:g}",
                        residual_rank=None,
                        scale=scale,
                        head_type="patch_residual_svd",
                        patch_size=16,
                        patch_residual_rank=patch_rank,
                        coefficient_calibration_type="ridge",
                        coefficient_calibration_blend=blend,
                    )
                )
    for patch_rank in [24, 32]:
        for blend in [0.5, 1.0]:
            for scale in [1.0, 1.1, 1.2]:
                candidates.append(
                    ResidualCandidate(
                        f"coeffcal_patch16_rank{patch_rank}_blend{blend:g}_scale{scale:g}",
                        residual_rank=None,
                        scale=scale,
                        head_type="patch_residual_svd",
                        patch_size=16,
                        patch_residual_rank=patch_rank,
                        coefficient_calibration_type="ridge",
                        coefficient_calibration_blend=blend,
                    )
                )
    for patch_rank in [24, 32]:
        for innovation_rank in [8, 16, 32]:
            for scale in [1.1, 1.2, 1.3]:
                candidates.append(
                    ResidualCandidate(
                        f"patch16_rank{patch_rank}_ir{innovation_rank}_scale{scale:g}",
                        residual_rank=None,
                        scale=scale,
                        head_type="patch_residual_svd",
                        patch_size=16,
                        patch_residual_rank=patch_rank,
                        innovation_rank=innovation_rank,
                        innovation_include_norms=True,
                    )
                )
    for blend in [0.1, 0.25, 0.5, 0.75]:
        candidates.append(
            ResidualCandidate(
                f"coeffdelta_base_blend{blend:g}",
                residual_rank=None,
                scale=0.0,
                head_type="none",
                coefficient_calibration_type="delta_ridge",
                coefficient_calibration_blend=blend,
            )
        )
        for innovation_rank in [8, 16, 32]:
            candidates.append(
                ResidualCandidate(
                    f"coeffdelta_base_ir{innovation_rank}_blend{blend:g}",
                    residual_rank=None,
                    scale=0.0,
                    head_type="none",
                    coefficient_calibration_type="delta_ridge",
                    coefficient_calibration_blend=blend,
                    coefficient_calibration_innovation_rank=innovation_rank,
                    coefficient_calibration_include_norms=True,
                )
            )
        for residual_rank in [64, 128]:
            for scale in [0.1, 0.25, 0.5]:
                candidates.append(
                    ResidualCandidate(
                        f"coeffdelta_svd{residual_rank}_ir16_blend{blend:g}_scale{scale:g}",
                        residual_rank=residual_rank,
                        scale=scale,
                        coefficient_calibration_type="delta_ridge",
                        coefficient_calibration_blend=blend,
                        coefficient_calibration_innovation_rank=16,
                        coefficient_calibration_include_norms=True,
                    )
                )
    for smoothing_alpha in [0.1, 0.2, 0.35, 0.5]:
        candidates.append(
            ResidualCandidate(
                f"tempsmooth_base_a{smoothing_alpha:g}",
                residual_rank=None,
                scale=0.0,
                head_type="none",
                coefficient_temporal_smoothing_alpha=smoothing_alpha,
            )
        )
        for scale in [1.0, 1.1, 1.2]:
            candidates.append(
                ResidualCandidate(
                    f"tempsmooth_svd256_a{smoothing_alpha:g}_scale{scale:g}",
                    residual_rank=256,
                    scale=scale,
                    coefficient_temporal_smoothing_alpha=smoothing_alpha,
                )
            )
        for residual_rank in [64, 128]:
            for scale in [0.1, 0.25, 0.5]:
                candidates.append(
                    ResidualCandidate(
                        f"tempsmooth_coeffcal_svd{residual_rank}_a{smoothing_alpha:g}_blend1_scale{scale:g}",
                        residual_rank=residual_rank,
                        scale=scale,
                        coefficient_calibration_type="ridge",
                        coefficient_calibration_blend=1.0,
                        coefficient_temporal_smoothing_alpha=smoothing_alpha,
                    )
                )
        if smoothing_alpha in {0.1, 0.2, 0.35}:
            for patch_rank in [24, 32]:
                for scale in [1.2, 1.3]:
                    candidates.append(
                        ResidualCandidate(
                            f"tempsmooth_patch16_rank{patch_rank}_a{smoothing_alpha:g}_scale{scale:g}",
                            residual_rank=None,
                            scale=scale,
                            head_type="patch_residual_svd",
                            patch_size=16,
                            patch_residual_rank=patch_rank,
                            coefficient_temporal_smoothing_alpha=smoothing_alpha,
                        )
                    )
    for patch_size, patch_ranks, scales_for_patch in [
        (16, [8, 12, 16, 24, 32, 48], [1.0, 1.1, 1.2, 1.3]),
        (8, [4, 8, 12], [1.0, 1.1, 1.2]),
    ]:
        for patch_rank in patch_ranks:
            for scale in scales_for_patch:
                candidates.append(
                    ResidualCandidate(
                        f"patch{patch_size}_rank{patch_rank}_scale{scale:g}",
                        residual_rank=None,
                        scale=scale,
                        head_type="patch_residual_svd",
                        patch_size=patch_size,
                        patch_residual_rank=patch_rank,
                    )
                )
    for cutoff in [0.25, 0.35, 0.45]:
        for rank in [128, 192, 256]:
            for scale in [0.5, 0.75, 1.0]:
                candidates.append(
                    ResidualCandidate(
                        f"highpass_svd{rank}_cut{cutoff:g}_scale{scale:g}",
                        residual_rank=rank,
                        scale=scale,
                        residual_target="highpass",
                        highpass_cutoff_fraction=cutoff,
                    )
                )
            for scale in [1.0, 1.1, 1.2]:
                candidates.append(
                    ResidualCandidate(
                        f"hfweight_svd{rank}_cut{cutoff:g}_scale{scale:g}",
                        residual_rank=rank,
                        scale=scale,
                        residual_weighting="highpass_energy",
                        highpass_cutoff_fraction=cutoff,
                    )
                )
        for patch_rank in [16, 24, 32]:
            for scale in [0.5, 0.75, 1.0]:
                candidates.append(
                    ResidualCandidate(
                        f"highpass_patch16_rank{patch_rank}_cut{cutoff:g}_scale{scale:g}",
                        residual_rank=None,
                        scale=scale,
                        head_type="patch_residual_svd",
                        patch_size=16,
                        patch_residual_rank=patch_rank,
                        residual_target="highpass",
                        highpass_cutoff_fraction=cutoff,
                    )
                )
            for scale in [1.0, 1.1, 1.2]:
                candidates.append(
                    ResidualCandidate(
                        f"hfweight_patch16_rank{patch_rank}_cut{cutoff:g}_scale{scale:g}",
                        residual_rank=None,
                        scale=scale,
                        head_type="patch_residual_svd",
                        patch_size=16,
                        patch_residual_rank=patch_rank,
                        residual_weighting="highpass_energy",
                        highpass_cutoff_fraction=cutoff,
                    )
                )
    for cutoff in [0.35, 0.45]:
        for patch_rank in [24, 32]:
            for hf_rank in [192, 256]:
                for patch_scale in [1.1, 1.3]:
                    for hf_scale in [0.1, 0.25, 0.4]:
                        candidates.append(
                            ResidualCandidate(
                                (
                                    f"composite_patch16_rank{patch_rank}_hf{hf_rank}_"
                                    f"cut{cutoff:g}_p{patch_scale:g}_hf{hf_scale:g}"
                                ),
                                residual_rank=hf_rank,
                                scale=patch_scale,
                                hf_scale=hf_scale,
                                head_type="composite_patch_hf_svd",
                                patch_size=16,
                                patch_residual_rank=patch_rank,
                                highpass_cutoff_fraction=cutoff,
                            )
                        )
    return candidates


def filter_candidates(
    candidates: list[ResidualCandidate],
    *,
    family: str,
    candidate_names: list[str] | None = None,
) -> list[ResidualCandidate]:
    """Select a reproducible subset for targeted fast experiments."""
    family = family.lower()
    if family == "all":
        family_candidates = list(candidates)
    elif family not in {"highpass", "hfweight", "composite", "coeffdelta", "temporal"}:
        raise ValueError(f"Unknown candidate family: {family}")
    else:
        controls = {
            "smoke_svd16_scale1.0",
            "smoke_coeffcal_base_blend1.0",
            "uniform_svd256_scale1.1",
            "patch16_rank32_scale1.3",
            "coeffcal_base_blend1",
        }
        if family == "hfweight":
            family_candidates = [
                candidate
                for candidate in candidates
                if "hfweight" in candidate.name or candidate.name in controls
            ]
        elif family == "coeffdelta":
            family_candidates = [
                candidate
                for candidate in candidates
                if "coeffdelta" in candidate.name or candidate.name in controls
            ]
        elif family == "temporal":
            family_candidates = [
                candidate
                for candidate in candidates
                if "tempsmooth" in candidate.name or candidate.name in controls
            ]
        elif family == "composite":
            family_candidates = [
                candidate
                for candidate in candidates
                if "composite" in candidate.name
                or candidate.name
                in {
                    *controls,
                    "hfweight_svd256_cut0.45_scale1.2",
                    "hfweight_svd256_cut0.35_scale1.2",
                }
            ]
        else:
            family_candidates = [
                candidate
                for candidate in candidates
                if "highpass" in candidate.name
                or "hfweight" in candidate.name
                or candidate.name in controls
            ]
    if not candidate_names:
        return family_candidates
    by_name = {candidate.name: candidate for candidate in family_candidates}
    missing = [name for name in candidate_names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown or filtered-out candidate name(s): {missing}")
    return [by_name[name] for name in candidate_names]


def build_dev_blocks(
    n_trajectories: int,
    *,
    dev_count: int,
    n_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_trajectories < 2:
        raise ValueError("Need at least two trajectories")
    if dev_count <= 0 or n_splits <= 0:
        raise ValueError("dev_count and n_splits must be positive")
    if dev_count >= n_trajectories:
        raise ValueError("dev_count leaves no training trajectories")
    max_start = n_trajectories - dev_count
    starts = np.linspace(0, max_start, num=n_splits, dtype=int)
    blocks: list[tuple[np.ndarray, np.ndarray]] = []
    all_indices = np.arange(n_trajectories)
    for split_idx, start in enumerate(starts):
        dev_idx = np.arange(start, start + dev_count)
        train_idx = np.setdiff1d(all_indices, dev_idx, assume_unique=True)
        if train_idx.size == 0:
            raise ValueError(f"Split {split_idx} has no train trajectories")
        blocks.append((train_idx, dev_idx))
    return blocks


def _json_safe(value: Any) -> Any:
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


def _metrics_with_mae(target: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    metrics = _compute_regression_metrics(target, pred)
    metrics["mae"] = float(np.mean(np.abs(np.asarray(target) - np.asarray(pred))))
    return metrics


def compute_hard_bucket_metrics(
    target: np.ndarray,
    pred: np.ndarray,
    base_pred: np.ndarray,
    *,
    hard_fraction: float = 0.2,
    prefix: str = "dev",
) -> dict[str, float | int]:
    """Evaluate prediction quality on easy/mid/hard frame buckets.

    Buckets are defined by the base predictor's per-frame RMSE. This keeps the
    diagnostic independent of the candidate under evaluation and answers:
    "Does this residual head improve the frames the base path finds difficult?"
    """
    target_array = np.asarray(target, dtype=np.float64)
    pred_array = np.asarray(pred, dtype=np.float64)
    base_array = np.asarray(base_pred, dtype=np.float64)
    if target_array.shape != pred_array.shape or target_array.shape != base_array.shape:
        raise ValueError("target, pred, and base_pred must have identical shapes")
    if target_array.ndim < 2:
        raise ValueError("frame arrays must include a sample axis and spatial axes")
    if not 0.0 < hard_fraction <= 0.5:
        raise ValueError("hard_fraction must be in (0, 0.5]")

    n_frames = int(target_array.shape[0])
    if n_frames == 0:
        raise ValueError("at least one frame is required")
    base_frame_rmse = np.sqrt(
        np.mean((target_array - base_array).reshape(n_frames, -1) ** 2, axis=1)
    )
    order = np.argsort(base_frame_rmse)
    bucket_size = max(1, int(np.ceil(n_frames * hard_fraction)))
    easy_idx = order[:bucket_size]
    hard_idx = order[-bucket_size:]
    mid_idx = (
        order[bucket_size:-bucket_size] if n_frames > 2 * bucket_size else np.array([], dtype=int)
    )

    def add_bucket(payload: dict[str, float | int], name: str, indices: np.ndarray) -> None:
        payload[f"{prefix}_{name}_count"] = int(indices.size)
        if indices.size == 0:
            for metric in [
                "r2",
                "rmse",
                "mae",
                "rel_frob_err",
                "base_r2",
                "base_rmse",
                "delta_r2",
                "delta_rmse",
            ]:
                payload[f"{prefix}_{name}_{metric}"] = float("nan")
            return
        candidate_metrics = _metrics_with_mae(target_array[indices], pred_array[indices])
        base_metrics = _metrics_with_mae(target_array[indices], base_array[indices])
        payload[f"{prefix}_{name}_r2"] = candidate_metrics["r2"]
        payload[f"{prefix}_{name}_rmse"] = candidate_metrics["rmse"]
        payload[f"{prefix}_{name}_mae"] = candidate_metrics["mae"]
        payload[f"{prefix}_{name}_rel_frob_err"] = candidate_metrics["rel_frob_err"]
        payload[f"{prefix}_{name}_base_r2"] = base_metrics["r2"]
        payload[f"{prefix}_{name}_base_rmse"] = base_metrics["rmse"]
        payload[f"{prefix}_{name}_delta_r2"] = candidate_metrics["r2"] - base_metrics["r2"]
        payload[f"{prefix}_{name}_delta_rmse"] = candidate_metrics["rmse"] - base_metrics["rmse"]

    result: dict[str, float | int] = {
        f"{prefix}_bucket_fraction": float(hard_fraction),
        f"{prefix}_base_frame_rmse_p50": float(np.quantile(base_frame_rmse, 0.50)),
        f"{prefix}_base_frame_rmse_p80": float(np.quantile(base_frame_rmse, 0.80)),
        f"{prefix}_base_frame_rmse_p95": float(np.quantile(base_frame_rmse, 0.95)),
    }
    add_bucket(result, "easy", easy_idx)
    add_bucket(result, "mid", mid_idx)
    add_bucket(result, "hard", hard_idx)
    return result


def _sensing_diagnostics(dictionary: np.ndarray, sensor_indices: np.ndarray) -> dict[str, float]:
    sensing_matrix = history_sensor_matrix(dictionary, sensor_indices)
    singular_values = np.linalg.svd(sensing_matrix, compute_uv=False)
    positive = singular_values[singular_values > 1e-12]
    condition = float(positive[0] / positive[-1]) if positive.size else float("inf")
    return {
        "sensing_rows": int(sensing_matrix.shape[0]),
        "sensing_cols": int(sensing_matrix.shape[1]),
        "sensing_rank": int(np.linalg.matrix_rank(sensing_matrix, tol=1e-10)),
        "sensing_condition_proxy": condition,
        "sensing_min_singular_value": float(positive[-1]) if positive.size else 0.0,
        "sensing_max_singular_value": float(positive[0]) if positive.size else 0.0,
    }


def build_residual_basis_caches(
    target_frames: np.ndarray,
    base_predictions: np.ndarray,
    candidates: list[ResidualCandidate],
    *,
    random_state: int,
) -> dict[tuple[str, float], dict[str, Any]]:
    """Precompute residual SVD bases once per weighting/floor pair."""
    grouped: dict[tuple[str, float, str, float], int] = {}
    for candidate in candidates:
        if candidate.head_type != "global_residual_svd":
            continue
        if candidate.residual_rank is None:
            continue
        key = (
            candidate.residual_weighting,
            float(candidate.residual_weight_floor),
            candidate.residual_target,
            float(candidate.highpass_cutoff_fraction),
        )
        grouped[key] = max(grouped.get(key, 0), int(candidate.residual_rank))

    caches: dict[tuple[str, float, str, float], dict[str, Any]] = {}
    for (weighting, floor, residual_target, cutoff), max_rank in grouped.items():
        residual = residual_target_frames(
            target_frames,
            base_predictions,
            residual_target=residual_target,
            highpass_cutoff_fraction=cutoff,
        ).reshape(target_frames.shape[0], -1)
        residual_mean = residual.mean(axis=0)
        centered_residual = residual - residual_mean
        if weighting == "uniform":
            residual_weights = np.ones(centered_residual.shape[1], dtype=np.float64)
        elif weighting == "residual_energy":
            if floor <= 0:
                raise ValueError("residual_weight_floor must be positive")
            residual_weights = np.sqrt(np.mean(centered_residual**2, axis=0) + 1e-12)
            residual_weights /= max(float(np.mean(residual_weights)), 1e-12)
            residual_weights = np.maximum(residual_weights, floor)
            residual_weights /= max(float(np.mean(residual_weights)), 1e-12)
        elif weighting == "highpass_energy":
            if floor <= 0:
                raise ValueError("residual_weight_floor must be positive")
            highpass = fft_highpass_frames(
                centered_residual.reshape(
                    centered_residual.shape[0],
                    *target_frames.shape[1:],
                ),
                cutoff_fraction=cutoff,
            ).reshape(centered_residual.shape)
            residual_weights = np.sqrt(np.mean(highpass**2, axis=0) + 1e-12)
            residual_weights /= max(float(np.mean(residual_weights)), 1e-12)
            residual_weights = np.maximum(residual_weights, floor)
            residual_weights /= max(float(np.mean(residual_weights)), 1e-12)
        else:
            raise ValueError(f"Unknown residual_weighting: {weighting}")
        weighted_residual = centered_residual * residual_weights
        actual_rank = min(max_rank, weighted_residual.shape[0], weighted_residual.shape[1])
        if actual_rank <= 0:
            raise ValueError("residual_rank must be positive")
        _, _, vt = randomized_svd(
            weighted_residual,
            n_components=actual_rank,
            n_iter=4,
            random_state=random_state,
        )
        caches[(weighting, floor, residual_target, cutoff)] = {
            "residual_mean": residual_mean,
            "residual_weights": residual_weights,
            "weighted_residual": weighted_residual,
            "residual_basis": vt,
            "max_rank": int(actual_rank),
            "residual_target": residual_target,
            "highpass_cutoff_fraction": float(cutoff),
        }
    return caches


def fit_cached_ridge_residual_corrector(
    target_frames: np.ndarray,
    base_predictions: np.ndarray,
    coeffs: np.ndarray,
    candidate: ResidualCandidate,
    residual_basis_caches: dict[tuple[str, float, str, float], dict[str, Any]],
    *,
    feature_matrix: np.ndarray | None = None,
) -> dict[str, Any]:
    target_flat = np.asarray(target_frames, dtype=np.float64).reshape(target_frames.shape[0], -1)
    base_flat = np.asarray(base_predictions, dtype=np.float64).reshape(
        base_predictions.shape[0], -1
    )
    coeffs = np.asarray(coeffs, dtype=np.float64)
    model_features = (
        coeffs if feature_matrix is None else np.asarray(feature_matrix, dtype=np.float64)
    )
    features = np.concatenate(
        [model_features, np.ones((model_features.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    if candidate.alpha < 0:
        raise ValueError("alpha must be non-negative")

    mode = "full"
    residual_basis = None
    residual_mean = None
    residual_weights = None
    actual_residual_rank = None
    regression_target = target_flat - base_flat
    if candidate.head_type != "global_residual_svd":
        raise ValueError("fit_cached_ridge_residual_corrector only supports global residual SVD")
    if candidate.residual_rank is not None:
        key = (
            candidate.residual_weighting,
            float(candidate.residual_weight_floor),
            candidate.residual_target,
            float(candidate.highpass_cutoff_fraction),
        )
        basis_cache = residual_basis_caches[key]
        actual_residual_rank = min(int(candidate.residual_rank), int(basis_cache["max_rank"]))
        residual_basis = np.asarray(basis_cache["residual_basis"], dtype=np.float64)[
            :actual_residual_rank
        ]
        residual_mean = np.asarray(basis_cache["residual_mean"], dtype=np.float64)
        residual_weights = np.asarray(basis_cache["residual_weights"], dtype=np.float64)
        regression_target = (
            np.asarray(basis_cache["weighted_residual"], dtype=np.float64) @ residual_basis.T
        )
        mode = "residual_svd"

    gram = features.T @ features
    penalty = candidate.alpha * np.eye(features.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    rhs = features.T @ regression_target
    try:
        weights = np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(gram + penalty, rhs, rcond=None)[0]

    corrector = {
        "alpha": float(candidate.alpha),
        "weights": weights,
        "feature_dim": int(model_features.shape[1]),
        "coefficient_dim": int(coeffs.shape[1]),
        "output_dim": int(target_flat.shape[1]),
        "mode": mode,
        "residual_rank": actual_residual_rank,
        "residual_weighting": candidate.residual_weighting if mode == "residual_svd" else "uniform",
        "residual_target": candidate.residual_target,
        "highpass_cutoff_fraction": float(candidate.highpass_cutoff_fraction),
    }
    if mode == "residual_svd":
        corrector["residual_basis"] = residual_basis
        corrector["residual_mean"] = residual_mean
        corrector["residual_weights"] = residual_weights
    return corrector


def build_split_cache(
    states: np.ndarray,
    *,
    train_idx: np.ndarray,
    dev_idx: np.ndarray,
    base_config: CachedBaseConfig,
) -> dict[str, Any]:
    split_start = time.perf_counter()
    train_states = np.asarray(states[train_idx], dtype=np.float64)
    dev_states = np.asarray(states[dev_idx], dtype=np.float64)
    spatial_mean = np.mean(train_states, axis=(0, 1))
    train_centered = train_states - spatial_mean
    dev_centered = dev_states - spatial_mean
    train_segments, train_refs = build_forecast_segment_tensor_with_refs(
        train_centered,
        history_length=base_config.history_length,
        stride=1,
        max_segments=base_config.max_train_segments,
    )
    dev_segments, dev_refs = build_forecast_segment_tensor_with_refs(
        dev_centered,
        history_length=base_config.history_length,
        stride=1,
        max_segments=None,
    )
    dictionary, tbmd_summary = fit_segment_dictionary(
        train_segments,
        ranks=list(base_config.ranks),
        random_state=base_config.random_state,
        dtype=base_config.dtype,
    )
    history_dictionary, _ = history_and_target(dictionary)
    spatial_mask, sensor_indices = place_fixed_spatial_sensors(
        history_dictionary,
        n_spatial_sensors=base_config.n_spatial_sensors,
        random_state=base_config.random_state,
    )
    decoder_payload = fit_sensor_coefficient_decoder(
        dictionary,
        sensor_indices,
        decoder=base_config.sensor_decoder,
        rcond=base_config.sensor_rcond,
        ridge_lambda=base_config.decoder_ridge_lambda,
    )
    train_base, train_coeffs, train_measurements = predict_next_sensor_decoder_with_measurements(
        train_segments,
        dictionary,
        sensor_indices,
        decoder_payload,
    )
    dev_base, dev_coeffs, dev_measurements = predict_next_sensor_decoder_with_measurements(
        dev_segments,
        dictionary,
        sensor_indices,
        decoder_payload,
    )
    train_targets = target_frames_from_segments(train_segments)
    dev_targets = target_frames_from_segments(dev_segments)
    elapsed = time.perf_counter() - split_start
    return {
        "train_targets": train_targets,
        "train_base": train_base,
        "train_coeffs": train_coeffs,
        "train_refs": train_refs,
        "train_measurements": train_measurements,
        "dev_targets": dev_targets,
        "dev_base": dev_base,
        "dev_coeffs": dev_coeffs,
        "dev_refs": dev_refs,
        "dev_measurements": dev_measurements,
        "decoder_payload": decoder_payload,
        "dictionary": dictionary,
        "base_train_metrics": _metrics_with_mae(train_targets, train_base),
        "base_dev_metrics": _metrics_with_mae(dev_targets, dev_base),
        "tbmd_summary": tbmd_summary,
        "dictionary_shape": list(dictionary.shape),
        "actual_spatial_sensors": int(spatial_mask.sum()),
        "sensing_diagnostics": _sensing_diagnostics(dictionary, sensor_indices),
        "cache_time_seconds": float(elapsed),
        "train_shape": list(train_states.shape),
        "dev_shape": list(dev_states.shape),
        "train_segments_shape": list(train_segments.shape),
        "dev_segments_shape": list(dev_segments.shape),
    }


def evaluate_residual_candidate(
    cache: dict[str, Any],
    candidate: ResidualCandidate,
    residual_basis_caches: dict[tuple[str, float, str, float], dict[str, Any]],
    *,
    include_structure_metrics: bool = False,
    hard_bucket_fraction: float = 0.2,
) -> dict[str, Any]:
    start = time.perf_counter()
    coefficient_calibrator = fit_coefficient_calibrator(
        cache["train_coeffs"],
        cache["train_targets"],
        cache["dictionary"],
        calibration_type=candidate.coefficient_calibration_type,
        target="target",
        alpha=candidate.coefficient_calibration_alpha,
        blend=candidate.coefficient_calibration_blend,
        rcond=1e-6,
        measurements=cache["train_measurements"],
        decoder_payload=cache["decoder_payload"],
        innovation_rank=candidate.coefficient_calibration_innovation_rank,
        include_norms=candidate.coefficient_calibration_include_norms,
        random_state=int(cache.get("split_index", 0)),
    )
    train_coeffs = apply_coefficient_calibrator(
        cache["train_coeffs"],
        coefficient_calibrator,
        measurements=cache["train_measurements"],
        decoder_payload=cache["decoder_payload"],
    )
    dev_coeffs = apply_coefficient_calibrator(
        cache["dev_coeffs"],
        coefficient_calibrator,
        measurements=cache["dev_measurements"],
        decoder_payload=cache["decoder_payload"],
    )
    if candidate.coefficient_temporal_smoothing_alpha > 0.0:
        train_coeffs = smooth_coefficients_by_segment_refs(
            train_coeffs,
            cache["train_refs"],
            alpha=candidate.coefficient_temporal_smoothing_alpha,
            reset_on_gap=candidate.coefficient_temporal_reset_on_gap,
        )
        dev_coeffs = smooth_coefficients_by_segment_refs(
            dev_coeffs,
            cache["dev_refs"],
            alpha=candidate.coefficient_temporal_smoothing_alpha,
            reset_on_gap=candidate.coefficient_temporal_reset_on_gap,
        )
    coefficients_changed = (
        coefficient_calibrator.get("type", "none") != "none"
        or candidate.coefficient_temporal_smoothing_alpha > 0.0
    )
    train_base = (
        cache["train_base"]
        if not coefficients_changed
        else reconstruct_target_from_coefficients(train_coeffs, cache["dictionary"])
    )
    dev_base = (
        cache["dev_base"]
        if not coefficients_changed
        else reconstruct_target_from_coefficients(dev_coeffs, cache["dictionary"])
    )
    innovation_encoder = fit_sensor_innovation_encoder(
        cache["train_measurements"],
        train_coeffs,
        cache["decoder_payload"],
        rank=candidate.innovation_rank,
        include_norms=candidate.innovation_include_norms,
        random_state=int(cache.get("split_index", 0)),
    )
    feature_probe = {"innovation_encoder": innovation_encoder}
    train_feature_matrix = build_correction_feature_matrix(
        train_coeffs,
        feature_probe,
        measurements=cache["train_measurements"],
        decoder_payload=cache["decoder_payload"],
    )
    if candidate.head_type == "none":
        corrector = fit_noop_residual_corrector(cache["train_targets"], train_coeffs)
    elif candidate.head_type == "composite_patch_hf_svd":
        if candidate.patch_size is None or candidate.patch_residual_rank is None:
            raise ValueError("composite candidates require patch_size and patch_residual_rank")
        if candidate.residual_rank is None:
            raise ValueError("composite candidates require residual_rank as HF rank")
        corrector = fit_composite_patch_hf_residual_corrector(
            cache["train_targets"],
            train_base,
            train_coeffs,
            alpha=candidate.alpha,
            patch_size=candidate.patch_size,
            patch_residual_rank=candidate.patch_residual_rank,
            hf_residual_rank=candidate.residual_rank,
            patch_scale=candidate.scale,
            hf_scale=candidate.hf_scale,
            highpass_cutoff_fraction=candidate.highpass_cutoff_fraction,
            residual_weight_floor=candidate.residual_weight_floor,
            feature_matrix=train_feature_matrix,
        )
    elif candidate.head_type == "patch_residual_svd":
        if candidate.patch_size is None or candidate.patch_residual_rank is None:
            raise ValueError(
                "patch_residual_svd candidates require patch_size and patch_residual_rank"
            )
        corrector = fit_patch_residual_svd_corrector(
            cache["train_targets"],
            train_base,
            train_coeffs,
            alpha=candidate.alpha,
            patch_size=candidate.patch_size,
            patch_residual_rank=candidate.patch_residual_rank,
            residual_weighting=candidate.residual_weighting,
            residual_weight_floor=candidate.residual_weight_floor,
            sample_weighting=candidate.sample_weighting,
            sample_weight_power=candidate.sample_weight_power,
            sample_weight_floor=candidate.sample_weight_floor,
            sample_weight_clip=candidate.sample_weight_clip,
            residual_target=candidate.residual_target,
            highpass_cutoff_fraction=candidate.highpass_cutoff_fraction,
            feature_matrix=train_feature_matrix,
        )
    else:
        if (
            coefficient_calibrator.get("type", "none") == "none"
            and candidate.sample_weighting == "uniform"
        ):
            corrector = fit_cached_ridge_residual_corrector(
                cache["train_targets"],
                train_base,
                train_coeffs,
                candidate,
                residual_basis_caches,
                feature_matrix=train_feature_matrix,
            )
        else:
            corrector = fit_ridge_residual_corrector(
                cache["train_targets"],
                train_base,
                train_coeffs,
                alpha=candidate.alpha,
                residual_rank=candidate.residual_rank,
                residual_weighting=candidate.residual_weighting,
                residual_weight_floor=candidate.residual_weight_floor,
                sample_weighting=candidate.sample_weighting,
                sample_weight_power=candidate.sample_weight_power,
                sample_weight_floor=candidate.sample_weight_floor,
                sample_weight_clip=candidate.sample_weight_clip,
                residual_target=candidate.residual_target,
                highpass_cutoff_fraction=candidate.highpass_cutoff_fraction,
                feature_matrix=train_feature_matrix,
            )
    corrector["innovation_encoder"] = innovation_encoder
    corrector = attach_coefficient_gate(
        corrector,
        train_coeffs,
        gate_type=candidate.gate_type,
        threshold=candidate.gate_threshold,
        strength=candidate.gate_strength,
        gate_min=candidate.gate_min,
        measurements=cache["train_measurements"],
        decoder_payload=cache["decoder_payload"],
    )
    train_pred = apply_ridge_residual_corrector(
        train_base,
        train_coeffs,
        corrector,
        scale=1.0 if candidate.head_type == "composite_patch_hf_svd" else candidate.scale,
        measurements=cache["train_measurements"],
        decoder_payload=cache["decoder_payload"],
    )
    dev_pred = apply_ridge_residual_corrector(
        dev_base,
        dev_coeffs,
        corrector,
        scale=1.0 if candidate.head_type == "composite_patch_hf_svd" else candidate.scale,
        measurements=cache["dev_measurements"],
        decoder_payload=cache["decoder_payload"],
    )
    elapsed = time.perf_counter() - start
    train_metrics = _metrics_with_mae(cache["train_targets"], train_pred)
    dev_metrics = _metrics_with_mae(cache["dev_targets"], dev_pred)
    base_train_metrics = _metrics_with_mae(cache["train_targets"], train_base)
    base_dev_metrics = _metrics_with_mae(cache["dev_targets"], dev_base)
    row = {
        "candidate": candidate.name,
        "residual_rank": candidate.residual_rank,
        "actual_residual_rank": corrector.get("residual_rank"),
        "head_type": candidate.head_type,
        "patch_size": candidate.patch_size,
        "patch_residual_rank": candidate.patch_residual_rank,
        "actual_patch_residual_rank": corrector.get("patch_residual_rank"),
        "gate_type": candidate.gate_type,
        "gate_threshold": candidate.gate_threshold,
        "gate_strength": candidate.gate_strength,
        "gate_min": candidate.gate_min,
        "innovation_rank": candidate.innovation_rank,
        "actual_innovation_rank": innovation_encoder.get("innovation_rank", 0),
        "innovation_include_norms": candidate.innovation_include_norms,
        "coefficient_calibration_type": candidate.coefficient_calibration_type,
        "coefficient_calibration_blend": candidate.coefficient_calibration_blend,
        "coefficient_calibration_alpha": candidate.coefficient_calibration_alpha,
        "coefficient_calibration_innovation_rank": candidate.coefficient_calibration_innovation_rank,
        "coefficient_calibration_include_norms": candidate.coefficient_calibration_include_norms,
        "coefficient_temporal_smoothing_alpha": candidate.coefficient_temporal_smoothing_alpha,
        "coefficient_temporal_reset_on_gap": candidate.coefficient_temporal_reset_on_gap,
        "base_train_r2": base_train_metrics["r2"],
        "base_dev_r2": base_dev_metrics["r2"],
        "base_dev_rmse": base_dev_metrics["rmse"],
        "scale": candidate.scale,
        "hf_scale": candidate.hf_scale,
        "alpha": candidate.alpha,
        "residual_target": candidate.residual_target,
        "highpass_cutoff_fraction": candidate.highpass_cutoff_fraction,
        "residual_weighting": candidate.residual_weighting,
        "residual_weight_floor": candidate.residual_weight_floor,
        "sample_weighting": candidate.sample_weighting,
        "sample_weight_power": candidate.sample_weight_power,
        "sample_weight_floor": candidate.sample_weight_floor,
        "sample_weight_clip": candidate.sample_weight_clip,
        "sample_weight_min": corrector.get("sample_weight_min", 1.0),
        "sample_weight_max": corrector.get("sample_weight_max", 1.0),
        "train_r2": train_metrics["r2"],
        "train_rmse": train_metrics["rmse"],
        "train_mae": train_metrics["mae"],
        "train_rel_frob_err": train_metrics["rel_frob_err"],
        "dev_r2": dev_metrics["r2"],
        "dev_rmse": dev_metrics["rmse"],
        "dev_mae": dev_metrics["mae"],
        "dev_rel_frob_err": dev_metrics["rel_frob_err"],
        "correction_fit_eval_time_seconds": float(elapsed),
    }
    row.update(
        compute_hard_bucket_metrics(
            cache["dev_targets"],
            dev_pred,
            dev_base,
            hard_fraction=hard_bucket_fraction,
        )
    )
    if include_structure_metrics:
        structure_metrics = compute_structure_metrics(cache["dev_targets"], dev_pred)
        row.update(
            {
                "dev_structure_score": structure_metrics["structure_score"],
                "dev_gradient_rel_frob_err": structure_metrics["gradient_rel_frob_err"],
                "dev_laplacian_rel_frob_err": structure_metrics["laplacian_rel_frob_err"],
                "dev_radial_spectrum_rel_err": structure_metrics["radial_spectrum_rel_err"],
                "dev_high_frequency_energy_rel_err": structure_metrics[
                    "high_frequency_energy_rel_err"
                ],
                "dev_spatial_corr": structure_metrics["spatial_corr"],
            }
        )
    return row


def aggregate_candidate_results(split_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in split_results:
        by_candidate.setdefault(str(row["candidate"]), []).append(row)
    aggregated = []
    for name, rows in by_candidate.items():
        dev_r2 = np.array([row["dev_r2"] for row in rows], dtype=np.float64)
        dev_rmse = np.array([row["dev_rmse"] for row in rows], dtype=np.float64)
        dev_mae = np.array([row["dev_mae"] for row in rows], dtype=np.float64)
        train_r2 = np.array([row["train_r2"] for row in rows], dtype=np.float64)
        first = rows[0]
        aggregated.append(
            {
                "candidate": name,
                "residual_rank": first["residual_rank"],
                "actual_residual_rank": first["actual_residual_rank"],
                "head_type": first["head_type"],
                "patch_size": first["patch_size"],
                "patch_residual_rank": first["patch_residual_rank"],
                "actual_patch_residual_rank": first["actual_patch_residual_rank"],
                "gate_type": first["gate_type"],
                "gate_threshold": first["gate_threshold"],
                "gate_strength": first["gate_strength"],
                "gate_min": first["gate_min"],
                "innovation_rank": first["innovation_rank"],
                "actual_innovation_rank": first["actual_innovation_rank"],
                "innovation_include_norms": first["innovation_include_norms"],
                "coefficient_calibration_type": first["coefficient_calibration_type"],
                "coefficient_calibration_blend": first["coefficient_calibration_blend"],
                "coefficient_calibration_alpha": first["coefficient_calibration_alpha"],
                "coefficient_calibration_innovation_rank": first.get(
                    "coefficient_calibration_innovation_rank",
                    0,
                ),
                "coefficient_calibration_include_norms": first.get(
                    "coefficient_calibration_include_norms",
                    False,
                ),
                "coefficient_temporal_smoothing_alpha": first.get(
                    "coefficient_temporal_smoothing_alpha",
                    0.0,
                ),
                "coefficient_temporal_reset_on_gap": first.get(
                    "coefficient_temporal_reset_on_gap",
                    True,
                ),
                "mean_base_dev_r2": float(np.mean([row["base_dev_r2"] for row in rows])),
                "mean_base_dev_rmse": float(np.mean([row["base_dev_rmse"] for row in rows])),
                "scale": first["scale"],
                "hf_scale": first.get("hf_scale", 0.0),
                "alpha": first["alpha"],
                "residual_target": first["residual_target"],
                "highpass_cutoff_fraction": first["highpass_cutoff_fraction"],
                "residual_weighting": first["residual_weighting"],
                "sample_weighting": first["sample_weighting"],
                "sample_weight_power": first["sample_weight_power"],
                "sample_weight_floor": first["sample_weight_floor"],
                "sample_weight_clip": first["sample_weight_clip"],
                "mean_sample_weight_min": float(
                    np.mean([row["sample_weight_min"] for row in rows])
                ),
                "mean_sample_weight_max": float(
                    np.mean([row["sample_weight_max"] for row in rows])
                ),
                "n_splits": len(rows),
                "mean_dev_r2": float(np.mean(dev_r2)),
                "std_dev_r2": float(np.std(dev_r2)),
                "worst_dev_r2": float(np.min(dev_r2)),
                "best_dev_r2": float(np.max(dev_r2)),
                "mean_dev_rmse": float(np.mean(dev_rmse)),
                "mean_dev_mae": float(np.mean(dev_mae)),
                "mean_train_r2": float(np.mean(train_r2)),
                "mean_correction_time_seconds": float(
                    np.mean([row["correction_fit_eval_time_seconds"] for row in rows])
                ),
            }
        )
        for metric_key in [
            "dev_structure_score",
            "dev_gradient_rel_frob_err",
            "dev_laplacian_rel_frob_err",
            "dev_radial_spectrum_rel_err",
            "dev_high_frequency_energy_rel_err",
            "dev_spatial_corr",
            "dev_easy_r2",
            "dev_easy_rmse",
            "dev_easy_base_r2",
            "dev_easy_base_rmse",
            "dev_easy_delta_r2",
            "dev_easy_delta_rmse",
            "dev_mid_r2",
            "dev_mid_rmse",
            "dev_mid_base_r2",
            "dev_mid_base_rmse",
            "dev_mid_delta_r2",
            "dev_mid_delta_rmse",
            "dev_hard_r2",
            "dev_hard_rmse",
            "dev_hard_base_r2",
            "dev_hard_base_rmse",
            "dev_hard_delta_r2",
            "dev_hard_delta_rmse",
            "dev_base_frame_rmse_p50",
            "dev_base_frame_rmse_p80",
            "dev_base_frame_rmse_p95",
        ]:
            if metric_key in first:
                values = np.asarray([row[metric_key] for row in rows], dtype=np.float64)
                aggregated[-1][f"mean_{metric_key}"] = float(np.mean(values))
                aggregated[-1][f"worst_{metric_key}"] = float(
                    np.min(values)
                    if metric_key.endswith("_r2") or metric_key == "dev_spatial_corr"
                    else np.max(values)
                )
        for metric_key in ["dev_easy_count", "dev_mid_count", "dev_hard_count"]:
            if metric_key in first:
                values = np.asarray([row[metric_key] for row in rows], dtype=np.float64)
                aggregated[-1][f"mean_{metric_key}"] = float(np.mean(values))
        if "dev_bucket_fraction" in first:
            aggregated[-1]["dev_bucket_fraction"] = float(first["dev_bucket_fraction"])
    return sorted(
        aggregated,
        key=lambda item: (
            item["mean_dev_r2"],
            item["worst_dev_r2"],
            -item["mean_dev_rmse"],
        ),
        reverse=True,
    )


def select_best_robust_result(
    aggregated: list[dict[str, Any]],
    *,
    objective: str = "mean",
    mean_r2_tolerance: float = 0.0,
) -> dict[str, Any]:
    if not aggregated:
        raise ValueError("aggregated results must not be empty")
    if objective not in {"mean", "hard_bucket"}:
        raise ValueError("objective must be 'mean' or 'hard_bucket'")
    if objective == "hard_bucket" and "mean_dev_hard_r2" in aggregated[0]:
        best_mean = max(float(item["mean_dev_r2"]) for item in aggregated)
        eligible = [
            item
            for item in aggregated
            if float(item["mean_dev_r2"]) >= best_mean - float(mean_r2_tolerance)
        ]
        return max(
            eligible,
            key=lambda item: (
                item["mean_dev_hard_r2"],
                item["worst_dev_hard_r2"],
                item.get("mean_dev_hard_delta_r2", 0.0),
                item["mean_dev_r2"],
                item["worst_dev_r2"],
                -item["mean_dev_rmse"],
            ),
        )
    return max(
        aggregated,
        key=lambda item: (
            item["mean_dev_r2"],
            item["worst_dev_r2"],
            -item["mean_dev_rmse"],
        ),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Stage 5 Cached Residual Sweep",
        "",
        "Protocol: train-only multi-dev sweep. Official test is not loaded or evaluated.",
        "",
        "## Selected",
        "",
    ]
    selected = payload["selected_by_multi_dev"]
    lines.append(
        "| candidate | head | mean dev R2 | hard dev R2 | hard delta R2 | worst dev R2 | std dev R2 | mean RMSE | scale | rank | patch | target | calib | innov | gate | weighting |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|")
    lines.append(
        "| {candidate} | {head_type} | {mean_dev_r2:.4f} | {hard_r2:.4f} | {hard_delta:.4f} | {worst_dev_r2:.4f} | "
        "{std_dev_r2:.4f} | {mean_dev_rmse:.4f} | {scale:g} | {rank} | {patch} | {target} | "
        "{calib} | {innov} | {gate} | {residual_weighting} |".format(
            rank=selected.get("actual_residual_rank") or selected.get("actual_patch_residual_rank"),
            hard_r2=selected.get("mean_dev_hard_r2", float("nan")),
            hard_delta=selected.get("mean_dev_hard_delta_r2", float("nan")),
            patch=(
                ""
                if selected.get("patch_size") is None
                else f"{selected.get('patch_size')}/{selected.get('actual_patch_residual_rank')}"
            ),
            gate=(
                "none"
                if selected.get("gate_type") == "none"
                else f"{selected.get('gate_threshold')}/{selected.get('gate_strength')}"
            ),
            innov=(
                ""
                if selected.get("actual_innovation_rank", 0) == 0
                and not selected.get("innovation_include_norms")
                else f"{selected.get('actual_innovation_rank')}/norms={selected.get('innovation_include_norms')}"
            ),
            calib=(
                "none"
                if selected.get("coefficient_calibration_type") == "none"
                else f"{selected.get('coefficient_calibration_type')}/{selected.get('coefficient_calibration_blend')}"
            ),
            target=(
                selected.get("residual_target", "field")
                if selected.get("residual_target", "field") == "field"
                else f"{selected.get('residual_target')}/{selected.get('highpass_cutoff_fraction')}"
            ),
            **selected,
        )
    )
    lines.extend(["", "## Top Candidates", ""])
    lines.append(
        "| candidate | head | mean dev R2 | hard dev R2 | hard delta R2 | worst dev R2 | std dev R2 | mean RMSE | scale | rank | patch | target | calib | innov | gate | weighting |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|")
    for row in payload["aggregated_results"][:12]:
        lines.append(
            "| {candidate} | {head_type} | {mean_dev_r2:.4f} | {hard_r2:.4f} | {hard_delta:.4f} | {worst_dev_r2:.4f} | "
            "{std_dev_r2:.4f} | {mean_dev_rmse:.4f} | {scale:g} | {rank} | {patch} | {target} | "
            "{calib} | {innov} | {gate} | {residual_weighting} |".format(
                rank=row.get("actual_residual_rank") or row.get("actual_patch_residual_rank"),
                hard_r2=row.get("mean_dev_hard_r2", float("nan")),
                hard_delta=row.get("mean_dev_hard_delta_r2", float("nan")),
                patch=(
                    ""
                    if row.get("patch_size") is None
                    else f"{row.get('patch_size')}/{row.get('actual_patch_residual_rank')}"
                ),
                gate=(
                    "none"
                    if row.get("gate_type") == "none"
                    else f"{row.get('gate_threshold')}/{row.get('gate_strength')}"
                ),
                innov=(
                    ""
                    if row.get("actual_innovation_rank", 0) == 0
                    and not row.get("innovation_include_norms")
                    else f"{row.get('actual_innovation_rank')}/norms={row.get('innovation_include_norms')}"
                ),
                calib=(
                    "none"
                    if row.get("coefficient_calibration_type") == "none"
                    else f"{row.get('coefficient_calibration_type')}/{row.get('coefficient_calibration_blend')}"
                ),
                target=(
                    row.get("residual_target", "field")
                    if row.get("residual_target", "field") == "field"
                    else f"{row.get('residual_target')}/{row.get('highpass_cutoff_fraction')}"
                ),
                **row,
            )
        )
    lines.extend(["", "## Base Diagnostics", ""])
    lines.append(
        "| split | base dev R2 | base dev RMSE | sensors | condition proxy | cache time sec | residual SVD sec |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for row in payload["split_diagnostics"]:
        lines.append(
            "| {split_index} | {base_dev_r2:.4f} | {base_dev_rmse:.4f} | {actual_spatial_sensors} | "
            "{sensing_condition_proxy:.2f} | {cache_time_seconds:.2f} | "
            "{residual_basis_cache_time_seconds:.2f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "fast"], default="smoke")
    parser.add_argument("--n-train-trajectories", type=int, default=None)
    parser.add_argument("--n-dev-trajectories", type=int, default=None)
    parser.add_argument("--n-splits", type=int, default=None)
    parser.add_argument("--history-length", type=int, default=7)
    parser.add_argument("--ranks", type=int, nargs=4, default=None)
    parser.add_argument("--n-spatial-sensors", type=int, default=None)
    parser.add_argument("--max-train-segments", type=int, default=None)
    parser.add_argument("--sensor-decoder", choices=["lstsq", "ridge"], default="ridge")
    parser.add_argument("--decoder-ridge-lambda", type=float, default=1e-8)
    parser.add_argument("--sensor-rcond", type=float, default=1e-6)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--structure-metrics", action="store_true")
    parser.add_argument(
        "--candidate-family",
        choices=["all", "highpass", "hfweight", "composite", "coeffdelta", "temporal"],
        default="all",
    )
    parser.add_argument(
        "--candidate-names",
        nargs="*",
        default=None,
        help="Optional exact candidate names to run after applying the family filter.",
    )
    parser.add_argument("--hard-bucket-fraction", type=float, default=0.2)
    parser.add_argument(
        "--selection-objective",
        choices=["mean", "hard-bucket"],
        default="mean",
        help="Use mean dev R2 selection or hard-bucket selection within a mean-R2 tolerance.",
    )
    parser.add_argument("--hard-r2-tolerance", type=float, default=0.005)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def resolve_limits(args: argparse.Namespace) -> tuple[int, int, int]:
    if args.mode == "smoke":
        default_train = 96
        default_dev = 16
        default_splits = 2
    else:
        default_train = 800
        default_dev = 80
        default_splits = 2
    n_train = args.n_train_trajectories or default_train
    n_dev = args.n_dev_trajectories or default_dev
    n_splits = args.n_splits or default_splits
    if n_train > DEFAULT_N_TRAIN_TRAJECTORIES:
        raise ValueError(f"n-train-trajectories cannot exceed {DEFAULT_N_TRAIN_TRAJECTORIES}")
    return n_train, n_dev, n_splits


def main() -> None:
    args = parse_args()
    n_train, n_dev, n_splits = resolve_limits(args)
    base_config = build_base_config(args.mode, args)
    candidates = filter_candidates(
        build_residual_candidates(args.mode),
        family=args.candidate_family,
        candidate_names=args.candidate_names,
    )

    dataset = load_navier_stokes_trajectory_dataset(DATA_ROOT)
    train_pool = dataset.train_states[:n_train]
    blocks = build_dev_blocks(
        train_pool.shape[0],
        dev_count=n_dev,
        n_splits=n_splits,
    )

    split_results: list[dict[str, Any]] = []
    split_diagnostics: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    run_start = time.perf_counter()
    for split_index, (train_idx, dev_idx) in enumerate(blocks):
        cache = build_split_cache(
            train_pool,
            train_idx=train_idx,
            dev_idx=dev_idx,
            base_config=base_config,
        )
        base_dev = cache["base_dev_metrics"]
        sensing = cache["sensing_diagnostics"]
        residual_cache_start = time.perf_counter()
        residual_basis_caches = build_residual_basis_caches(
            cache["train_targets"],
            cache["train_base"],
            candidates,
            random_state=base_config.random_state + split_index,
        )
        residual_cache_time = time.perf_counter() - residual_cache_start
        split_diagnostics.append(
            {
                "split_index": split_index,
                "train_indices": train_idx.tolist(),
                "dev_indices": dev_idx.tolist(),
                "train_shape": cache["train_shape"],
                "dev_shape": cache["dev_shape"],
                "train_segments_shape": cache["train_segments_shape"],
                "dev_segments_shape": cache["dev_segments_shape"],
                "dictionary_shape": cache["dictionary_shape"],
                "actual_spatial_sensors": cache["actual_spatial_sensors"],
                "base_dev_r2": base_dev["r2"],
                "base_dev_rmse": base_dev["rmse"],
                "base_dev_mae": base_dev["mae"],
                "base_train_r2": cache["base_train_metrics"]["r2"],
                "cache_time_seconds": cache["cache_time_seconds"],
                "residual_basis_cache_time_seconds": float(residual_cache_time),
                **sensing,
            }
        )
        for candidate in candidates:
            row = evaluate_residual_candidate(
                cache,
                candidate,
                residual_basis_caches,
                include_structure_metrics=args.structure_metrics,
                hard_bucket_fraction=args.hard_bucket_fraction,
            )
            row["split_index"] = split_index
            split_results.append(row)
        details.append(
            {
                "split_index": split_index,
                "tbmd_summary": cache["tbmd_summary"],
                "base_train_metrics": cache["base_train_metrics"],
                "base_dev_metrics": cache["base_dev_metrics"],
                "sensing_diagnostics": sensing,
                "residual_basis_cache_shapes": {
                    f"{key[0]}:{key[1]:g}": list(value["residual_basis"].shape)
                    for key, value in residual_basis_caches.items()
                },
                "residual_basis_cache_time_seconds": float(residual_cache_time),
            }
        )

    aggregated = aggregate_candidate_results(split_results)
    selected = select_best_robust_result(
        aggregated,
        objective=args.selection_objective.replace("-", "_"),
        mean_r2_tolerance=args.hard_r2_tolerance,
    )
    payload = {
        "stage": "stage5_fast_tplus1_cached_residual",
        "protocol": (
            "Train-only multi-dev residual sweep. The official test split is not evaluated "
            "and must not be used for candidate selection."
        ),
        "mode": args.mode,
        "base_config": asdict(base_config),
        "limits": {
            "n_train_trajectories": n_train,
            "n_dev_trajectories_per_split": n_dev,
            "n_splits": n_splits,
            "official_test_evaluated": False,
            "structure_metrics": bool(args.structure_metrics),
            "candidate_family": args.candidate_family,
            "hard_bucket_fraction": float(args.hard_bucket_fraction),
            "selection_objective": args.selection_objective,
            "hard_r2_tolerance": float(args.hard_r2_tolerance),
            "candidate_names": None if args.candidate_names is None else list(args.candidate_names),
        },
        "candidate_count": len(candidates),
        "split_diagnostics": split_diagnostics,
        "split_results": split_results,
        "aggregated_results": aggregated,
        "selected_by_multi_dev": selected,
        "details": details,
        "run_time_seconds": float(time.perf_counter() - run_start),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    _write_csv(args.output.with_suffix(".csv"), aggregated)
    _write_markdown(args.output.with_suffix(".md"), payload)
    print(f"Saved cached residual sweep to {args.output}")
    print(
        "Selected {candidate}: mean_dev_r2={mean_dev_r2:.4f}, worst_dev_r2={worst_dev_r2:.4f}".format(
            **selected
        )
    )


if __name__ == "__main__":
    main()
