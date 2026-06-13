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
    sensor_decoder: str = "lstsq"
    decoder_ridge_lambda: float = 1e-4
    decoder_l1_lambda: float = 1e-3
    decoder_max_iter: int = 25
    decoder_tol: float = 1e-6
    correction_alpha: float = 1e-8
    correction_scale: float = 1.0
    correction_residual_rank: int | None = None
    correction_residual_weighting: str = "uniform"
    correction_residual_weight_floor: float = 0.1
    correction_head_type: str = "ridge"
    correction_hidden_size: int = 128
    correction_num_epochs: int = 120
    correction_batch_size: int = 256
    correction_learning_rate: float = 1e-3
    correction_weight_decay: float = 1e-6

    def config(self, *, max_train_segments: int | None, random_state: int) -> FastWindowedTBMDQRCSConfig:
        return FastWindowedTBMDQRCSConfig(
            history_length=self.history_length,
            ranks=[self.r_tau, self.r_x, self.r_y, self.r_segment],
            n_spatial_sensors=self.n_spatial_sensors,
            max_train_segments=max_train_segments,
            correction_alpha=self.correction_alpha,
            correction_scale=self.correction_scale,
            correction_residual_rank=self.correction_residual_rank,
            correction_residual_weighting=self.correction_residual_weighting,
            correction_residual_weight_floor=self.correction_residual_weight_floor,
            correction_head_type=self.correction_head_type,
            correction_hidden_size=self.correction_hidden_size,
            correction_num_epochs=self.correction_num_epochs,
            correction_batch_size=self.correction_batch_size,
            correction_learning_rate=self.correction_learning_rate,
            correction_weight_decay=self.correction_weight_decay,
            sensor_rcond=1e-6,
            sensor_decoder=self.sensor_decoder,
            decoder_ridge_lambda=self.decoder_ridge_lambda,
            decoder_l1_lambda=self.decoder_l1_lambda,
            decoder_max_iter=self.decoder_max_iter,
            decoder_tol=self.decoder_tol,
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
        FastTPlus1Candidate(
            name="mlp_residual_svd128_h64_e80_r300_s600",
            groups=("mlp_head",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "small nonlinear head on residual-SVD codes"},
            correction_residual_rank=128,
            correction_head_type="mlp_residual_svd",
            correction_hidden_size=64,
            correction_num_epochs=80,
            correction_batch_size=256,
            correction_learning_rate=1e-3,
            correction_weight_decay=1e-6,
        ),
        FastTPlus1Candidate(
            name="mlp_residual_svd128_h128_e80_r300_s600",
            groups=("mlp_head",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "wider nonlinear head on residual-SVD codes"},
            correction_residual_rank=128,
            correction_head_type="mlp_residual_svd",
            correction_hidden_size=128,
            correction_num_epochs=80,
            correction_batch_size=256,
            correction_learning_rate=1e-3,
            correction_weight_decay=1e-6,
        ),
        FastTPlus1Candidate(
            name="mlp_residual_svd256_h128_e80_r300_s600",
            groups=("mlp_head",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "nonlinear head with higher residual rank"},
            correction_residual_rank=256,
            correction_head_type="mlp_residual_svd",
            correction_hidden_size=128,
            correction_num_epochs=80,
            correction_batch_size=256,
            correction_learning_rate=1e-3,
            correction_weight_decay=1e-6,
        ),
        FastTPlus1Candidate(
            name="quality_s600_scale0.8",
            groups=("correction_scale",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "dampen full residual correction"},
            correction_scale=0.8,
        ),
        FastTPlus1Candidate(
            name="quality_s600_scale0.9",
            groups=("correction_scale",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "slightly dampen full residual correction"},
            correction_scale=0.9,
        ),
        FastTPlus1Candidate(
            name="quality_s600_scale1.1",
            groups=("correction_scale",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "slightly amplify full residual correction"},
            correction_scale=1.1,
        ),
        FastTPlus1Candidate(
            name="residual_svd256_scale0.8_r300_s600",
            groups=("correction_scale",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "dampen residual-SVD correction"},
            correction_residual_rank=256,
            correction_scale=0.8,
        ),
        FastTPlus1Candidate(
            name="residual_svd256_scale0.9_r300_s600",
            groups=("correction_scale",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "slightly dampen residual-SVD correction"},
            correction_residual_rank=256,
            correction_scale=0.9,
        ),
        FastTPlus1Candidate(
            name="residual_svd256_scale1.1_r300_s600",
            groups=("correction_scale",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "slightly amplify residual-SVD correction"},
            correction_residual_rank=256,
            correction_scale=1.1,
        ),
        FastTPlus1Candidate(
            name="residual_svd256_scale1.1_ridge_lam1e-4_r300_s300",
            groups=("decoder_recovery",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=300,
            notes={"hypothesis": "production-scale precomputed ridge decoder with residual-SVD head"},
            sensor_decoder="ridge",
            decoder_ridge_lambda=1e-4,
            correction_residual_rank=256,
            correction_scale=1.1,
        ),
        FastTPlus1Candidate(
            name="residual_svd256_scale1.1_fista_i25_r300_s300",
            groups=("decoder_recovery",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=300,
            notes={"hypothesis": "production-scale FISTA decoder with residual-SVD head"},
            sensor_decoder="fista",
            decoder_l1_lambda=1e-3,
            decoder_max_iter=25,
            decoder_tol=1e-6,
            correction_residual_rank=256,
            correction_scale=1.1,
        ),
        FastTPlus1Candidate(
            name="residual_svd256_scale1.1_ridge_lam1e-4_r300_s600",
            groups=("decoder_recovery",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "quality-scale precomputed ridge decoder with residual-SVD head"},
            sensor_decoder="ridge",
            decoder_ridge_lambda=1e-4,
            correction_residual_rank=256,
            correction_scale=1.1,
        ),
        FastTPlus1Candidate(
            name="residual_svd256_scale1.1_fista_i25_r300_s600",
            groups=("decoder_recovery",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "quality-scale FISTA decoder with residual-SVD head"},
            sensor_decoder="fista",
            decoder_l1_lambda=1e-3,
            decoder_max_iter=25,
            decoder_tol=1e-6,
            correction_residual_rank=256,
            correction_scale=1.1,
        ),
        FastTPlus1Candidate(
            name="decoder_reg_ridge_lam1e-8_r300_s600",
            groups=("decoder_regularization",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "near-lstsq ridge decoder at production scale"},
            sensor_decoder="ridge",
            decoder_ridge_lambda=1e-8,
            correction_residual_rank=256,
            correction_scale=1.1,
        ),
        FastTPlus1Candidate(
            name="decoder_reg_ridge_lam1e-2_r300_s600",
            groups=("decoder_regularization",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "moderate ridge regularization improves recovery stability"},
            sensor_decoder="ridge",
            decoder_ridge_lambda=1e-2,
            correction_residual_rank=256,
            correction_scale=1.1,
        ),
        FastTPlus1Candidate(
            name="decoder_reg_ridge_lam1_r300_s600",
            groups=("decoder_regularization",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "strong ridge regularization may reduce coefficient noise"},
            sensor_decoder="ridge",
            decoder_ridge_lambda=1.0,
            correction_residual_rank=256,
            correction_scale=1.1,
        ),
        FastTPlus1Candidate(
            name="decoder_reg_fista_l1_1e-4_i50_r300_s600",
            groups=("decoder_regularization",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "lower L1 and more FISTA iterations recover denser coefficients"},
            sensor_decoder="fista",
            decoder_l1_lambda=1e-4,
            decoder_max_iter=50,
            decoder_tol=1e-6,
            correction_residual_rank=256,
            correction_scale=1.1,
        ),
        FastTPlus1Candidate(
            name="decoder_reg_fista_l1_3e-4_i50_r300_s600",
            groups=("decoder_regularization",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "middle L1 tests sparse/biased recovery trade-off"},
            sensor_decoder="fista",
            decoder_l1_lambda=3e-4,
            decoder_max_iter=50,
            decoder_tol=1e-6,
            correction_residual_rank=256,
            correction_scale=1.1,
        ),
        FastTPlus1Candidate(
            name="decoder_reg_fista_l1_1e-3_i50_r300_s600",
            groups=("decoder_regularization",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "same L1 as previous FISTA but longer optimization"},
            sensor_decoder="fista",
            decoder_l1_lambda=1e-3,
            decoder_max_iter=50,
            decoder_tol=1e-6,
            correction_residual_rank=256,
            correction_scale=1.1,
        ),
        FastTPlus1Candidate(
            name="sensor_overbudget_r300_s800_residual_svd256_scale1.1",
            groups=("sensor_overbudget",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=800,
            notes={"hypothesis": "extra QR+leverage sensors reduce coefficient recovery error"},
            sensor_decoder="ridge",
            decoder_ridge_lambda=1e-8,
            correction_residual_rank=256,
            correction_scale=1.1,
        ),
        FastTPlus1Candidate(
            name="sensor_overbudget_r300_s1000_residual_svd256_scale1.1",
            groups=("sensor_overbudget",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=1000,
            notes={"hypothesis": "larger fixed spatial sensor budget may cross dev 0.78"},
            sensor_decoder="ridge",
            decoder_ridge_lambda=1e-8,
            correction_residual_rank=256,
            correction_scale=1.1,
        ),
        FastTPlus1Candidate(
            name="sensor_overbudget_r300_s1200_residual_svd256_scale1.1",
            groups=("sensor_overbudget_upper",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=1200,
            notes={"hypothesis": "upper sensor budget continues the s600->s1000 trend"},
            sensor_decoder="ridge",
            decoder_ridge_lambda=1e-8,
            correction_residual_rank=256,
            correction_scale=1.1,
        ),
        FastTPlus1Candidate(
            name="sensor_overbudget_r300_s1500_residual_svd256_scale1.1",
            groups=("sensor_overbudget_upper",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=1500,
            notes={"hypothesis": "large sensor budget tests whether quality saturates before 0.85"},
            sensor_decoder="ridge",
            decoder_ridge_lambda=1e-8,
            correction_residual_rank=256,
            correction_scale=1.1,
        ),
        FastTPlus1Candidate(
            name="s1000_residual_svd256_scale1.0",
            groups=("sensor_overbudget_scale",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=1000,
            notes={"hypothesis": "unscaled residual-SVD with sensor overbudget"},
            sensor_decoder="ridge",
            decoder_ridge_lambda=1e-8,
            correction_residual_rank=256,
            correction_scale=1.0,
        ),
        FastTPlus1Candidate(
            name="s1000_residual_svd256_scale1.05",
            groups=("sensor_overbudget_scale",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=1000,
            notes={"hypothesis": "slightly below selected scale may generalize better"},
            sensor_decoder="ridge",
            decoder_ridge_lambda=1e-8,
            correction_residual_rank=256,
            correction_scale=1.05,
        ),
        FastTPlus1Candidate(
            name="s1000_residual_svd256_scale1.15",
            groups=("sensor_overbudget_scale",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=1000,
            notes={"hypothesis": "slightly above selected scale"},
            sensor_decoder="ridge",
            decoder_ridge_lambda=1e-8,
            correction_residual_rank=256,
            correction_scale=1.15,
        ),
        FastTPlus1Candidate(
            name="s1000_residual_svd256_scale1.2",
            groups=("sensor_overbudget_scale",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=1000,
            notes={"hypothesis": "moderate residual amplification under larger sensor budget"},
            sensor_decoder="ridge",
            decoder_ridge_lambda=1e-8,
            correction_residual_rank=256,
            correction_scale=1.2,
        ),
        FastTPlus1Candidate(
            name="s1000_residual_svd256_scale1.25",
            groups=("sensor_overbudget_scale",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=1000,
            notes={"hypothesis": "upper local residual amplification under larger sensor budget"},
            sensor_decoder="ridge",
            decoder_ridge_lambda=1e-8,
            correction_residual_rank=256,
            correction_scale=1.25,
        ),
        FastTPlus1Candidate(
            name="s1000_residual_svd256_scale1.3",
            groups=("sensor_overbudget_scale_upper",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=1000,
            notes={"hypothesis": "upper local residual scale after scale1.25 dev gain"},
            sensor_decoder="ridge",
            decoder_ridge_lambda=1e-8,
            correction_residual_rank=256,
            correction_scale=1.3,
        ),
        FastTPlus1Candidate(
            name="s1000_residual_svd256_scale1.4",
            groups=("sensor_overbudget_scale_upper",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=1000,
            notes={"hypothesis": "test whether sensor-overbudget tolerates stronger residual scale"},
            sensor_decoder="ridge",
            decoder_ridge_lambda=1e-8,
            correction_residual_rank=256,
            correction_scale=1.4,
        ),
        FastTPlus1Candidate(
            name="s1000_residual_svd256_scale1.5",
            groups=("sensor_overbudget_scale_upper",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=1000,
            notes={"hypothesis": "upper bound before high-scale overfit risk"},
            sensor_decoder="ridge",
            decoder_ridge_lambda=1e-8,
            correction_residual_rank=256,
            correction_scale=1.5,
        ),
        FastTPlus1Candidate(
            name="residual_energy_svd128_r300_s600",
            groups=("weighted_residual",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "error-sensitive residual basis with rank 128"},
            correction_residual_rank=128,
            correction_residual_weighting="residual_energy",
        ),
        FastTPlus1Candidate(
            name="residual_energy_svd256_r300_s600",
            groups=("weighted_residual",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "error-sensitive residual basis with rank 256"},
            correction_residual_rank=256,
            correction_residual_weighting="residual_energy",
        ),
        FastTPlus1Candidate(
            name="residual_energy_svd128_scale1.1_r300_s600",
            groups=("weighted_residual",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "scaled error-sensitive residual basis with rank 128"},
            correction_residual_rank=128,
            correction_residual_weighting="residual_energy",
            correction_scale=1.1,
        ),
        FastTPlus1Candidate(
            name="residual_energy_svd256_scale1.1_r300_s600",
            groups=("weighted_residual",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "scaled error-sensitive residual basis with rank 256"},
            correction_residual_rank=256,
            correction_residual_weighting="residual_energy",
            correction_scale=1.1,
        ),
        FastTPlus1Candidate(
            name="residual_svd256_scale1.05_r300_s600",
            groups=("correction_scale_fine",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "fine scale sweep around residual-SVD correction"},
            correction_residual_rank=256,
            correction_scale=1.05,
        ),
        FastTPlus1Candidate(
            name="residual_svd256_scale1.15_r300_s600",
            groups=("correction_scale_fine",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "fine scale sweep above previous best scale"},
            correction_residual_rank=256,
            correction_scale=1.15,
        ),
        FastTPlus1Candidate(
            name="residual_svd256_scale1.2_r300_s600",
            groups=("correction_scale_fine",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "larger residual amplification"},
            correction_residual_rank=256,
            correction_scale=1.2,
        ),
        FastTPlus1Candidate(
            name="residual_svd256_scale1.25_r300_s600",
            groups=("correction_scale_fine",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "upper fine residual amplification"},
            correction_residual_rank=256,
            correction_scale=1.25,
        ),
        FastTPlus1Candidate(
            name="residual_svd256_scale1.3_r300_s600",
            groups=("correction_scale_upper",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "upper residual amplification sweep"},
            correction_residual_rank=256,
            correction_scale=1.3,
        ),
        FastTPlus1Candidate(
            name="residual_svd256_scale1.4_r300_s600",
            groups=("correction_scale_upper",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "upper residual amplification sweep"},
            correction_residual_rank=256,
            correction_scale=1.4,
        ),
        FastTPlus1Candidate(
            name="residual_svd256_scale1.5_r300_s600",
            groups=("correction_scale_upper",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "upper residual amplification sweep"},
            correction_residual_rank=256,
            correction_scale=1.5,
        ),
        FastTPlus1Candidate(
            name="residual_svd256_scale1.6_r300_s600",
            groups=("correction_scale_upper",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "upper residual amplification sweep"},
            correction_residual_rank=256,
            correction_scale=1.6,
        ),
        FastTPlus1Candidate(
            name="residual_svd256_scale1.8_r300_s600",
            groups=("correction_scale_high",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "high residual amplification sweep"},
            correction_residual_rank=256,
            correction_scale=1.8,
        ),
        FastTPlus1Candidate(
            name="residual_svd256_scale2.0_r300_s600",
            groups=("correction_scale_high",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "high residual amplification sweep"},
            correction_residual_rank=256,
            correction_scale=2.0,
        ),
        FastTPlus1Candidate(
            name="residual_svd256_scale2.25_r300_s600",
            groups=("correction_scale_high",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "high residual amplification sweep"},
            correction_residual_rank=256,
            correction_scale=2.25,
        ),
        FastTPlus1Candidate(
            name="residual_svd256_scale2.5_r300_s600",
            groups=("correction_scale_high",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "high residual amplification sweep"},
            correction_residual_rank=256,
            correction_scale=2.5,
        ),
        FastTPlus1Candidate(
            name="residual_svd256_scale1.55_r300_s600",
            groups=("correction_scale_peak",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "peak residual amplification sweep"},
            correction_residual_rank=256,
            correction_scale=1.55,
        ),
        FastTPlus1Candidate(
            name="residual_svd256_scale1.65_r300_s600",
            groups=("correction_scale_peak",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "peak residual amplification sweep"},
            correction_residual_rank=256,
            correction_scale=1.65,
        ),
        FastTPlus1Candidate(
            name="residual_svd256_scale1.7_r300_s600",
            groups=("correction_scale_peak",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "peak residual amplification sweep"},
            correction_residual_rank=256,
            correction_scale=1.7,
        ),
        FastTPlus1Candidate(
            name="residual_svd256_scale1.75_r300_s600",
            groups=("correction_scale_peak",),
            history_length=7,
            r_tau=8,
            r_x=32,
            r_y=32,
            r_segment=300,
            n_spatial_sensors=600,
            notes={"hypothesis": "peak residual amplification sweep"},
            correction_residual_rank=256,
            correction_scale=1.75,
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
            "sensor_decoder": candidate.sensor_decoder,
            "decoder_ridge_lambda": candidate.decoder_ridge_lambda,
            "decoder_l1_lambda": candidate.decoder_l1_lambda,
            "decoder_max_iter": candidate.decoder_max_iter,
            "correction_alpha": candidate.correction_alpha,
            "correction_scale": candidate.correction_scale,
            "correction_residual_rank": candidate.correction_residual_rank,
            "correction_residual_weighting": candidate.correction_residual_weighting,
            "correction_head_type": candidate.correction_head_type,
            "correction_hidden_size": candidate.correction_hidden_size,
            "correction_num_epochs": candidate.correction_num_epochs,
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
