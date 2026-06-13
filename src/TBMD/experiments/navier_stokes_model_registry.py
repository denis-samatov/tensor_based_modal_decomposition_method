"""
Shared Navier-Stokes experiment model registry.

This module keeps final benchmark/example model definitions in one place so
quantitative reports and qualitative artifacts cannot silently drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from TBMD.config import LatentModalForecasterConfig, MultiResolutionTBMDConfig
from TBMD.experiments.navier_stokes_fast_tplus1 import (
    FastWindowedTBMDQRCSConfig,
    FastWindowedTBMDQRCSForecaster,
)
from TBMD.experiments.navier_stokes_forecasting import (
    TrajectoryAwareDMDForecaster,
    TrajectoryAwareLatentForecaster,
    TrajectoryAwareMultiResolutionForecaster,
    TrajectoryAwarePersistenceForecaster,
    TrajectoryAwareResidualCorrectedForecaster,
)

DEFAULT_NAVIER_STOKES_RANKS = [64, 64, 15]
DEFAULT_DMD_RANK = 20
DEFAULT_COMMON_WARMUP_STEPS = 7
DEFAULT_N_TRAIN_TRAJECTORIES = 1000
DEFAULT_FAST_TPLUS1_RANKS = [8, 32, 32, 300]


@dataclass(frozen=True)
class NavierStokesModelSpec:
    """Frozen benchmark/example model definition."""

    name: str
    slug: str
    factory: Callable[[], Any]
    family: str = ""
    notes: dict[str, Any] = field(default_factory=dict)


def get_navier_stokes_model_specs() -> list[NavierStokesModelSpec]:
    """Return the shared Navier-Stokes benchmark model set."""

    return [
        NavierStokesModelSpec(
            name="Persistence Forecaster",
            slug="persistence_forecaster",
            family="baseline",
            factory=lambda: TrajectoryAwarePersistenceForecaster(),
            notes={
                "baseline": "x_t",
            },
        ),
        NavierStokesModelSpec(
            name="DMD Forecaster",
            slug="dmd_forecaster",
            family="dmd",
            factory=lambda: TrajectoryAwareDMDForecaster(rank=DEFAULT_DMD_RANK),
            notes={
                "rank": DEFAULT_DMD_RANK,
                "spatial_mean_centering": True,
            },
        ),
        NavierStokesModelSpec(
            name="Linear Forecaster",
            slug="linear_forecaster",
            family="linear",
            factory=lambda: TrajectoryAwareLatentForecaster(
                config=LatentModalForecasterConfig(
                    ranks=DEFAULT_NAVIER_STOKES_RANKS,
                    forecaster_type="linear",
                    verbose=False,
                )
            ),
        ),
        NavierStokesModelSpec(
            name="MLP Forecaster",
            slug="mlp_forecaster",
            family="mlp",
            factory=lambda: TrajectoryAwareLatentForecaster(
                config=LatentModalForecasterConfig(
                    ranks=DEFAULT_NAVIER_STOKES_RANKS,
                    forecaster_type="mlp",
                    delta_forecast=False,
                    verbose=False,
                    mlp_hidden_size=128,
                    mlp_num_layers=2,
                    mlp_num_epochs=150,
                )
            ),
            notes={
                "delta_forecast": False,
                "selected_preset": "mlp_h128_l2_e150",
            },
        ),
        NavierStokesModelSpec(
            name="LSTM Forecaster",
            slug="lstm_forecaster",
            family="lstm",
            factory=lambda: TrajectoryAwareLatentForecaster(
                config=LatentModalForecasterConfig(
                    ranks=DEFAULT_NAVIER_STOKES_RANKS,
                    forecaster_type="lstm",
                    verbose=False,
                    delta_forecast=False,
                    lstm_hidden_size=128,
                    lstm_num_layers=2,
                    lstm_seq_length=10,
                    lstm_num_epochs=150,
                    lstm_use_scheduled_sampling=True,
                    lstm_ss_unroll_steps=5,
                    lstm_ss_decay_rate=0.01,
                ),
                feature_mode="latent_plus_delta",
            ),
            notes={
                "selected_preset": "lstm_h128_l2_s10_e150_plus_delta_features_ss_u5",
                "feature_mode": "latent_plus_delta",
                "delta_forecast": False,
                "scheduled_sampling": True,
            },
        ),
        NavierStokesModelSpec(
            name="LSTM + T+1 Residual Corrector",
            slug="lstm_t_plus_1_residual_corrected",
            family="lstm_residual_corrected",
            factory=lambda: TrajectoryAwareResidualCorrectedForecaster(
                config=LatentModalForecasterConfig(
                    ranks=DEFAULT_NAVIER_STOKES_RANKS,
                    forecaster_type="lstm",
                    verbose=False,
                    delta_forecast=False,
                    lstm_hidden_size=128,
                    lstm_num_layers=2,
                    lstm_seq_length=10,
                    lstm_num_epochs=150,
                    lstm_use_scheduled_sampling=True,
                    lstm_ss_unroll_steps=5,
                    lstm_ss_decay_rate=0.01,
                ),
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
                correction_spatial_loss_weight=0.0,
                correction_rel_frob_loss_weight=0.0,
            ),
            notes={
                "selected_preset": "lstm_h128_l2_s10_e150_plus_delta_features_ss_u5_with_t_plus_1_residual_head",
                "feature_mode": "latent_plus_delta",
                "delta_forecast": False,
                "scheduled_sampling": True,
                "correction_head": "mlp_h64_l2_e120",
                "priority_metric": "one_step",
                "correction_loss": {
                    "latent": 1.0,
                    "spatial": 0.0,
                    "rel_frob": 0.0,
                },
                "mixed_one_step_loss_available": True,
            },
        ),
        NavierStokesModelSpec(
            name="Multi-Resolution Linear",
            slug="multi_resolution_linear",
            family="multi_resolution_linear",
            factory=lambda: TrajectoryAwareMultiResolutionForecaster(
                config=MultiResolutionTBMDConfig(
                    level_ranks=[[64, 64, 5], [64, 64, 15]],
                    level_forecaster_types=["linear", "linear"],
                    verbose=False,
                )
            ),
        ),
    ]


def get_fast_tplus1_model_specs() -> list[NavierStokesModelSpec]:
    """Return final Stage 5 fast one-step TBMD+QR+CS model presets."""

    return [
        NavierStokesModelSpec(
            name="Fast TBMD+QR+CS T+1 Practical",
            slug="fast_tplus1_r300_s300",
            family="fast_tbmd_qr_cs_tplus1",
            factory=lambda: FastWindowedTBMDQRCSForecaster(
                FastWindowedTBMDQRCSConfig(
                    history_length=7,
                    ranks=DEFAULT_FAST_TPLUS1_RANKS,
                    n_spatial_sensors=300,
                    max_train_segments=6144,
                    correction_alpha=1e-8,
                    sensor_rcond=1e-6,
                    random_state=0,
                )
            ),
            notes={
                "label": "practical",
                "purpose": "fast practical t+1 predictor",
                "method": "windowed TBMD/HOSVD + fixed-spatial QR sensors + gappy coefficient recovery + coefficient-ridge correction",
                "rank": 300,
                "sensors": 300,
                "history_length": 7,
                "history_measurements": 2100,
            },
        ),
        NavierStokesModelSpec(
            name="Fast TBMD+QR+CS T+1 Quality-Max",
            slug="fast_tplus1_r300_s600",
            family="fast_tbmd_qr_cs_tplus1",
            factory=lambda: FastWindowedTBMDQRCSForecaster(
                FastWindowedTBMDQRCSConfig(
                    history_length=7,
                    ranks=DEFAULT_FAST_TPLUS1_RANKS,
                    n_spatial_sensors=600,
                    max_train_segments=6144,
                    correction_alpha=1e-8,
                    sensor_rcond=1e-6,
                    random_state=0,
                )
            ),
            notes={
                "label": "quality-max",
                "purpose": "higher-quality t+1 predictor",
                "method": "windowed TBMD/HOSVD + fixed-spatial QR sensors + gappy coefficient recovery + coefficient-ridge correction",
                "rank": 300,
                "sensors": 600,
                "history_length": 7,
                "history_measurements": 4200,
            },
        ),
        NavierStokesModelSpec(
            name="Fast TBMD+QR+CS T+1 Residual-SVD Candidate",
            slug="fast_tplus1_r300_s600_residual_svd256",
            family="fast_tbmd_qr_cs_tplus1",
            factory=lambda: FastWindowedTBMDQRCSForecaster(
                FastWindowedTBMDQRCSConfig(
                    history_length=7,
                    ranks=DEFAULT_FAST_TPLUS1_RANKS,
                    n_spatial_sensors=600,
                    max_train_segments=6144,
                    correction_alpha=1e-8,
                    correction_residual_rank=256,
                    sensor_rcond=1e-6,
                    random_state=0,
                )
            ),
            notes={
                "label": "residual-svd-dev-candidate",
                "purpose": "dev-selected low-rank residual correction candidate for t+1 accuracy",
                "method": "windowed TBMD/HOSVD + fixed-spatial QR sensors + gappy coefficient recovery + residual-SVD coefficient-ridge correction",
                "rank": 300,
                "sensors": 600,
                "history_length": 7,
                "history_measurements": 4200,
                "correction_residual_rank": 256,
                "selection_source": "stage5_fast_tplus1_residual_head_fine_train640 dev sweep",
            },
        ),
        NavierStokesModelSpec(
            name="Fast TBMD+QR+CS T+1 Residual-SVD Scale Candidate",
            slug="fast_tplus1_r300_s600_residual_svd256_scale11",
            family="fast_tbmd_qr_cs_tplus1",
            factory=lambda: FastWindowedTBMDQRCSForecaster(
                FastWindowedTBMDQRCSConfig(
                    history_length=7,
                    ranks=DEFAULT_FAST_TPLUS1_RANKS,
                    n_spatial_sensors=600,
                    max_train_segments=6144,
                    correction_alpha=1e-8,
                    correction_scale=1.1,
                    correction_residual_rank=256,
                    sensor_rcond=1e-6,
                    random_state=0,
                )
            ),
            notes={
                "label": "residual-svd-scale-dev-candidate",
                "purpose": "dev-selected scaled residual-SVD correction candidate for t+1 accuracy",
                "method": "windowed TBMD/HOSVD + fixed-spatial QR sensors + gappy coefficient recovery + scaled residual-SVD coefficient-ridge correction",
                "rank": 300,
                "sensors": 600,
                "history_length": 7,
                "history_measurements": 4200,
                "correction_residual_rank": 256,
                "correction_scale": 1.1,
                "selection_source": "stage5_fast_tplus1_correction_scale_train640 dev sweep",
            },
        ),
        NavierStokesModelSpec(
            name="Fast TBMD+QR+CS T+1 Residual-SVD Peak-Scale Candidate",
            slug="fast_tplus1_r300_s600_residual_svd256_scale165",
            family="fast_tbmd_qr_cs_tplus1",
            factory=lambda: FastWindowedTBMDQRCSForecaster(
                FastWindowedTBMDQRCSConfig(
                    history_length=7,
                    ranks=DEFAULT_FAST_TPLUS1_RANKS,
                    n_spatial_sensors=600,
                    max_train_segments=6144,
                    correction_alpha=1e-8,
                    correction_scale=1.65,
                    correction_residual_rank=256,
                    sensor_rcond=1e-6,
                    random_state=0,
                )
            ),
            notes={
                "label": "residual-svd-scale-peak-dev-candidate",
                "purpose": "dev-selected peak scaled residual-SVD correction candidate for t+1 accuracy",
                "method": "windowed TBMD/HOSVD + fixed-spatial QR sensors + gappy coefficient recovery + peak-scaled residual-SVD coefficient-ridge correction",
                "rank": 300,
                "sensors": 600,
                "history_length": 7,
                "history_measurements": 4200,
                "correction_residual_rank": 256,
                "correction_scale": 1.65,
                "selection_source": "stage5_fast_tplus1_correction_scale_peak_train640 dev sweep",
            },
        ),
        NavierStokesModelSpec(
            name="Fast TBMD+QR+CS T+1 Sensor-Overbudget Candidate",
            slug="fast_tplus1_r300_s1000_residual_svd256_scale11",
            family="fast_tbmd_qr_cs_tplus1",
            factory=lambda: FastWindowedTBMDQRCSForecaster(
                FastWindowedTBMDQRCSConfig(
                    history_length=7,
                    ranks=DEFAULT_FAST_TPLUS1_RANKS,
                    n_spatial_sensors=1000,
                    max_train_segments=6144,
                    sensor_decoder="ridge",
                    decoder_ridge_lambda=1e-8,
                    correction_alpha=1e-8,
                    correction_scale=1.1,
                    correction_residual_rank=256,
                    sensor_rcond=1e-6,
                    random_state=0,
                )
            ),
            notes={
                "label": "sensor-overbudget-dev-candidate",
                "purpose": "dev-selected higher-sensor t+1 quality candidate",
                "method": "windowed TBMD/HOSVD + fixed-spatial QR/leverage sensors + precomputed ridge recovery + scaled residual-SVD correction",
                "rank": 300,
                "sensors": 1000,
                "history_length": 7,
                "history_measurements": 7000,
                "correction_residual_rank": 256,
                "correction_scale": 1.1,
                "sensor_decoder": "ridge",
                "decoder_ridge_lambda": 1e-8,
                "selection_source": "stage5_fast_tplus1_sensor_overbudget_train640 dev sweep",
            },
        ),
        NavierStokesModelSpec(
            name="Fast TBMD+QR+CS T+1 Sensor-Overbudget Scale Candidate",
            slug="fast_tplus1_r300_s1000_residual_svd256_scale13",
            family="fast_tbmd_qr_cs_tplus1",
            factory=lambda: FastWindowedTBMDQRCSForecaster(
                FastWindowedTBMDQRCSConfig(
                    history_length=7,
                    ranks=DEFAULT_FAST_TPLUS1_RANKS,
                    n_spatial_sensors=1000,
                    max_train_segments=6144,
                    sensor_decoder="ridge",
                    decoder_ridge_lambda=1e-8,
                    correction_alpha=1e-8,
                    correction_scale=1.3,
                    correction_residual_rank=256,
                    sensor_rcond=1e-6,
                    random_state=0,
                )
            ),
            notes={
                "label": "sensor-overbudget-cached-multidev-candidate",
                "purpose": "multi-dev-selected higher-sensor scaled-residual t+1 quality candidate",
                "method": "windowed TBMD/HOSVD + fixed-spatial QR/leverage sensors + precomputed ridge recovery + scaled residual-SVD correction",
                "rank": 300,
                "sensors": 1000,
                "history_length": 7,
                "history_measurements": 7000,
                "correction_residual_rank": 256,
                "correction_scale": 1.3,
                "sensor_decoder": "ridge",
                "decoder_ridge_lambda": 1e-8,
                "selection_source": "stage5_fast_tplus1_cached_residual_fast multi-dev train-only sweep",
            },
        ),
        NavierStokesModelSpec(
            name="Fast TBMD+QR+CS T+1 Patch Residual Candidate",
            slug="fast_tplus1_r300_s1000_patch16_svd32_scale13",
            family="fast_tbmd_qr_cs_tplus1",
            factory=lambda: FastWindowedTBMDQRCSForecaster(
                FastWindowedTBMDQRCSConfig(
                    history_length=7,
                    ranks=DEFAULT_FAST_TPLUS1_RANKS,
                    n_spatial_sensors=1000,
                    max_train_segments=6144,
                    sensor_decoder="ridge",
                    decoder_ridge_lambda=1e-8,
                    correction_head_type="patch_residual_svd",
                    correction_alpha=1e-8,
                    correction_scale=1.3,
                    correction_patch_size=16,
                    correction_patch_residual_rank=32,
                    sensor_rcond=1e-6,
                    random_state=0,
                )
            ),
            notes={
                "label": "sensor-overbudget-patch-residual-multidev-candidate",
                "purpose": "multi-dev-selected local patch residual t+1 quality candidate",
                "method": "windowed TBMD/HOSVD + fixed-spatial QR/leverage sensors + precomputed ridge recovery + local patch residual-SVD correction",
                "rank": 300,
                "sensors": 1000,
                "history_length": 7,
                "history_measurements": 7000,
                "correction_head_type": "patch_residual_svd",
                "correction_patch_size": 16,
                "correction_patch_residual_rank": 32,
                "correction_scale": 1.3,
                "sensor_decoder": "ridge",
                "decoder_ridge_lambda": 1e-8,
                "selection_source": "stage5_fast_tplus1_cached_patch_residual_refine_fast multi-dev train-only sweep",
            },
        ),
        NavierStokesModelSpec(
            name="Fast TBMD+QR+CS T+1 Temporal Patch Residual Candidate",
            slug="fast_tplus1_r300_s1000_tempsmooth_patch16_svd32_a01_scale13",
            family="fast_tbmd_qr_cs_tplus1",
            factory=lambda: FastWindowedTBMDQRCSForecaster(
                FastWindowedTBMDQRCSConfig(
                    history_length=7,
                    ranks=DEFAULT_FAST_TPLUS1_RANKS,
                    n_spatial_sensors=1000,
                    max_train_segments=6144,
                    sensor_decoder="ridge",
                    decoder_ridge_lambda=1e-8,
                    coefficient_temporal_smoothing_alpha=0.1,
                    coefficient_temporal_reset_on_gap=True,
                    correction_head_type="patch_residual_svd",
                    correction_alpha=1e-8,
                    correction_scale=1.3,
                    correction_patch_size=16,
                    correction_patch_residual_rank=32,
                    sensor_rcond=1e-6,
                    random_state=0,
                )
            ),
            notes={
                "label": "sensor-overbudget-temporal-patch-residual-multidev-candidate",
                "purpose": "train/dev-selected causal coefficient smoothing plus local patch residual t+1 candidate",
                "method": "windowed TBMD/HOSVD + fixed-spatial QR/leverage sensors + precomputed ridge recovery + causal coefficient smoothing + local patch residual-SVD correction",
                "rank": 300,
                "sensors": 1000,
                "history_length": 7,
                "history_measurements": 7000,
                "coefficient_temporal_smoothing_alpha": 0.1,
                "correction_head_type": "patch_residual_svd",
                "correction_patch_size": 16,
                "correction_patch_residual_rank": 32,
                "correction_scale": 1.3,
                "sensor_decoder": "ridge",
                "decoder_ridge_lambda": 1e-8,
                "selection_source": "stage5_fast_tplus1_temporal_patch_fast_train640 multi-dev train-only sweep",
            },
        ),
        NavierStokesModelSpec(
            name="Fast TBMD+QR+CS T+1 Composite Patch+HF Candidate",
            slug="fast_tplus1_r300_s1000_composite_patch24_hf256_scale13_hf04",
            family="fast_tbmd_qr_cs_tplus1",
            factory=lambda: FastWindowedTBMDQRCSForecaster(
                FastWindowedTBMDQRCSConfig(
                    history_length=7,
                    ranks=DEFAULT_FAST_TPLUS1_RANKS,
                    n_spatial_sensors=1000,
                    max_train_segments=6144,
                    sensor_decoder="ridge",
                    decoder_ridge_lambda=1e-8,
                    correction_head_type="composite_patch_hf_svd",
                    correction_alpha=1e-8,
                    correction_scale=1.3,
                    correction_hf_scale=0.4,
                    correction_patch_size=16,
                    correction_patch_residual_rank=24,
                    correction_residual_rank=256,
                    correction_highpass_cutoff_fraction=0.45,
                    sensor_rcond=1e-6,
                    random_state=0,
                )
            ),
            notes={
                "label": "sensor-overbudget-composite-patch-hf-confirmed-candidate",
                "purpose": "train-only confirmed local patch plus high-frequency residual t+1 quality candidate",
                "method": "windowed TBMD/HOSVD + fixed-spatial QR/leverage sensors + precomputed ridge recovery + composite local patch and high-frequency residual correction",
                "rank": 300,
                "sensors": 1000,
                "history_length": 7,
                "history_measurements": 7000,
                "correction_head_type": "composite_patch_hf_svd",
                "correction_patch_size": 16,
                "correction_patch_residual_rank": 24,
                "correction_residual_rank": 256,
                "correction_scale": 1.3,
                "correction_hf_scale": 0.4,
                "correction_highpass_cutoff_fraction": 0.45,
                "sensor_decoder": "ridge",
                "decoder_ridge_lambda": 1e-8,
                "selection_source": "stage5_fast_tplus1_composite_residual_confirm_train640 train-only dev sweep",
            },
        ),
        NavierStokesModelSpec(
            name="Fast TBMD+QR+CS T+1 Composite Patch+HF Seg2048 Candidate",
            slug="fast_tplus1_r300_s1000_composite_patch24_hf256_scale13_hf04_seg2048",
            family="fast_tbmd_qr_cs_tplus1",
            factory=lambda: FastWindowedTBMDQRCSForecaster(
                FastWindowedTBMDQRCSConfig(
                    history_length=7,
                    ranks=DEFAULT_FAST_TPLUS1_RANKS,
                    n_spatial_sensors=1000,
                    max_train_segments=2048,
                    sensor_decoder="ridge",
                    decoder_ridge_lambda=1e-8,
                    correction_head_type="composite_patch_hf_svd",
                    correction_alpha=1e-8,
                    correction_scale=1.3,
                    correction_hf_scale=0.4,
                    correction_patch_size=16,
                    correction_patch_residual_rank=24,
                    correction_residual_rank=256,
                    correction_highpass_cutoff_fraction=0.45,
                    sensor_rcond=1e-6,
                    random_state=0,
                )
            ),
            notes={
                "label": "sensor-overbudget-composite-patch-hf-selected-seg2048-candidate",
                "purpose": "exact train/dev-selected composite candidate with matched segment budget",
                "method": "windowed TBMD/HOSVD + fixed-spatial QR/leverage sensors + precomputed ridge recovery + composite local patch and high-frequency residual correction",
                "rank": 300,
                "sensors": 1000,
                "history_length": 7,
                "history_measurements": 7000,
                "max_train_segments": 2048,
                "correction_head_type": "composite_patch_hf_svd",
                "correction_patch_size": 16,
                "correction_patch_residual_rank": 24,
                "correction_residual_rank": 256,
                "correction_scale": 1.3,
                "correction_hf_scale": 0.4,
                "correction_highpass_cutoff_fraction": 0.45,
                "sensor_decoder": "ridge",
                "decoder_ridge_lambda": 1e-8,
                "selection_source": "stage5_fast_tplus1_targeted_hard_bucket_train1000 train-only dev sweep",
            },
        ),
        NavierStokesModelSpec(
            name="Fast TBMD+QR+CS T+1 Sensor-Overbudget High-Scale Candidate",
            slug="fast_tplus1_r300_s1000_residual_svd256_scale15",
            family="fast_tbmd_qr_cs_tplus1",
            factory=lambda: FastWindowedTBMDQRCSForecaster(
                FastWindowedTBMDQRCSConfig(
                    history_length=7,
                    ranks=DEFAULT_FAST_TPLUS1_RANKS,
                    n_spatial_sensors=1000,
                    max_train_segments=6144,
                    sensor_decoder="ridge",
                    decoder_ridge_lambda=1e-8,
                    correction_alpha=1e-8,
                    correction_scale=1.5,
                    correction_residual_rank=256,
                    sensor_rcond=1e-6,
                    random_state=0,
                )
            ),
            notes={
                "label": "sensor-overbudget-scale-dev-candidate",
                "purpose": "dev-confirmed higher-sensor scaled-residual t+1 quality candidate",
                "method": "windowed TBMD/HOSVD + fixed-spatial QR/leverage sensors + precomputed ridge recovery + scaled residual-SVD correction",
                "rank": 300,
                "sensors": 1000,
                "history_length": 7,
                "history_measurements": 7000,
                "correction_residual_rank": 256,
                "correction_scale": 1.5,
                "sensor_decoder": "ridge",
                "decoder_ridge_lambda": 1e-8,
                "selection_source": "stage5_fast_tplus1_sensor_overbudget_scale_confirm_train800 dev sweep",
            },
        ),
    ]


__all__ = [
    "DEFAULT_COMMON_WARMUP_STEPS",
    "DEFAULT_DMD_RANK",
    "DEFAULT_N_TRAIN_TRAJECTORIES",
    "DEFAULT_NAVIER_STOKES_RANKS",
    "DEFAULT_FAST_TPLUS1_RANKS",
    "NavierStokesModelSpec",
    "get_fast_tplus1_model_specs",
    "get_navier_stokes_model_specs",
]
