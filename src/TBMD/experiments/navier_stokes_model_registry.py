"""
Shared Navier-Stokes experiment model registry.

This module keeps final benchmark/example model definitions in one place so
quantitative reports and qualitative artifacts cannot silently drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from TBMD.config import LatentModalForecasterConfig, MultiResolutionTBMDConfig
from TBMD.experiments.navier_stokes_forecasting import (
    TrajectoryAwareDMDForecaster,
    TrajectoryAwareLatentForecaster,
    TrajectoryAwareMultiResolutionForecaster,
    TrajectoryAwarePersistenceForecaster,
    TrajectoryAwareResidualCorrectedForecaster,
)
from TBMD.experiments.navier_stokes_fast_tplus1 import (
    FastWindowedTBMDQRCSConfig,
    FastWindowedTBMDQRCSForecaster,
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
