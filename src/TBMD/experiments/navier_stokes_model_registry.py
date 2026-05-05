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
    TrajectoryAwareLatentForecaster,
    TrajectoryAwareMultiResolutionForecaster,
    TrajectoryAwareResidualCorrectedForecaster,
)

DEFAULT_NAVIER_STOKES_RANKS = [64, 64, 5]
DEFAULT_COMMON_WARMUP_STEPS = 7
DEFAULT_N_TRAIN_TRAJECTORIES = 1000


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
                    lstm_seq_length=7,
                    lstm_num_epochs=150,
                ),
                feature_mode="latent_plus_delta",
            ),
            notes={
                "selected_preset": "lstm_h128_l2_s7_e150_plus_delta_features",
                "feature_mode": "latent_plus_delta",
                "delta_forecast": False,
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
                    lstm_seq_length=7,
                    lstm_num_epochs=150,
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
                "selected_preset": "lstm_h128_l2_s7_e150_plus_delta_features_with_t_plus_1_residual_head",
                "feature_mode": "latent_plus_delta",
                "delta_forecast": False,
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


__all__ = [
    "DEFAULT_COMMON_WARMUP_STEPS",
    "DEFAULT_N_TRAIN_TRAJECTORIES",
    "DEFAULT_NAVIER_STOKES_RANKS",
    "NavierStokesModelSpec",
    "get_navier_stokes_model_specs",
]
