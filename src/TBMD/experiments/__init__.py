"""
Analytics module for TBMD experiments.

Provides unified experiment running and visualization tools.
"""

from TBMD.config.experiments import ExperimentConfig
from .runner import (
    ExperimentRunner,
    ensure_sensor_values_are_int,
)
from .navier_stokes_forecasting import (
    NavierStokesTrajectoryDataset,
    TrajectoryAwareCSForecaster,
    TrajectoryAwareDMDForecaster,
    TrajectoryAwareEigenvalueProjectedDMDForecaster,
    TrajectoryAwareLatentForecaster,
    TrajectoryAwareMultiResolutionForecaster,
    TrajectoryAwarePersistenceForecaster,
    TrajectoryAwareResidualCorrectedForecaster,
    TrajectoryAwareStableDMDForecaster,
    load_navier_stokes_trajectory_dataset,
)
from .navier_stokes_fast_tplus1 import (
    FastWindowedTBMDQRCSConfig,
    FastWindowedTBMDQRCSForecaster,
)
from .navier_stokes_examples import (
    build_examples_manifest,
    compute_common_horizon_diagnostics,
    compute_common_horizon_metrics,
    compute_spatial_metrics,
    extract_common_horizon_predictions,
    make_frame_filename,
    save_comparison_sheet,
    save_contact_sheet,
    save_t_plus_one_diagnostics_sheet,
    save_rollout_frame,
    save_rollout_gif,
    select_fixed_rollout_steps,
    select_fixed_trajectory_indices,
    split_train_dev_trajectories,
)
from .navier_stokes_model_registry import (
    DEFAULT_COMMON_WARMUP_STEPS,
    DEFAULT_DMD_RANK,
    DEFAULT_N_TRAIN_TRAJECTORIES,
    DEFAULT_NAVIER_STOKES_RANKS,
    NavierStokesModelSpec,
    get_fast_tplus1_model_specs,
    get_navier_stokes_model_specs,
)
from TBMD.visualization.experiments import (
    plot_analytics,
)

__all__ = [
    'ExperimentConfig',
    'ExperimentRunner',
    'ensure_sensor_values_are_int',
    'NavierStokesTrajectoryDataset',
    'TrajectoryAwareCSForecaster',
    'TrajectoryAwareDMDForecaster',
    'TrajectoryAwareEigenvalueProjectedDMDForecaster',
    'TrajectoryAwareLatentForecaster',
    'TrajectoryAwareMultiResolutionForecaster',
    'TrajectoryAwarePersistenceForecaster',
    'TrajectoryAwareResidualCorrectedForecaster',
    'TrajectoryAwareStableDMDForecaster',
    'load_navier_stokes_trajectory_dataset',
    'FastWindowedTBMDQRCSConfig',
    'FastWindowedTBMDQRCSForecaster',
    'build_examples_manifest',
    'compute_common_horizon_diagnostics',
    'compute_common_horizon_metrics',
    'compute_spatial_metrics',
    'extract_common_horizon_predictions',
    'make_frame_filename',
    'save_comparison_sheet',
    'save_contact_sheet',
    'save_t_plus_one_diagnostics_sheet',
    'save_rollout_frame',
    'save_rollout_gif',
    'select_fixed_rollout_steps',
    'select_fixed_trajectory_indices',
    'split_train_dev_trajectories',
    'DEFAULT_COMMON_WARMUP_STEPS',
    'DEFAULT_DMD_RANK',
    'DEFAULT_N_TRAIN_TRAJECTORIES',
    'DEFAULT_NAVIER_STOKES_RANKS',
    'NavierStokesModelSpec',
    'get_fast_tplus1_model_specs',
    'get_navier_stokes_model_specs',
    'plot_analytics',
]
