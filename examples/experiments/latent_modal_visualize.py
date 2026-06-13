#!/usr/bin/env python3
"""
Latent Modal Forecaster: trajectory-aware visualization script.
"""

import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MPL_CACHE_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from TBMD.config import LatentModalForecasterConfig
from TBMD.experiments.navier_stokes_forecasting import (
    TrajectoryAwareLatentForecaster,
    load_navier_stokes_trajectory_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("viz")


def main():
    data_root = PROJECT_ROOT / "data" / "navier_stokes"
    if not data_root.exists():
        logger.error(f"Data directory not found: {data_root}")
        sys.exit(1)

    logger.info("Loading trajectory-aware Navier-Stokes data...")
    dataset = load_navier_stokes_trajectory_dataset(data_root)
    train_states = dataset.train_states
    test_states = dataset.test_states

    out_dir = Path(__file__).parent / "plots"
    out_dir.mkdir(exist_ok=True)

    config = LatentModalForecasterConfig(
        ranks=[20, 20, 10],
        forecaster_type="linear",
        verbose=True,
    )

    logger.info("Training trajectory-aware latent forecaster...")
    forecaster = TrajectoryAwareLatentForecaster(config=config)
    forecaster.fit(train_states)

    logger.info("Evaluating rollout on official test trajectories...")
    metrics = forecaster.evaluate_rollout(test_states)
    x_target = metrics["target_spatial"]
    x_pred = metrics["pred_spatial"]

    timesteps_to_plot = [0, 10, 50, len(x_target) - 1]
    for t in timesteps_to_plot:
        if t >= len(x_target):
            continue

        save_path = str(out_dir / f"navier_prediction_t{t}.png")
        logger.info(f"Generating plot for rollout step {t} -> {save_path}")

        forecaster._adapter.plot_spatial_comparison(
            X_target=x_target[t],
            X_pred=x_pred[t],
            time_idx=t,
            title="Latent LINEAR: Navier-Stokes Rollout",
            save_path=save_path,
            show=False,
        )

    logger.info(f"All plots saved to {out_dir}")


if __name__ == "__main__":
    main()
