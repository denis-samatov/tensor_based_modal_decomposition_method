import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MPL_CACHE_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

from TBMD.config.forecaster import LatentModalForecasterConfig
from TBMD.experiments.navier_stokes_forecasting import (
    TrajectoryAwareLatentForecaster,
    load_navier_stokes_trajectory_dataset,
)


def main():
    print("=" * 60)
    print(" TBMD State-of-the-Art Forecaster for Navier-Stokes")
    print("=" * 60)

    project_root = PROJECT_ROOT
    data_root = project_root / "data" / "navier_stokes"

    if not data_root.exists():
        print(f"Data not found: {data_root}")
        return

    print(f"Loading trajectory-aware data from {data_root}...")
    dataset = load_navier_stokes_trajectory_dataset(data_root)
    train_states = dataset.train_states
    test_states = dataset.test_states
    print(f"Train trajectories: {train_states.shape}")
    print(f"Test trajectories:  {test_states.shape}")

    r1, r2, r3 = 64, 64, 5
    print(f"\nInitializing trajectory-aware latent forecaster (R1={r1}, R2={r2}, R3={r3})")
    print("Prediction backend: Linear Forecaster")

    forecaster = TrajectoryAwareLatentForecaster(
        config=LatentModalForecasterConfig(
            ranks=[r1, r2, r3],
            forecaster_type="lstm",
            lstm_hidden_size=256,
            lstm_num_epochs=100,
            lstm_learning_rate=0.001,
            spatial_mean_centering=True,
            latent_normalization=True,
            delta_forecast=True,
            verbose=False,
        )
    )

    print("Training pipeline (decomposition + explicit trajectory-safe regression)...")
    forecaster.fit(train_states)
    print("Training complete.")

    print("\nEvaluating on official test trajectories:")
    one_step = forecaster.evaluate_one_step(test_states)
    rollout = forecaster.evaluate_rollout(test_states)

    print(f"One-step spatial R2:        {one_step['spatial_r2']:.4f}")
    print(f"Rollout spatial R2:         {rollout['spatial_r2']:.4f}")
    print(f"Rollout RMSE:               {rollout['spatial_rmse']:.4f}")
    print(f"Rollout relative Frob err:  {rollout['spatial_rel_frob_err']:.4f}")

    res_dir = project_root / "results"
    res_dir.mkdir(exist_ok=True)
    save_path = res_dir / "sota_navier_forecast_test.png"

    target_idx = min(10, rollout["pred_spatial"].shape[0] - 1)
    print(f"\nSaving rollout comparison for aggregated test step {target_idx}...")

    forecaster._adapter.plot_spatial_comparison(
        X_target=rollout["target_spatial"][target_idx],
        X_pred=rollout["pred_spatial"][target_idx],
        time_idx=target_idx,
        title=f"Trajectory-aware rollout (R2={rollout['spatial_r2']:.3f})",
        save_path=str(save_path),
        show=False,
    )
    print(f"Saved plot: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
