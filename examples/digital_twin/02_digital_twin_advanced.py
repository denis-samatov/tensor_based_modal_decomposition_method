"""
Digital Twin Example - Comprehensive Demonstration

This example demonstrates the complete workflow for building and using a
digital twin of a reservoir based on TBMD.

The example covers:
1. Training the digital twin from historical data
2. Real-time monitoring and field reconstruction
3. Scenario analysis (what-if studies)
4. Alert detection and model updating
5. Performance evaluation

Author: TBMD Team
Date: 2025
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from TBMD.config import SEED
from TBMD.core.utils.misc import set_seed
from TBMD.modules.DigitalTwinTBMD import (
    DigitalTwinConfig,
    DigitalTwinTBMD,
    ReservoirState,
    WellControl,
)

# Set random seed for reproducibility
set_seed(SEED)


def generate_synthetic_reservoir_data(spatial_shape=(50, 50), n_timesteps=100, n_wells=5, seed=42):
    """
    Generate synthetic reservoir data for demonstration.

    Simulates pressure fields evolving over time with well influences.

    Parameters
    ----------
    spatial_shape : tuple
        Spatial dimensions (nx, ny)
    n_timesteps : int
        Number of time steps
    n_wells : int
        Number of wells
    seed : int
        Random seed

    Returns
    -------
    data : torch.Tensor
        Pressure field data (nx, ny, t)
    well_locations : list
        Well locations
    well_controls : list
        Well controls over time
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    nx, ny = spatial_shape

    # Create spatial grid
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Generate random well locations
    well_locations = []
    for _ in range(n_wells):
        wx = np.random.randint(nx // 4, 3 * nx // 4)
        wy = np.random.randint(ny // 4, 3 * ny // 4)
        well_locations.append((wx, wy))

    # Generate time-varying pressure fields
    data = np.zeros((nx, ny, n_timesteps))

    # Base pressure field (gradually decreasing)
    base_pressure = 100.0

    # Well controls (some producers, some injectors)
    well_controls_history = []

    for t in range(n_timesteps):
        # Time-varying component
        time_factor = np.exp(-0.01 * t)

        # Spatial modes (could be replaced with actual physics)
        mode1 = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        mode2 = np.cos(np.pi * X) * np.sin(3 * np.pi * Y)
        mode3 = np.sin(3 * np.pi * X) * np.cos(np.pi * Y)

        # Combine modes with time-varying amplitudes
        field = (
            base_pressure * time_factor
            + 10 * np.sin(0.1 * t) * mode1
            + 5 * np.cos(0.15 * t) * mode2
            + 3 * np.sin(0.08 * t) * mode3
        )

        # Add well influences
        well_controls = []
        for i, (wx, wy) in enumerate(well_locations):
            # Alternate between producers and injectors
            if i % 2 == 0:
                rate = -5.0 - 2 * np.sin(0.05 * t)  # Producer
            else:
                rate = 8.0 + 3 * np.cos(0.07 * t)  # Injector

            well_controls.append(WellControl(f"WELL_{i + 1}", "rate", rate, (wx, wy)))

            # Gaussian influence around well
            dist = np.sqrt((X - x[wx]) ** 2 + (Y - y[wy]) ** 2)
            influence = rate * np.exp(-50 * dist**2)
            field += influence

        well_controls_history.append(well_controls)

        # Add noise
        field += np.random.normal(0, 0.5, field.shape)

        data[:, :, t] = field

    # Convert to torch
    data_tensor = torch.from_numpy(data).float()

    return data_tensor, well_locations, well_controls_history


def plot_pressure_field(field, title="Pressure Field", wells=None, sensors=None):
    """Plot a pressure field with optional wells and sensors."""
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(field, cmap="viridis", origin="lower", aspect="auto")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)

    # Plot wells
    if wells is not None:
        well_x = [w[1] for w in wells]
        well_y = [w[0] for w in wells]
        ax.scatter(
            well_x,
            well_y,
            c="red",
            marker="s",
            s=100,
            label="Wells",
            edgecolors="black",
            linewidths=2,
        )

    # Plot sensors
    if sensors is not None:
        sensor_positions = np.argwhere(sensors.cpu().numpy() == 1)
        if len(sensor_positions) > 0:
            ax.scatter(
                sensor_positions[:, 1],
                sensor_positions[:, 0],
                c="cyan",
                marker="o",
                s=50,
                label="Sensors",
                edgecolors="black",
                linewidths=1,
            )

    if wells is not None or sensors is not None:
        ax.legend(fontsize=10, loc="upper right")

    plt.colorbar(im, ax=ax, label="Pressure")
    plt.tight_layout()
    return fig


def plot_scenario_comparison(scenarios_results, kpi="avg_pressure"):
    """Plot comparison of scenarios."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for scenario_name, result in scenarios_results.items():
        states = result["forecasted_states"]
        times = [s.time for s in states]

        if kpi == "avg_pressure":
            values = [torch.mean(s.pressure).item() for s in states]
            ylabel = "Average Pressure"
        elif kpi == "max_pressure":
            values = [torch.max(s.pressure).item() for s in states]
            ylabel = "Maximum Pressure"
        else:
            values = [torch.mean(s.pressure).item() for s in states]
            ylabel = "Average Pressure"

        ax.plot(times, values, marker="o", linewidth=2, label=scenario_name)

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"Scenario Comparison - {kpi}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_monitoring_history(twin):
    """Plot monitoring history."""
    summary = twin.get_monitoring_summary()

    if summary.get("status") == "no monitoring data":
        print("No monitoring data to plot")
        return None

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Prediction errors over time
    times = twin.state.history["times"]
    errors = twin.state.history["errors"]
    alerts = twin.state.history["alerts"]

    ax1 = axes[0]
    ax1.plot(times, errors, marker="o", linewidth=2, color="blue")
    ax1.axhline(
        y=twin.config.alert_threshold, color="orange", linestyle="--", label="Warning Threshold"
    )
    ax1.axhline(
        y=2 * twin.config.alert_threshold, color="red", linestyle="--", label="Critical Threshold"
    )
    ax1.set_xlabel("Time", fontsize=12)
    ax1.set_ylabel("Relative Error", fontsize=12)
    ax1.set_title("Prediction Error History", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Alert status over time
    ax2 = axes[1]
    alert_colors = {"normal": "green", "warning": "orange", "critical": "red"}
    alert_values = {"normal": 0, "warning": 1, "critical": 2}

    colors = [alert_colors[a] for a in alerts]
    values = [alert_values[a] for a in alerts]

    ax2.scatter(times, values, c=colors, s=50, alpha=0.7)
    ax2.set_xlabel("Time", fontsize=12)
    ax2.set_ylabel("Alert Level", fontsize=12)
    ax2.set_title("Alert Status History", fontsize=14, fontweight="bold")
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["Normal", "Warning", "Critical"])
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    return fig


def main():
    """Main demonstration function."""
    print("=" * 70)
    print(" Digital Twin Demonstration ".center(70, "="))
    print("=" * 70)
    print()

    # =========================================================================
    # Step 1: Generate synthetic data
    # =========================================================================
    print("[Step 1] Generating synthetic reservoir data...")

    spatial_shape = (50, 50)
    n_timesteps = 100
    n_wells = 5

    data, well_locations, well_controls_history = generate_synthetic_reservoir_data(
        spatial_shape=spatial_shape, n_timesteps=n_timesteps, n_wells=n_wells, seed=SEED
    )

    print(f"  Data shape: {data.shape}")
    print(f"  Number of wells: {n_wells}")
    print(f"  Time steps: {n_timesteps}")
    print()

    # Split into training and monitoring
    train_data = data[..., :80]
    monitor_data = data[..., 80:]
    train_controls = well_controls_history[:80]
    monitor_controls = well_controls_history[80:]

    # =========================================================================
    # Step 2: Initialize and train digital twin
    # =========================================================================
    print("[Step 2] Initializing and training digital twin...")

    config = DigitalTwinConfig(
        n_spatial_modes=30,
        n_temporal_modes=15,
        n_sensors=25,
        proxy_model_type="linear",
        use_geometry_aware=False,
        reconstruction_method="admm",
        update_frequency=5,
        alert_threshold=0.10,
        device="cpu",
    )

    twin = DigitalTwinTBMD(config)

    # Train the twin
    training_summary = twin.train(historical_data=train_data, historical_controls=train_controls)

    print("\n  Training Summary:")
    print(
        f"    Reconstruction error: {training_summary['decomposition']['reconstruction_error']:.4f}"
    )
    print(f"    Sensors placed: {training_summary['sensor_placement']['actual_sensors']}")
    print(f"    Calibration MSE: {training_summary['calibration'].get('mse', 'N/A')}")
    print()

    # Visualize initial setup
    print("[Visualization] Plotting initial field and sensor placement...")
    plot_pressure_field(
        train_data[..., 0].numpy(),
        title="Initial Pressure Field with Wells and Sensors",
        wells=well_locations,
        sensors=twin.sensor_locations,
    )
    plt.savefig("digital_twin_setup.png", dpi=150, bbox_inches="tight")
    print("  Saved: digital_twin_setup.png")
    plt.close()

    # =========================================================================
    # Step 3: Real-time monitoring simulation
    # =========================================================================
    print("\n[Step 3] Simulating real-time monitoring...")

    # Initialize with last training state
    initial_state = ReservoirState(pressure=train_data[..., -1], time=80.0)
    twin.state.reservoir_state = initial_state

    # Monitor over time
    for t_idx in range(len(monitor_data.shape[-1])):
        current_time = 80.0 + t_idx

        # Predict next state
        twin.predict_next_state(
            current_state=twin.state.reservoir_state,
            well_controls=monitor_controls[t_idx] if t_idx < len(monitor_controls) else [],
            time_horizon=1.0,
            time_steps=1,
        )

        # Simulate sensor readings
        true_field = monitor_data[..., t_idx]
        sensor_mask = twin.sensor_locations.bool()
        sensor_readings = true_field[sensor_mask]

        # Update from sensors
        update_result = twin.update_from_sensors(
            sensor_readings=sensor_readings,
            sensor_locations=twin.sensor_locations,
            current_time=current_time,
        )

        if t_idx % 5 == 0:
            print(
                f"  t={current_time:.1f}: error={update_result['metrics'].get('relative_error', 0):.4f}, "
                f"status={update_result['alert_status']}"
            )

    print()

    # Plot monitoring history
    print("[Visualization] Plotting monitoring history...")
    fig2 = plot_monitoring_history(twin)
    if fig2:
        plt.savefig("digital_twin_monitoring.png", dpi=150, bbox_inches="tight")
        print("  Saved: digital_twin_monitoring.png")
        plt.close()

    # =========================================================================
    # Step 4: Scenario analysis
    # =========================================================================
    print("\n[Step 4] Performing scenario analysis...")

    # Define scenarios
    scenarios = {}

    # Baseline scenario
    scenarios["Baseline"] = [
        WellControl(f"WELL_{i + 1}", "rate", -5.0 if i % 2 == 0 else 8.0, well_locations[i])
        for i in range(n_wells)
    ]

    # Increased injection scenario
    scenarios["Increased Injection"] = [
        WellControl(
            f"WELL_{i + 1}",
            "rate",
            -5.0 if i % 2 == 0 else 12.0,  # Increased injection
            well_locations[i],
        )
        for i in range(n_wells)
    ]

    # Reduced production scenario
    scenarios["Reduced Production"] = [
        WellControl(
            f"WELL_{i + 1}",
            "rate",
            -3.0 if i % 2 == 0 else 8.0,  # Reduced production
            well_locations[i],
        )
        for i in range(n_wells)
    ]

    # Evaluate scenarios
    scenario_results = twin.evaluate_scenarios(
        scenarios=scenarios, time_horizon=10.0, time_steps=10
    )

    # Print KPIs
    print("\n  Scenario KPIs:")
    for name, result in scenario_results.items():
        kpis = result["kpis"]
        print(f"\n  {name}:")
        print(f"    Avg Pressure: {kpis['avg_pressure']:.2f}")
        print(f"    Total Production: {kpis['total_production']:.2f}")
        print(f"    Total Injection: {kpis['total_injection']:.2f}")

    # Plot scenario comparison
    print("\n[Visualization] Plotting scenario comparison...")
    plot_scenario_comparison(scenario_results, kpi="avg_pressure")
    plt.savefig("digital_twin_scenarios.png", dpi=150, bbox_inches="tight")
    print("  Saved: digital_twin_scenarios.png")
    plt.close()

    # =========================================================================
    # Step 5: Summary
    # =========================================================================
    print("\n[Step 5] Digital Twin Summary")
    print("=" * 70)

    summary = twin.get_monitoring_summary()
    print(f"  Total monitoring updates: {summary['total_updates']}")
    print(f"  Mean prediction error: {summary['mean_error']:.4f}")
    print(f"  Max prediction error: {summary['max_error']:.4f}")
    print(
        f"  Alerts - Normal: {summary['normal_count']}, "
        f"Warning: {summary['warning_count']}, "
        f"Critical: {summary['critical_count']}"
    )

    # =========================================================================
    # Step 6: Save digital twin
    # =========================================================================
    print("\n[Step 6] Saving digital twin...")
    save_dir = Path(__file__).parent / "digital_twin_saved"
    twin.save(save_dir)
    print(f"  Digital twin saved to: {save_dir}")

    print("\n" + "=" * 70)
    print(" Demonstration Complete ".center(70, "="))
    print("=" * 70)
    print("\nGenerated visualizations:")
    print("  - digital_twin_setup.png")
    print("  - digital_twin_monitoring.png")
    print("  - digital_twin_scenarios.png")
    print("\nThe digital twin demonstrates:")
    print("  ✓ Efficient data representation via TBMD")
    print("  ✓ Optimal sensor placement")
    print("  ✓ Real-time field reconstruction")
    print("  ✓ Prediction vs observation monitoring")
    print("  ✓ Fast what-if scenario analysis")
    print("  ✓ Alert detection and model updating")


if __name__ == "__main__":
    main()
