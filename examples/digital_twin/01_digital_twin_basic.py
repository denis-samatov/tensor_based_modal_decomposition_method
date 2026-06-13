#!/usr/bin/env python3
"""
Digital Twin Basic Example

Demonstrates basic Digital Twin capabilities with TBMD.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from TBMD.config import DigitalTwinConfig
from TBMD.digital_twin import DigitalTwin

print("=" * 60)
print("TBMD Digital Twin - Basic Example")
print("=" * 60)

# 1. Create synthetic reservoir data.
print("\n1. Creating synthetic data...")
I = 100  # Spatial points
J = 2  # Variables: pressure and saturation
T = 50  # Time steps

# Simulate simple dynamics.
np.random.seed(42)
torch.manual_seed(42)

# Base field.
x = torch.linspace(0, 1, I)
base_pressure = torch.sin(2 * np.pi * x).unsqueeze(0).unsqueeze(-1)  # (1, I, 1)
base_saturation = torch.cos(2 * np.pi * x).unsqueeze(0).unsqueeze(-1)  # (1, I, 1)

# Temporal dynamics.
time = torch.linspace(0, 1, T)
temporal_evolution = torch.exp(-0.1 * time).unsqueeze(0).unsqueeze(0)  # (1, 1, T)

# Historical data (I x J x T).
historical_data = torch.zeros(I, J, T)
historical_data[:, 0, :] = (base_pressure.squeeze() * temporal_evolution.squeeze()).T
historical_data[:, 1, :] = (base_saturation.squeeze() * (1 - temporal_evolution.squeeze())).T

# Add noise.
historical_data += 0.1 * torch.randn_like(historical_data)

print(f"   Data created: {historical_data.shape}")
print(
    f"   Pressure range: [{historical_data[:, 0, :].min():.2f}, {historical_data[:, 0, :].max():.2f}]"
)
print(
    f"   Saturation range: [{historical_data[:, 1, :].min():.2f}, {historical_data[:, 1, :].max():.2f}]"
)

# 2. Create and train Digital Twin.
print("\n2. Training Digital Twin...")
config = DigitalTwinConfig(n_spatial_modes=20, n_temporal_modes=10, n_sensors=15, verbose=True)

twin = DigitalTwin(config)
twin.train(historical_data, normalize=True)

print("   Twin trained.")
print(f"   Sensors placed: {len(twin.get_sensor_locations())}")

# 3. Forecasting.
print("\n3. Forecasting future states...")
current_state = historical_data[:, :, -1]  # Last state.
forecast = twin.predict(current_state, n_steps=10, return_full_field=True)

print(f"   Forecast created: {forecast.shape}")
print(f"   Forecast range: [{forecast.min():.2f}, {forecast.max():.2f}]")

# 4. Measurement simulation and reconstruction.
print("\n4. Reconstructing from sensor measurements...")
sensor_indices = twin.get_sensor_locations()
sensor_index_tensor = torch.as_tensor(sensor_indices, dtype=torch.long)
sensor_i = sensor_index_tensor // J
sensor_j = sensor_index_tensor % J

# Simulate measurements from flat sensor indices.
true_field = current_state.clone()
sensor_measurements = true_field.reshape(-1)[sensor_index_tensor]

print(f"   Measurements from {len(sensor_indices)} sensors")

# Reconstruct.
reconstruction_result = twin.update_from_sensors(sensor_measurements)
reconstructed = reconstruction_result["reconstructed_field"]

# Compute error.
error = torch.norm(reconstructed - true_field) / torch.norm(true_field)
print(f"   Reconstruction error: {error:.4f}")

# 5. Scenario analysis.
print("\n5. Scenario analysis...")
scenarios = [
    {"name": "baseline", "description": "Current operating mode"},
    {"name": "optimistic", "description": "Optimistic scenario"},
    {"name": "pessimistic", "description": "Pessimistic scenario"},
]

results = twin.evaluate_scenarios(scenarios, n_steps=10)

print(f"   Evaluated {len(results)} scenarios:")
for name, metrics in results.items():
    print(f"   - {name}: mean={metrics['mean_value']:.3f}, std={metrics['std_value']:.3f}")

# 6. Anomaly detection.
print("\n6. Anomaly detection...")

# Create data with an anomaly.
sensor_data_normal = torch.randn(len(sensor_indices), 10) * 0.1
sensor_data_anomaly = sensor_data_normal.clone()
sensor_data_anomaly[:, 5] += 5.0  # Add an anomaly at step 5.

anomalies = twin.detect_anomalies(sensor_data_anomaly, threshold=2.0)

print(f"   Detected anomalies: {len(anomalies)}")
for anomaly in anomalies:
    print(
        f"   - Timestamp: {anomaly['timestamp']}, "
        f"Residual: {anomaly['residual']:.3f}, "
        f"Severity: {anomaly['severity']}"
    )

# 7. Statistics.
print("\n7. Digital Twin statistics...")
stats = twin.get_statistics()
print(f"   Calibrated: {stats['is_calibrated']}")
print(f"   Spatial modes: {stats['n_spatial_modes']}")
print(f"   Sensors: {stats['n_sensors']}")
print(f"   Status: {stats['alert_status']}")

# 8. Visualization.
print("\n8. Creating visualization...")
try:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Digital Twin Results", fontsize=16)

    # Historical data.
    axes[0, 0].plot(historical_data[50, 0, :].numpy(), label="Pressure")
    axes[0, 0].set_title("Historical Pressure (Point 50)")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].legend()

    # Current state.
    axes[0, 1].plot(current_state[:, 0].numpy(), label="Pressure")
    pressure_sensor_points = sensor_i[sensor_j == 0]
    if len(pressure_sensor_points) > 0:
        axes[0, 1].scatter(
            pressure_sensor_points.numpy(),
            current_state[pressure_sensor_points, 0].numpy(),
            c="red",
            label="Sensors",
            zorder=5,
        )
    axes[0, 1].set_title("Current Field + Sensors")
    axes[0, 1].legend()

    # Forecast.
    axes[0, 2].plot(forecast[50, 0, :].numpy())
    axes[0, 2].set_title("Forecast (Point 50)")
    axes[0, 2].set_xlabel("Forecast Steps")

    # Reconstruction.
    axes[1, 0].plot(true_field[:, 0].numpy(), label="True")
    axes[1, 0].plot(reconstructed[:, 0].numpy(), label="Reconstructed", alpha=0.7)
    axes[1, 0].set_title("Reconstruction vs True")
    axes[1, 0].legend()

    # Scenarios.
    scenario_names = list(results.keys())
    scenario_means = [results[name]["mean_value"] for name in scenario_names]
    axes[1, 1].bar(range(len(scenario_names)), scenario_means)
    axes[1, 1].set_xticks(range(len(scenario_names)))
    axes[1, 1].set_xticklabels(scenario_names, rotation=45)
    axes[1, 1].set_title("Scenarios - Mean Values")

    # Anomalies.
    residuals = [a["residual"] for a in anomalies]
    timestamps = [a["timestamp"] for a in anomalies]
    if anomalies:
        axes[1, 2].scatter(timestamps, residuals, c="red", s=100)
        axes[1, 2].axhline(y=2.0, color="orange", linestyle="--", label="Threshold")
        axes[1, 2].set_title("Detected Anomalies")
        axes[1, 2].set_xlabel("Timestamp")
        axes[1, 2].set_ylabel("Residual")
        axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig("digital_twin_results.png", dpi=150, bbox_inches="tight")
    print("   Visualization saved: digital_twin_results.png")

except Exception as e:
    print(f"   Visualization skipped: {e}")

print("\n" + "=" * 60)
print("Digital Twin Example completed successfully.")
print("=" * 60)
