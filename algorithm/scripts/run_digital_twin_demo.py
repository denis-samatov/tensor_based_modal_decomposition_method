
"""
TBMD Digital Twin Demo
======================

This script demonstrates the "Light" Digital Twin workflow using TBMD and a simplified
hydrodynamic proxy model (Linear Dynamics).

Workflow:
1. Generate synthetic reservoir data (Pressure fields).
2. Train the Digital Twin:
    - Perform TBMD decomposition to find spatial modes.
    - Optimize sensor placement using Tensor QR.
    - Calibrate the simplified hydrodynamic model (Proxy Model).
3. Simulate Real-Time Operation:
    - Predict next state using the Proxy Model.
    - "Measure" data at sparse sensor locations.
    - Reconstruct the full field from sparse measurements.
    - Compare prediction vs observation and trigger alerts.
"""

import torch
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from typing import List

# Ensure algorithm is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'algorithm')))

from algorithm.TBMD.core.digital_twin.system import DigitalTwinTBMD, DigitalTwinConfig, WellControl
from algorithm.TBMD.models.ReservoirProxyModel import ReservoirState

def generate_synthetic_data(spatial_shape=(20, 20), n_time_steps=50):
    """
    Generate synthetic pressure data with some dynamics.
    Simulates a simple diffusion process with sources/sinks.
    """
    print(f"Generating synthetic data: {spatial_shape} x {n_time_steps} steps...")
    
    data = torch.zeros(*spatial_shape, n_time_steps)
    
    # Initial pressure
    data[..., 0] = 200.0 # bar
    
    # Grid
    x = torch.linspace(0, 1, spatial_shape[0])
    y = torch.linspace(0, 1, spatial_shape[1])
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Wells locations
    prod_loc = (5, 5)
    inj_loc = (15, 15)
    
    # Simple diffusion simulation
    dt = 0.01
    diffusivity = 0.1
    
    controls = []
    
    for t in range(n_time_steps - 1):
        current = data[..., t]
        
        # Laplacian
        laplacian = (
            torch.roll(current, 1, 0) + torch.roll(current, -1, 0) +
            torch.roll(current, 1, 1) + torch.roll(current, -1, 1) -
            4 * current
        )
        
        # Source/Sink terms
        # Vary rates slightly
        prod_rate = -50.0 + np.sin(t * 0.5) * 10
        inj_rate = 50.0 + np.cos(t * 0.5) * 10
        
        source = torch.zeros_like(current)
        source[prod_loc] = prod_rate
        source[inj_loc] = inj_rate
        
        # Update
        next_step = current + dt * (diffusivity * laplacian + source)
        
        # Boundary conditions (no flow - simplified)
        next_step[0, :] = next_step[1, :]
        next_step[-1, :] = next_step[-2, :]
        next_step[:, 0] = next_step[:, 1]
        next_step[:, -1] = next_step[:, -2]
        
        data[..., t+1] = next_step
        
        # Record controls
        ctrls = [
            WellControl('prod1', 'rate', prod_rate, prod_loc),
            WellControl('inj1', 'rate', inj_rate, inj_loc)
        ]
        controls.append(ctrls)
        
    # Add last control for consistency
    controls.append(controls[-1])
    
    return data, controls

def run_demo():
    # 1. Setup
    spatial_shape = (30, 30)
    n_time_steps = 100
    n_train = 80
    
    data, controls = generate_synthetic_data(spatial_shape, n_time_steps)
    
    # Split into train and test
    train_data = data[..., :n_train]
    train_controls = controls[:n_train]
    
    test_data = data[..., n_train:]
    test_controls = controls[n_train:]
    
    # 2. Configure Digital Twin
    config = DigitalTwinConfig(
        n_spatial_modes=10,
        n_temporal_modes=10,
        n_sensors=20,
        proxy_model_type='physics_informed', # Uses simplified hydrodynamic constraints
        reconstruction_method='admm',
        update_frequency=5,
        device='cpu'
    )
    
    twin = DigitalTwinTBMD(config)
    
    # 3. Train
    print("\nTraining Digital Twin...")
    summary = twin.train(train_data, train_controls)
    print("Training Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
        
    # 4. Real-time Simulation Loop
    print("\nStarting Real-Time Simulation Loop...")
    print(f"{'Time':<10} | {'Status':<10} | {'Error %':<10} | {'Action'}")
    print("-" * 50)
    
    current_state = ReservoirState(
        pressure=train_data[..., -1],
        time=float(n_train-1),
        well_rates={c.well_name: c.value for c in train_controls[-1]}
    )
    
    # Initialize state in twin
    twin.state.reservoir_state = current_state
    twin.state.current_time = current_state.time
    
    history_errors = []
    
    for t in range(test_data.shape[-1]):
        # A. Predict Next State (using simplified model)
        # We need controls for the NEXT step
        next_ctrls = test_controls[t]
        
        # This updates twin.state.reservoir_state with prediction
        twin.predict_next_state(twin.state.reservoir_state, next_ctrls)
        
        # B. "Measure" Data (Simulate incoming sensor data)
        true_field = test_data[..., t]
        
        # Extract readings at sensor locations
        # Note: sensor_locations is flattened, so we flatten field
        sensor_mask = twin.sensor_locations.bool()
        sensor_readings = true_field.flatten()[sensor_mask]
        
        # C. Update Twin (Reconstruct & Compare)
        result = twin.update_from_sensors(sensor_readings)
        
        # Log
        error = result['metrics'].get('relative_error', 0.0) * 100
        status = result['alert_status']
        history_errors.append(error)
        
        action = "None"
        if status == 'critical':
            action = "CALIBRATE"
        elif status == 'warning':
            action = "CHECK"
            
        print(f"{n_train + t:<10} | {status:<10} | {error:<10.2f} | {action}")

    print("\nDemo Completed.")
    print(f"Average Prediction Error: {np.mean(history_errors):.2f}%")

if __name__ == "__main__":
    run_demo()
