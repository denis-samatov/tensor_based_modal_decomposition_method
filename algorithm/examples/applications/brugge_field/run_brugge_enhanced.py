
import torch
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
# Add project root to path
# We need 'algorithm' in path to import TBMD
current_dir = os.path.dirname(os.path.abspath(__file__))
# Path to algorithm: brugge_field -> applications -> examples -> algorithm
algorithm_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if algorithm_path not in sys.path:
    sys.path.append(algorithm_path)

# Import TBMD modules
# Import TBMD modules
from TBMD.digital_twin.digital_twin import DigitalTwin as DigitalTwinTBMD
from TBMD.config import (
    DecompositionConfig,
    SensorPlacementConfig,
    ReconstructionConfig,
    DigitalTwinConfig,
    ProcessingStrategy
)
from TBMD.core.forecasting.ReservoirProxyModel import ReservoirState, WellControl
from TBMD.core.data.loaders import DataLoader
from TBMD.core.data.processors import process_data, calculate_global_minmax_params

def main():
    print("="*60)
    print("Brugge Digital Twin Demo")
    print("="*60)

    # 1. Load Data
    print("\n[1/5] Loading Data...")
    print("\n[1/5] Loading Data...")
    project_root = os.path.dirname(algorithm_path)
    data_dir = os.path.join(project_root, "data", "Brugge data")
    
    data_path = os.path.join(data_dir, "data_exp_4_.h5")
    wells_path = os.path.join(data_dir, "all_wells_exp_4.json")
    
    try:
        tensors = DataLoader.load_h5_tensors(data_path)
        wells = DataLoader.load_wells_from_json(wells_path)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Process wells (swap x,y as in notebook)
    for case_id in wells:
        wells[case_id] = [[x, y] for x, y in wells[case_id]]

    # Select 'case1' for this demo
    case_id = 'case1'
    if case_id not in tensors['all']:
        print(f"Error: {case_id} not found in loaded tensors.")
        return
        
    print(f"Selected Case: {case_id}")
    raw_data_np = tensors['all'][case_id] # Expected shape: (139, 48, 2, Time)
    print(f"Data Shape: {raw_data_np.shape}")
    
    # Split into History (Train) and Future (Test)
    n_time = raw_data_np.shape[-1]
    train_steps = int(n_time * 0.8)
    
    print(f"Splitting data: {train_steps} steps for training, {n_time - train_steps} for simulation.")
    
    train_data_raw = raw_data_np[..., :train_steps]
    test_data_raw = raw_data_np[..., train_steps:]
    
    # Normalize
    # We wrap in dict because process_data expects dict
    train_dict = {case_id: train_data_raw}
    test_dict = {case_id: test_data_raw}
    
    min_val, max_val = calculate_global_minmax_params(train_dict)
    minmax_params = {'min': min_val, 'max': max_val}
    print(f"Normalization params: {minmax_params}")
    
    train_tensors_processed = process_data(
        train_dict,
        normalization_method="minmax",
        global_params=minmax_params
    )
    
    test_tensors_processed = process_data(
        test_dict,
        normalization_method="minmax",
        global_params=minmax_params
    )
    
    historical_data = train_tensors_processed[case_id] # torch.Tensor
    future_data = test_tensors_processed[case_id]      # torch.Tensor
    
    # Ensure float32
    if isinstance(historical_data, torch.Tensor):
        historical_data = historical_data.float()
    elif isinstance(historical_data, np.ndarray):
        historical_data = torch.from_numpy(historical_data).float()
        
    if isinstance(future_data, torch.Tensor):
        future_data = future_data.float()
    elif isinstance(future_data, np.ndarray):
        future_data = torch.from_numpy(future_data).float()
        
    print(f"Historical Data Dtype: {historical_data.dtype}")
    
    # 2. Setup Dummy Controls
    # Since we don't have actual rate data in the H5, we generate dummy controls 
    # at the specified well locations.
    print("\n[2/5] Generating Dummy Controls...")
    well_locs = wells[case_id]
    
    def generate_controls(n_steps):
        controls = []
        for t in range(n_steps):
            ctrls = []
            for i, (wx, wy) in enumerate(well_locs):
                # Basic check to ensure well is within grid bounds
                if 0 <= wx < historical_data.shape[0] and 0 <= wy < historical_data.shape[1]:
                    # Random rate with some trend
                    rate = 100.0 + np.sin(t/10) * 20 + np.random.randn() * 5
                    ctrls.append(WellControl(
                        well_name=f"well_{i}",
                        control_type="rate",
                        value=rate,
                        location=(wx, wy)
                    ))
            controls.append(ctrls)
        return controls

    historical_controls = generate_controls(train_steps)
    future_controls = generate_controls(n_time - train_steps)
    
    # 3. Initialize Digital Twin
    print("\n[3/5] Initializing Digital Twin...")
    config = DigitalTwinConfig(
        n_spatial_modes=20, 
        n_temporal_modes=10,
        n_sensors=50, # Number of sensors to place
        proxy_model_type='physics_informed',
        device='cpu' # Use 'mps' or 'cuda' if available
    )
    
    twin = DigitalTwinTBMD(config)
    
    # 4. Train
    print("\n[4/5] Training Digital Twin...")
    try:
        summary = twin.train(historical_data, historical_controls)
        print("Training Summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Real-time Simulation Loop
    print("\n[5/5] Starting Real-time Simulation Loop...")
    
    monitor = twin.monitor
    
    # Initial state for simulation is the last state of training
    current_state = ReservoirState(
        pressure=historical_data[..., -1],
        time=float(train_steps - 1),
        well_rates={wc.well_name: wc.value for wc in historical_controls[-1]}
    )
    
    # Metrics
    errors = []
    
    print(f"{'Step':<10} | {'Status':<10} | {'Error %':<10} | {'Action':<10}")
    print("-" * 50)
    
    for t in range(future_data.shape[-1]):
        # 1. Get next controls (scenario)
        next_ctrls = future_controls[t]
        
        # 2. Predict next state (Digital Twin)
        # predict_next_state returns a list of ReservoirState
        prediction_states = twin.predict_next_state(current_state, next_ctrls)
        prediction = prediction_states[0].pressure
        
        # 3. Get "Real" measurement (from test data)
        # Simulate sparse sensors
        true_field = future_data[..., t]
        
        # Flatten true field to match sensor mask which is flat
        true_field_flat = true_field.flatten()
        sensor_mask = twin.sensor_locations.bool()
        sensor_readings = true_field_flat[sensor_mask]
        
        # 4. Update Twin with sensor data
        update_result = twin.update_from_sensors(sensor_readings)
        
        # 5. Monitor
        # Compare full reconstruction (from sensors) with prediction
        # Note: update_result['reconstructed_field'] is the field reconstructed from sensors
        reconstructed = update_result['reconstructed_field']
        
        # Calculate error between Prediction and Reality (approximated by Reconstruction or True Field)
        # Here we compare Prediction vs True Field for demo purposes
        rel_error = torch.norm(prediction - true_field) / torch.norm(true_field) * 100
        errors.append(rel_error.item())
        
        # Check alert
        status = update_result['alert_status']
        action = "NONE"
        
        if status == 'critical':
            # In a real system, we might recalibrate here
            action = "CALIBRATE"
            # Simple online calibration (concept)
            # twin.calibrate(...) 
        
        print(f"{t:<10} | {status:<10} | {rel_error.item():<10.2f} | {action:<10}")
        
        # Update state for next step
        # Ideally, we use the reconstructed state as the new starting point (Data Assimilation)
        current_state = ReservoirState(
            pressure=reconstructed, # Use assimilated state
            time=float(train_steps + t),
            well_rates={wc.well_name: wc.value for wc in next_ctrls}
        )
        
    print("\nSimulation Complete.")
    print(f"Average Prediction Error: {np.mean(errors):.2f}%")

if __name__ == "__main__":
    main()
