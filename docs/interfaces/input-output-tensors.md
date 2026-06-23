# Input and Output Tensors

## Purpose
Details the expected formats for data passing through the library.

## Audience
Developers preparing datasets for analysis.

## Summary
The system primarily expects multi-dimensional PyTorch tensors. Data loaders must transform raw files (like CSV or HDF5) into the expected tensor shapes before interacting with the core library.

## Details

### Typical State Tensors
For a 2D spatial field evolving over time (e.g., fluid dynamics), the typical expected shape is:
`[features, x_dim, y_dim, time_steps]`

- `features`: Variables like pressure, saturation, or velocity components.
- `x_dim, y_dim`: Spatial grid dimensions.
- `time_steps`: Temporal snapshots.

### Sparse Measurements
When running online reconstruction, the sensor data is expected as a sparse vector `y` of shape `[num_sensors]`, mapping to the spatial points defined by the sensor placement algorithm.

### Outputs
The `inverse_transform` methods return dense tensors matching the original state tensor shape `[features, x_dim, y_dim, time_steps]`.

## Validation
Always print `tensor.shape` and assert it matches the initialized configuration before calling `.fit()`.
