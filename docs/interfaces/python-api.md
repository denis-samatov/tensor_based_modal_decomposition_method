# Python API

## Purpose
Documents the primary programmatic interfaces for the TBMD project.

## Audience
Developers and ML Engineers building scripts around the core library.

## Summary
This repository does not expose a stable REST/HTTP API. The primary interfaces are Python modules, scripts, and Jupyter notebooks.

## Details

### Core Classes
The library is structured around several key classes in `src/TBMD/core/`:

1. `TuckerDecomposer` / `GeometryAwareTuckerDecomposer`: 
   - Responsible for tensor factorization.
   - Key methods: `.fit()`, `.transform()`, `.inverse_transform()`.
   
2. `ADMMReconstructor`:
   - Solves the compressive sensing inverse problem.
   - Key methods: `.reconstruct()`.

3. `QRBasedSensorPlacement`:
   - Determines sparse sensor locations.
   - Key methods: `.compute_sensor_locations()`.

### Configuration Interfaces
Configurations are defined using `dataclasses` in `src/TBMD/config/`:
- `DecompositionConfig`
- `SensorPlacementConfig`
- `ReconstructionConfig`
- `FullPipelineConfig`

## Examples
See the Python scripts in `examples/applications/` for end-to-end usage of these classes.
