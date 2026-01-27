# Migration Guide: TBMD `modules` $\to$ `core`

**Date:** 2026-01-27
**Status:** `modules` is Deprecated.

## Overview
The TBMD codebase has been refactored. The legacy `modules/` directory is deprecated and will be removed in future versions. Functionality has been moved to `core/` with a new configuration system.

## Quick Reference

| Legacy (Deprecated) | New Core Component | New Config |
| :--- | :--- | :--- |
| `TBMD.modules.TensorHOSVD` | `TBMD.core.decomposition.hosvd` | `DecompositionConfig` |
| `TBMD.modules.TensorBasedCompressiveSensing` | `TBMD.core.reconstruction.tensor_compressive_sensing` | `CompressiveSensingConfig` |
| `TBMD.modules.GeometryAwareTensorHOSVD` | `TBMD.core.decomposition.geometry_aware` | `GeometryAwareConfig` |
| `TBMD.modules.TensorBasedTubeFiberPivotQRFactorization` | `TBMD.core.sensor_placement.tensor_qr_factorization` | `SensorPlacementConfig` |

## Migration Steps

1.  **Update Imports:** Change `from TBMD.modules...` to `from TBMD.core...`.
2.  **Use Config Objects:** Instead of passing many kwargs to `__init__`, pass a `config` dataclass.
    ```python
    # Old
    decomp = TuckerDecomposerInterface(tensor, ranks=[10,10,10])
    
    # New
    from TBMD.config.decomposition_config import DecompositionConfig
    cfg = DecompositionConfig(ranks=[10,10,10])
    decomp = TuckerDecomposerInterface(tensor, config=cfg)
    ```
3.  **Check Defaults:** Verify that default parameters in `Config` objects match your expectations.
