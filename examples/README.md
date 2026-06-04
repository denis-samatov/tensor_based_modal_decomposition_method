# TBMD Examples

This directory contains runnable examples for the TBMD package. Run commands from the repository root after installing the package.

## Directory Overview

| Directory | Contents |
| --- | --- |
| `basic/` | Minimal decomposition, sensor placement, reconstruction, and complete-pipeline examples. |
| `digital_twin/` | Digital twin workflow examples. |
| `geometry_aware/` | Examples for graph and mesh-aware workflows. |
| `advanced/` | Advanced and legacy workflows. |
| `applications/` | Dataset-specific scripts. |
| `experiments/` | Experimental visualization and validation scripts. |

## Basic Examples

```bash
python examples/basic/01_tucker_decomposition.py
python examples/basic/02_sensor_placement.py
python examples/basic/03_field_reconstruction.py
python examples/basic/04_complete_pipeline.py
```

## Digital Twin Examples

```bash
python examples/digital_twin/01_digital_twin_basic.py
python examples/digital_twin/02_digital_twin_advanced.py
python examples/digital_twin/04_digital_twin_type_demo.py
```

## Geometry-Aware Examples

```bash
python examples/geometry_aware/01_graph_based_tbmd.py
python examples/geometry_aware/02_geometry_aware_cs.py
python examples/geometry_aware/03_geometry_aware_decomposition.py
python examples/geometry_aware/04_geometry_utils.py
python examples/geometry_aware/05_test_components.py
python examples/geometry_aware/06_geometry_aware_run.py
```

## Dataset-Specific Examples

Application examples may require local datasets that are not included in git:

```bash
python examples/applications/brugge_field/run_brugge_enhanced.py
python examples/02_navier_stokes_optimal_forecasting.py
```

Check each script before running it on a new machine. Keep local datasets and generated outputs out of version control.

## Additional Documentation

- [Quick start](../docs/guides/quick_start.md)
- [API reference](../docs/api/api_reference.md)
- [Model and data guide](../docs/guides/data_and_models.md)
