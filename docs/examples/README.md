# Examples Documentation

The repository examples live in the top-level `examples/` directory. This page explains the stable entry points and their expected scope.

## Example Groups

| Directory | Purpose |
| --- | --- |
| [`examples/basic/`](../../examples/basic/) | Decomposition, sensor placement, reconstruction, and complete synthetic workflows. |
| [`examples/digital_twin/`](../../examples/digital_twin/) | Digital twin training, forecasting, and compatibility demonstrations. |
| [`examples/geometry_aware/`](../../examples/geometry_aware/) | Graph and mesh-aware decomposition, sensor placement, and reconstruction examples. |
| [`examples/advanced/`](../../examples/advanced/) | Advanced and legacy workflows for maintainers and experienced users. |
| [`examples/applications/`](../../examples/applications/) | Dataset-specific application scripts. These may require local data. |

## Run Basic Examples

Run commands from the repository root:

```bash
python examples/basic/01_tucker_decomposition.py
python examples/basic/02_sensor_placement.py
python examples/basic/03_field_reconstruction.py
python examples/basic/04_complete_pipeline.py
```

## Run Digital Twin Examples

```bash
python examples/digital_twin/01_digital_twin_basic.py
python examples/digital_twin/02_digital_twin_advanced.py
python examples/digital_twin/04_digital_twin_type_demo.py
```

## Dataset-Specific Examples

`examples/applications/brugge_field/run_brugge_enhanced.py` expects local Brugge-style data files. The repository does not guarantee that those datasets are present in a fresh clone.

See [Model and data handling](../guides/data_and_models.md) before adding datasets or generated outputs.
