# Geometry-Aware Examples

These examples demonstrate graph and mesh-aware TBMD workflows.

## Files

| File | Purpose |
| --- | --- |
| `01_graph_based_tbmd.py` | End-to-end comparison of standard and geometry-aware workflows on synthetic data. |
| `02_geometry_aware_cs.py` | Geometry-aware compressive sensing example. |
| `03_geometry_aware_decomposition.py` | Geometry-aware decomposition example. |
| `04_geometry_utils.py` | Mesh and graph utility examples. |
| `05_test_components.py` | Component-level validation script. |
| `06_geometry_aware_run.py` | Consolidated geometry-aware run script. |

## Run

```bash
python examples/geometry_aware/01_graph_based_tbmd.py
python examples/geometry_aware/02_geometry_aware_cs.py
python examples/geometry_aware/03_geometry_aware_decomposition.py
python examples/geometry_aware/04_geometry_utils.py
python examples/geometry_aware/05_test_components.py
python examples/geometry_aware/06_geometry_aware_run.py
```

## Guidance

- Validate graph construction before using geometry-aware outputs.
- Record graph parameters and regularization weights with every experiment.
- Treat synthetic comparisons as examples, not general accuracy claims.

## Documentation

- [Geometry-aware TBMD guide](../../docs/guides/geometry_aware_tbmd.md)
- [API reference](../../docs/api/api_reference.md)
