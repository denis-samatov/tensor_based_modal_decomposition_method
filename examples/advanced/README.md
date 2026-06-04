# Advanced TBMD Examples

This directory contains advanced and legacy workflows. Prefer `examples/basic/` for first-time usage.

## Files

| File | Purpose |
| --- | --- |
| `01_modal_tensor_processing.py` | Modal tensor processing workflow. |
| `02_unified_experiments.py` | Combined experiment workflow for advanced users. |
| `03_sensor_values_fix.py` | Sensor-data validation and correction helper. |
| `04_legacy_tbmd.py` | Legacy TBMD API example. |
| `05_legacy_qr.py` | Legacy QR sensor placement example. |
| `06_legacy_cs.py` | Legacy compressive sensing example. |
| `07_legacy_pipeline.py` | Legacy end-to-end pipeline example. |

## Legacy Status

Files prefixed with `legacy` or described as legacy are kept for compatibility and historical context. New examples should use the `TBMD.core` and `TBMD.config` APIs documented in the quick start and API reference.

## Run

```bash
python examples/advanced/01_modal_tensor_processing.py
python examples/advanced/02_unified_experiments.py
python examples/advanced/03_sensor_values_fix.py
```

Run legacy scripts only when validating backward compatibility.

## Related Documentation

- [Quick start](../../docs/guides/quick_start.md)
- [API reference](../../docs/api/api_reference.md)
