# Digital Twin Examples

These examples demonstrate the TBMD digital twin orchestration layer on synthetic or local data.

## Files

| File | Purpose |
| --- | --- |
| `01_digital_twin_basic.py` | Basic synthetic workflow: train, forecast, sensor update, and simple visualization. |
| `02_digital_twin_advanced.py` | Extended workflow for users who want to inspect additional digital twin capabilities. |
| `04_digital_twin_type_demo.py` | Compatibility and type-handling demonstration. |

## Run

```bash
python examples/digital_twin/01_digital_twin_basic.py
python examples/digital_twin/02_digital_twin_advanced.py
python examples/digital_twin/04_digital_twin_type_demo.py
```

Run commands from the repository root so package imports resolve consistently.

## Typical Workflow

```text
historical data
    -> DigitalTwinConfig
    -> DigitalTwin.train()
    -> DigitalTwin.predict()
    -> DigitalTwin.update_from_sensors()
```

## Notes

- Synthetic examples are useful smoke tests, not benchmark evidence.
- Forecast quality depends on the dataset, split, ranks, sensor count, and forecaster configuration.
- Keep generated figures and local datasets out of git.

## Related Documentation

- [Digital twin guide](../../docs/guides/digital_twin.md)
- [Digital twin tutorial](../../docs/tutorials/digital_twin_tutorial.md)
- [API reference](../../docs/api/api_reference.md)
