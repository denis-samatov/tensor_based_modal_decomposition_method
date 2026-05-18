# Stage 5 Final Fast TBMD+QR+CS T+1 Refit

Protocol: selected configs are refit on the requested train trajectories with no dev selection. The official test split is evaluated once per selected preset.

| Preset | Label | Rank | Sensors | Test R² | RMSE | MAE | Rel. Frob. | Inference s/sample | Model Size MB |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fast_tplus1_r300_s300 | practical | 300 | 300 | 0.8330 | 0.3422 | 0.2252 | 0.2532 | 0.000647 | 39.44 |
| fast_tplus1_r300_s600 | quality-max | 300 | 600 | 0.8371 | 0.3381 | 0.2222 | 0.2501 | 0.000834 | 39.44 |

Interpretation: this stage is a one-step-ahead sparse-sensing predictor, not a rollout forecaster. Unlike heavy neural-operator approaches, the proposed Fast TBMD+QR+CS pipeline constructs a compact TBMD hidden state from sparse sensing and learns only a lightweight correction head for one-step-ahead prediction.

FNO/PINN comparison status: local FNO/PINN baselines are not present in this run and should not be treated as directly comparable until trained and evaluated on the same split and metrics.
