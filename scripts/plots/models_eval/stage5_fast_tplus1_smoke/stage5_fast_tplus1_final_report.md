# Stage 5 Final Fast TBMD+QR+CS T+1 Refit

Protocol: selected configs are refit on the requested train trajectories with no dev selection. The official test split is evaluated once per selected preset.

| Preset | Label | Rank | Sensors | Test R² | RMSE | MAE | Rel. Frob. | Inference s/sample | Model Size MB |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fast_tplus1_r300_s300 | practical | 300 | 300 | 0.9901 | 0.0817 | 0.0574 | 0.0611 | 0.004201 | 39.52 |

Interpretation: this stage is a one-step-ahead sparse-sensing predictor, not a rollout forecaster. Unlike heavy neural-operator approaches, the proposed Fast TBMD+QR+CS pipeline constructs a compact TBMD hidden state from sparse sensing and learns only a lightweight correction head for one-step-ahead prediction.

FNO/PINN comparison status: local FNO/PINN baselines are not present in this run and should not be treated as directly comparable until trained and evaluated on the same split and metrics.
