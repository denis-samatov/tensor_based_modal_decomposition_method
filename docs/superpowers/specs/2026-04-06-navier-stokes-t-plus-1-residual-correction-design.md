# Navier-Stokes T+1 Residual Correction Design

## Goal

Добавить поверх текущего trajectory-aware TBMD forecasting отдельную `t+1`-надстройку, которая не заменяет базовый forecaster, а корректирует его one-step ошибку в latent-space.

## Why This Design

Текущий лучший pipeline уже исправил dataset protocol и improved latent representation через `feature_mode="latent_plus_delta"`. Следующий узкий bottleneck для `t+1` находится не в loader и не в decomposition API, а в локальной one-step ошибке базового latent forecaster.

Поэтому выбран двухступенчатый путь:

1. Базовый `TrajectoryAwareLatentForecaster` обучается как раньше.
2. Отдельный correction head учится предсказывать residual для `c_{t+1}`.

Итоговый предикт:

`c_t -> baseline ĉ_{t+1} -> corrected c̃_{t+1} = ĉ_{t+1} + δc_{t+1}`

## Scope

- Только experiment layer для Navier-Stokes.
- Без ломки shared core API.
- С отдельными тестами, benchmark integration и qualitative diagnostics.

## Architecture

### 1. Base Model

Базой остаётся текущий лучший LSTM-path:

- `TrajectoryAwareLatentForecaster`
- `feature_mode="latent_plus_delta"`
- `delta_forecast=False`

### 2. Correction Features

Correction head получает:

- последнее model-state представление `z_t`
- baseline prediction `ẑ_{t+1}`
- разницу `ẑ_{t+1} - z_t`

Для `latent_plus_delta` в `z_t` уже находятся и `c_t`, и `Δc_t`.

### 3. Correction Target

Head не предсказывает полный augmented latent vector.

Он предсказывает только residual в пространстве абсолютных latent coefficients:

- target: `r_{t+1} = c_{t+1} - ĉ_{t+1}`

### 4. Training

- fit base forecaster
- freeze base forecaster
- построить residual dataset на train trajectories
- fit small MLP correction head
- early stopping по validation residual loss

### 5. Evaluation

- `evaluate_one_step` должен возвращать corrected metrics
- дополнительно сохранять baseline vs corrected predictions для diagnostics
- `evaluate_rollout` может использовать ту же correction logic autoregressively, но primary target этой надстройки — именно `t+1`

## Success Criteria

- unit tests pass
- новый wrapper воспроизводит baseline contract
- на реальном Navier-Stokes split one-step spatial `R²`/`RMSE` для corrected model лучше baseline LSTM
