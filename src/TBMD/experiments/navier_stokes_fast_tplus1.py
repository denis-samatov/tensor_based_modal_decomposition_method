"""Reusable fast t+1 Navier-Stokes forecaster using windowed TBMD + QR sensors."""

from __future__ import annotations

import contextlib
import io
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from TBMD.config import DecompositionConfig, SensorPlacementConfig
from TBMD.core.decomposition.hosvd import TuckerDecomposerInterface
from TBMD.core.sensor_placement import TensorTubeQRDecomposition
from TBMD.experiments.navier_stokes_forecasting import _compute_regression_metrics


@dataclass
class FastWindowedTBMDQRCSConfig:
    """Configuration for `FastWindowedTBMDQRCSForecaster`."""

    history_length: int = 7
    ranks: list[int] | tuple[int, int, int, int] = (8, 32, 32, 300)
    n_spatial_sensors: int = 300
    segment_stride: int = 1
    max_train_segments: Optional[int] = 6144
    sensor_rcond: float = 1e-6
    correction_alpha: float = 1e-8
    correction_scale: float = 1.0
    correction_residual_rank: Optional[int] = None
    random_state: int = 0
    spatial_mean_centering: bool = True
    dtype: str = "float32"


def build_forecast_segment_tensor(
    states: np.ndarray,
    *,
    history_length: int,
    stride: int,
    max_segments: int | None,
) -> np.ndarray:
    """Build causal forecasting segments as `(history+1,H,W,N_segments)`."""
    series = np.asarray(states, dtype=np.float64)
    if series.ndim != 4:
        raise ValueError("states must have shape `(B,T,H,W)`")
    if history_length <= 0 or stride <= 0:
        raise ValueError("history_length and stride must be positive")
    segment_length = history_length + 1
    if series.shape[1] < segment_length:
        raise ValueError("history_length + 1 exceeds trajectory length")

    refs = [
        (traj_idx, start)
        for traj_idx in range(series.shape[0])
        for start in range(0, series.shape[1] - segment_length + 1, stride)
    ]
    if max_segments is not None and len(refs) > max_segments:
        selected = np.linspace(0, len(refs) - 1, num=max_segments, dtype=int)
        refs = [refs[idx] for idx in selected]

    return np.stack(
        [series[traj_idx, start : start + segment_length] for traj_idx, start in refs],
        axis=-1,
    )


def _as_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def compute_segment_dictionary_from_tucker(
    core: np.ndarray | torch.Tensor,
    factors: list[np.ndarray | torch.Tensor],
) -> np.ndarray:
    """Convert Tucker decomposition of `(S,H,W,N)` segments to `(S,H,W,R)` modes."""
    core_np = _as_numpy(core)
    factor_np = [_as_numpy(factor) for factor in factors]
    if core_np.ndim != 4 or len(factor_np) != 4:
        raise ValueError("Expected 4D core and four factors for `(S,H,W,N)`")
    u_tau, u_x, u_y, _ = factor_np
    return np.einsum("ta,xb,yc,abcq->txyq", u_tau, u_x, u_y, core_np, optimize=True)


def fit_segment_dictionary(
    train_segments: np.ndarray,
    *,
    ranks: list[int] | tuple[int, int, int, int],
    random_state: int,
    dtype: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    config = DecompositionConfig(
        ranks=list(ranks),
        epsilon=1e-5,
        random_state=random_state,
        device="cpu",
        dtype=dtype,
        verbose=False,
        max_workers=1,
    )
    decomposer = TuckerDecomposerInterface(train_segments.astype(np.float32), config=config)
    decomposer.decompose()
    decomposer.reconstruct()
    dictionary = compute_segment_dictionary_from_tucker(
        decomposer.core_tensor,
        decomposer.factors,
    )
    return dictionary.astype(np.float64), {
        "core_shape": list(_as_numpy(decomposer.core_tensor).shape),
        "factor_shapes": [list(_as_numpy(factor).shape) for factor in decomposer.factors],
        "segment_reconstruction_rel_frob": float(decomposer.reconstruction_errors),
    }


def history_and_target(dictionary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if dictionary.shape[0] < 2:
        raise ValueError("dictionary must include history and target slices")
    return dictionary[:-1], dictionary[-1]


def augment_spatial_mask_by_leverage(
    qr_tensor: np.ndarray,
    spatial_mask: np.ndarray,
    *,
    target_sensors: int,
) -> np.ndarray:
    flat_mask = spatial_mask.reshape(-1).copy()
    n_extra = min(target_sensors - int(flat_mask.sum()), flat_mask.size - int(flat_mask.sum()))
    if n_extra <= 0:
        return flat_mask.reshape(spatial_mask.shape)
    leverage = np.sum(qr_tensor.reshape(-1, qr_tensor.shape[-1]) ** 2, axis=1)
    leverage[flat_mask] = -np.inf
    extra_indices = np.argpartition(-leverage, n_extra - 1)[:n_extra]
    extra_indices = extra_indices[np.argsort(-leverage[extra_indices])]
    flat_mask[extra_indices] = True
    return flat_mask.reshape(spatial_mask.shape)


def place_fixed_spatial_sensors(
    history_dictionary: np.ndarray,
    *,
    n_spatial_sensors: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select fixed spatial sensors using all history-time dictionary tubes."""
    if n_spatial_sensors <= 0:
        raise ValueError("n_spatial_sensors must be positive")
    qr_tensor = history_dictionary.transpose(1, 2, 0, 3).reshape(
        history_dictionary.shape[1],
        history_dictionary.shape[2],
        history_dictionary.shape[0] * history_dictionary.shape[-1],
    )
    core_sensor_count = min(n_spatial_sensors, qr_tensor.shape[-1])
    config = SensorPlacementConfig(
        n_sensors=core_sensor_count,
        random_state=random_state,
        verbose=False,
        dtype="float64",
    )
    qr = TensorTubeQRDecomposition(
        qr_tensor,
        N=core_sensor_count,
        config=config,
        dtype=torch.float64,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        mask, _, _ = qr.factorize()
    spatial_mask = mask.detach().cpu().numpy().astype(bool)
    if n_spatial_sensors > int(spatial_mask.sum()):
        spatial_mask = augment_spatial_mask_by_leverage(
            qr_tensor,
            spatial_mask,
            target_sensors=n_spatial_sensors,
        )
    return spatial_mask, np.flatnonzero(spatial_mask.reshape(-1))


def predict_from_history_sensor_lstsq(
    history: np.ndarray,
    dictionary: np.ndarray,
    spatial_sensor_indices: np.ndarray,
    *,
    rcond: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict centered next frames from centered history shaped `(N,L,H,W)`."""
    history_dictionary, target_dictionary = history_and_target(dictionary)
    if history.shape[1:] != history_dictionary.shape[:-1]:
        raise ValueError("history shape must match dictionary history slices")
    height_width = int(np.prod(history_dictionary.shape[1:3]))
    flat_history_dictionary = history_dictionary.reshape(history_dictionary.shape[0], height_width, -1)
    sensor_dictionary = flat_history_dictionary[:, spatial_sensor_indices, :].reshape(
        -1,
        history_dictionary.shape[-1],
    )
    flat_history = history.transpose(1, 2, 3, 0).reshape(
        history_dictionary.shape[0],
        height_width,
        history.shape[0],
    )
    measurements = flat_history[:, spatial_sensor_indices, :].reshape(-1, history.shape[0]).T
    coeffs = measurements @ np.linalg.pinv(sensor_dictionary, rcond=rcond).T
    predictions = coeffs @ target_dictionary.reshape(-1, target_dictionary.shape[-1]).T
    return predictions.reshape(coeffs.shape[0], *target_dictionary.shape[:-1]), coeffs


def predict_next_sensor_lstsq(
    segments: np.ndarray,
    dictionary: np.ndarray,
    spatial_sensor_indices: np.ndarray,
    *,
    rcond: float,
) -> tuple[np.ndarray, np.ndarray]:
    history = segments[:-1].transpose(3, 0, 1, 2)
    return predict_from_history_sensor_lstsq(
        history,
        dictionary,
        spatial_sensor_indices,
        rcond=rcond,
    )


def target_frames_from_segments(segments: np.ndarray) -> np.ndarray:
    return segments[-1].transpose(2, 0, 1)


def fit_ridge_residual_corrector(
    target_frames: np.ndarray,
    base_predictions: np.ndarray,
    coeffs: np.ndarray,
    *,
    alpha: float,
    residual_rank: Optional[int] = None,
) -> dict[str, Any]:
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    target_flat = np.asarray(target_frames, dtype=np.float64).reshape(target_frames.shape[0], -1)
    base_flat = np.asarray(base_predictions, dtype=np.float64).reshape(base_predictions.shape[0], -1)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    features = np.concatenate([coeffs, np.ones((coeffs.shape[0], 1), dtype=np.float64)], axis=1)
    residual = target_flat - base_flat
    residual_basis = None
    residual_mean = None
    regression_target = residual
    actual_residual_rank = None
    if residual_rank is not None:
        if residual_rank <= 0:
            raise ValueError("residual_rank must be positive when provided")
        actual_residual_rank = min(int(residual_rank), residual.shape[0], residual.shape[1])
        residual_mean = residual.mean(axis=0)
        centered_residual = residual - residual_mean
        _, _, vt = np.linalg.svd(centered_residual, full_matrices=False)
        residual_basis = vt[:actual_residual_rank]
        regression_target = centered_residual @ residual_basis.T
    gram = features.T @ features
    penalty = alpha * np.eye(features.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    rhs = features.T @ regression_target
    try:
        weights = np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(gram + penalty, rhs, rcond=None)[0]
    corrector = {
        "alpha": float(alpha),
        "weights": weights,
        "feature_dim": int(coeffs.shape[1]),
        "output_dim": int(target_flat.shape[1]),
        "mode": "residual_svd" if residual_basis is not None else "full",
        "residual_rank": actual_residual_rank,
    }
    if residual_basis is not None:
        corrector["residual_basis"] = residual_basis
        corrector["residual_mean"] = residual_mean
    return corrector


def apply_ridge_residual_corrector(
    base_predictions: np.ndarray,
    coeffs: np.ndarray,
    corrector: dict[str, Any],
    *,
    scale: float,
) -> np.ndarray:
    base_flat = np.asarray(base_predictions, dtype=np.float64).reshape(base_predictions.shape[0], -1)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    features = np.concatenate([coeffs, np.ones((coeffs.shape[0], 1), dtype=np.float64)], axis=1)
    correction = features @ np.asarray(corrector["weights"], dtype=np.float64)
    if corrector.get("mode") == "residual_svd":
        correction = (
            np.asarray(corrector["residual_mean"], dtype=np.float64)
            + correction @ np.asarray(corrector["residual_basis"], dtype=np.float64)
        )
    return (base_flat + scale * correction).reshape(base_predictions.shape)


class FastWindowedTBMDQRCSForecaster:
    """Fast one-step predictor based on windowed TBMD, QR sensors, and a small head."""

    def __init__(self, config: Optional[FastWindowedTBMDQRCSConfig] = None):
        self.config = config or FastWindowedTBMDQRCSConfig()
        self._spatial_mean: Optional[np.ndarray] = None
        self._dictionary: Optional[np.ndarray] = None
        self._spatial_mask: Optional[np.ndarray] = None
        self._spatial_sensor_indices: Optional[np.ndarray] = None
        self._coefficient_corrector: Optional[dict[str, Any]] = None
        self._metrics: dict[str, Any] = {}
        self._fitted = False

    def fit(self, train_states: np.ndarray) -> "FastWindowedTBMDQRCSForecaster":
        states = np.asarray(train_states, dtype=np.float64)
        if states.ndim != 4:
            raise ValueError("train_states must have shape `(B,T,H,W)`")
        self._spatial_mean = (
            np.mean(states, axis=(0, 1))
            if self.config.spatial_mean_centering
            else np.zeros(states.shape[2:], dtype=np.float64)
        )
        centered = states - self._spatial_mean
        segments = build_forecast_segment_tensor(
            centered,
            history_length=self.config.history_length,
            stride=self.config.segment_stride,
            max_segments=self.config.max_train_segments,
        )
        dictionary, tbmd_summary = fit_segment_dictionary(
            segments,
            ranks=list(self.config.ranks),
            random_state=self.config.random_state,
            dtype=self.config.dtype,
        )
        history_dictionary, _ = history_and_target(dictionary)
        spatial_mask, spatial_sensor_indices = place_fixed_spatial_sensors(
            history_dictionary,
            n_spatial_sensors=self.config.n_spatial_sensors,
            random_state=self.config.random_state,
        )
        base_pred, coeffs = predict_next_sensor_lstsq(
            segments,
            dictionary,
            spatial_sensor_indices,
            rcond=self.config.sensor_rcond,
        )
        targets = target_frames_from_segments(segments)
        corrector = fit_ridge_residual_corrector(
            targets,
            base_pred,
            coeffs,
            alpha=self.config.correction_alpha,
            residual_rank=self.config.correction_residual_rank,
        )
        corrected = apply_ridge_residual_corrector(
            base_pred,
            coeffs,
            corrector,
            scale=self.config.correction_scale,
        )
        self._dictionary = dictionary
        self._spatial_mask = spatial_mask
        self._spatial_sensor_indices = spatial_sensor_indices
        self._coefficient_corrector = corrector
        self._metrics = {
            "fit": {
                "n_train_segments": int(segments.shape[-1]),
                "train_base": _compute_regression_metrics(targets, base_pred),
                "train_corrected": _compute_regression_metrics(targets, corrected),
                "tbmd_summary": tbmd_summary,
                "dictionary_shape": list(dictionary.shape),
                "actual_spatial_sensors": int(spatial_mask.sum()),
                "total_history_measurements_per_prediction": int(
                    spatial_mask.sum() * self.config.history_length
                ),
            }
        }
        self._fitted = True
        return self

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() or load() before prediction")

    def predict_next(self, history_states: np.ndarray) -> np.ndarray:
        pred, _ = self.predict_next_with_coefficients(history_states)
        return pred

    def predict_next_with_coefficients(self, history_states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._require_fitted()
        history = np.asarray(history_states, dtype=np.float64)
        if history.ndim != 4:
            raise ValueError("history_states must have shape `(B,L,H,W)`")
        if history.shape[1:] != self._dictionary[:-1].shape[:-1]:
            raise ValueError("history_states must match fitted history shape")
        centered_history = history - self._spatial_mean
        base_pred, coeffs = predict_from_history_sensor_lstsq(
            centered_history,
            self._dictionary,
            self._spatial_sensor_indices,
            rcond=self.config.sensor_rcond,
        )
        corrected = apply_ridge_residual_corrector(
            base_pred,
            coeffs,
            self._coefficient_corrector,
            scale=self.config.correction_scale,
        )
        return corrected + self._spatial_mean, coeffs

    def evaluate_one_step(self, test_states: np.ndarray) -> dict[str, Any]:
        self._require_fitted()
        states = np.asarray(test_states, dtype=np.float64)
        centered = states - self._spatial_mean
        segments = build_forecast_segment_tensor(
            centered,
            history_length=self.config.history_length,
            stride=self.config.segment_stride,
            max_segments=None,
        )
        history = states[:, : self.config.history_length]
        predictions = []
        targets = []
        for start in range(0, states.shape[1] - self.config.history_length):
            history = states[:, start : start + self.config.history_length]
            pred = self.predict_next(history)
            predictions.append(pred)
            targets.append(states[:, start + self.config.history_length])
        pred_spatial = np.concatenate(predictions, axis=0)
        target_spatial = np.concatenate(targets, axis=0)
        metrics = _compute_regression_metrics(target_spatial, pred_spatial)
        payload = {
            "spatial_mse": metrics["mse"],
            "spatial_rmse": metrics["rmse"],
            "spatial_rel_frob_err": metrics["rel_frob_err"],
            "spatial_r2": metrics["r2"],
            "n_eval_samples": int(target_spatial.shape[0]),
            "target_spatial": target_spatial,
            "pred_spatial": pred_spatial,
            "segments_shape": list(segments.shape),
        }
        self._metrics["last_one_step"] = {
            key: value for key, value in payload.items() if key not in {"target_spatial", "pred_spatial"}
        }
        return payload

    def save(self, path: str | Path) -> None:
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            config=np.asarray([asdict(self.config)], dtype=object),
            spatial_mean=self._spatial_mean.astype(np.float32),
            dictionary=self._dictionary.astype(np.float32),
            spatial_mask=self._spatial_mask.astype(bool),
            spatial_sensor_indices=self._spatial_sensor_indices.astype(np.int64),
            coefficient_corrector_weights=self._coefficient_corrector["weights"].astype(np.float32),
            coefficient_corrector_alpha=np.asarray(self._coefficient_corrector["alpha"], dtype=np.float64),
            coefficient_corrector_mode=np.asarray(self._coefficient_corrector.get("mode", "full")),
            coefficient_corrector_residual_rank=np.asarray(
                -1
                if self._coefficient_corrector.get("residual_rank") is None
                else self._coefficient_corrector["residual_rank"],
                dtype=np.int64,
            ),
            coefficient_corrector_residual_basis=np.asarray(
                self._coefficient_corrector.get("residual_basis", np.empty((0, 0))),
                dtype=np.float32,
            ),
            coefficient_corrector_residual_mean=np.asarray(
                self._coefficient_corrector.get("residual_mean", np.empty((0,))),
                dtype=np.float32,
            ),
            metrics=np.asarray([self._metrics], dtype=object),
        )

    @classmethod
    def load(cls, path: str | Path) -> "FastWindowedTBMDQRCSForecaster":
        data = np.load(Path(path), allow_pickle=True)
        config_payload = data["config"][0]
        if hasattr(config_payload, "item"):
            config_payload = config_payload.item()
        config = FastWindowedTBMDQRCSConfig(**config_payload)
        model = cls(config)
        model._spatial_mean = data["spatial_mean"].astype(np.float64)
        model._dictionary = data["dictionary"].astype(np.float64)
        model._spatial_mask = data["spatial_mask"].astype(bool)
        model._spatial_sensor_indices = data["spatial_sensor_indices"].astype(int)
        weights = data["coefficient_corrector_weights"].astype(np.float64)
        mode = "full"
        if "coefficient_corrector_mode" in data:
            mode = str(data["coefficient_corrector_mode"].item())
        model._coefficient_corrector = {
            "alpha": float(data["coefficient_corrector_alpha"]),
            "weights": weights,
            "feature_dim": int(weights.shape[0] - 1),
            "output_dim": int(np.prod(model._dictionary.shape[1:3])),
            "mode": mode,
            "residual_rank": None,
        }
        if mode == "residual_svd":
            residual_rank = int(data["coefficient_corrector_residual_rank"])
            model._coefficient_corrector["residual_rank"] = residual_rank
            model._coefficient_corrector["residual_basis"] = data[
                "coefficient_corrector_residual_basis"
            ].astype(np.float64)
            model._coefficient_corrector["residual_mean"] = data[
                "coefficient_corrector_residual_mean"
            ].astype(np.float64)
        metrics_payload = data["metrics"][0]
        if hasattr(metrics_payload, "item"):
            metrics_payload = metrics_payload.item()
        model._metrics = metrics_payload
        model._fitted = True
        return model

    def get_config(self) -> dict[str, Any]:
        return asdict(self.config)

    def get_metrics(self) -> dict[str, Any]:
        return dict(self._metrics)


__all__ = [
    "FastWindowedTBMDQRCSConfig",
    "FastWindowedTBMDQRCSForecaster",
    "build_forecast_segment_tensor",
    "predict_from_history_sensor_lstsq",
]
