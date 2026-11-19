from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PopulationZScore:
    mean: np.ndarray
    observed_std: np.ndarray
    scale: np.ndarray
    fit_row_count: int
    minimum_std: float
    small_std_indices: tuple[int, ...]

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = _validated_matrix(values, "values", self.mean.size)
        transformed = (array.astype(np.float64) - self.mean) / self.scale
        if not np.isfinite(transformed).all():
            raise ValueError("Z-score transform produced non-finite values.")
        return transformed.astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        array = _validated_matrix(values, "values", self.mean.size)
        restored = array.astype(np.float64) * self.scale + self.mean
        if not np.isfinite(restored).all():
            raise ValueError("Inverse z-score transform produced non-finite values.")
        return restored.astype(np.float32)

    def to_dict(self) -> dict[str, object]:
        return {
            "fit_partition": "train",
            "fit_row_count": self.fit_row_count,
            "mean": self.mean.tolist(),
            "observed_population_std": self.observed_std.tolist(),
            "effective_scale": self.scale.tolist(),
            "minimum_std": self.minimum_std,
            "small_std_indices": list(self.small_std_indices),
            "small_std_policy": (
                "Use effective scale 1.0 when the Train population standard "
                "deviation is below minimum_std; retain the Train mean."
            ),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> PopulationZScore:
        if value.get("fit_partition") != "train":
            raise ValueError("Normalization parameters must be fitted on Train.")
        mean = np.asarray(value["mean"], dtype=np.float64)
        observed_std = np.asarray(value["observed_population_std"], dtype=np.float64)
        scale = np.asarray(value["effective_scale"], dtype=np.float64)
        if mean.ndim != 1 or observed_std.shape != mean.shape or scale.shape != mean.shape:
            raise ValueError("Saved normalization parameter shapes do not match.")
        if not (
            np.isfinite(mean).all()
            and np.isfinite(observed_std).all()
            and np.isfinite(scale).all()
            and (scale > 0).all()
        ):
            raise ValueError("Saved normalization parameters are invalid.")
        return cls(
            mean=mean,
            observed_std=observed_std,
            scale=scale,
            fit_row_count=int(value["fit_row_count"]),
            minimum_std=float(value["minimum_std"]),
            small_std_indices=tuple(int(index) for index in value["small_std_indices"]),
        )


def fit_population_zscore(
    train_values: np.ndarray,
    *,
    minimum_std: float,
) -> PopulationZScore:
    if not np.isfinite(minimum_std) or minimum_std <= 0:
        raise ValueError("minimum_std must be a positive finite value.")
    array = _validated_matrix(train_values, "train_values")
    values = array.astype(np.float64)
    mean = values.mean(axis=0)
    observed_std = values.std(axis=0, ddof=0)
    small = observed_std < minimum_std
    scale = observed_std.copy()
    scale[small] = 1.0
    return PopulationZScore(
        mean=mean,
        observed_std=observed_std,
        scale=scale,
        fit_row_count=int(values.shape[0]),
        minimum_std=float(minimum_std),
        small_std_indices=tuple(int(index) for index in np.flatnonzero(small)),
    )


def apply_voltage_normalization(
    inputs: dict[str, np.ndarray],
    *,
    discharge_scaler: PopulationZScore | None,
    charge_scaler: PopulationZScore | None,
) -> dict[str, np.ndarray]:
    expected = {"X_discharge", "X_charge", "X_physics"}
    if set(inputs) != expected:
        raise ValueError(f"Expected inputs {sorted(expected)}, got {sorted(inputs)}.")
    transformed: dict[str, np.ndarray] = {}
    for name in ("X_discharge", "X_charge"):
        scaler = discharge_scaler if name == "X_discharge" else charge_scaler
        values = np.asarray(inputs[name])
        transformed[name] = (
            scaler.transform(values)
            if scaler is not None
            else np.array(values, copy=True)
        )
    transformed["X_physics"] = np.array(inputs["X_physics"], copy=True)
    if not all(np.isfinite(values).all() for values in transformed.values()):
        raise ValueError("Preprocessed model inputs contain non-finite values.")
    return transformed


def _validated_matrix(
    values: np.ndarray,
    name: str,
    expected_columns: int | None = None,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2-D array.")
    if expected_columns is not None and array.shape[1] != expected_columns:
        raise ValueError(
            f"{name} has {array.shape[1]} columns; expected {expected_columns}."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array
