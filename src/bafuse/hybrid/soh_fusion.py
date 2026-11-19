from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence

import torch
from torch import nn


MODEL_VERSION = "bafuse-three-encoder-v1"
INPUT_SCHEMA_VERSION = "bafuse-physics-v1"
PHYSICS_FEATURE_NAMES = (
    "mean_discharge_temperature_native",
    "discharge_temperature_rise_native",
    "discharge_plateau_slope_v_per_progress",
    "discharge_slope_variability_v_per_progress",
    "mean_absolute_discharge_current_native",
)
EXCLUDED_FEATURES = {
    "step4_discharge_duration": (
        "Excluded from model input because constant-current duration can "
        "reconstruct same-RPT discharged capacity and leak the SOH target."
    )
}


def resample_curve(values: Sequence[float], n_points: int = 128) -> list[float]:
    if n_points < 2 or len(values) < 2:
        raise ValueError("Curve resampling requires at least two input and output points.")
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("Curve contains a non-finite voltage value.")
    last = len(values) - 1
    result: list[float] = []
    for index in range(n_points):
        position = index * last / (n_points - 1)
        lower = math.floor(position)
        upper = min(lower + 1, last)
        fraction = position - lower
        result.append(
            float(values[lower]) * (1 - fraction)
            + float(values[upper]) * fraction
        )
    return result


def _linear_slope(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise ValueError("A linear slope requires two equally sized sequences.")
    x_mean = statistics.mean(x_values)
    y_mean = statistics.mean(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if denominator <= 0:
        raise ValueError("Slope progress values have no variation.")
    return sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(x_values, y_values)
    ) / denominator


def build_physics_features(
    step4_voltage: Sequence[float],
    step4_current: Sequence[float],
    step4_temperature: Sequence[float],
) -> dict[str, float]:
    if (
        len(step4_voltage) == 0
        or len(step4_current) == 0
        or len(step4_temperature) == 0
    ):
        raise ValueError("All approved Step 4 physics measurements are required.")
    measurements = (*step4_voltage, *step4_current, *step4_temperature)
    if any(not math.isfinite(float(value)) for value in measurements):
        raise ValueError("Physics measurements contain non-finite values.")

    voltage = resample_curve(step4_voltage)
    progress = [index / (len(voltage) - 1) for index in range(len(voltage))]
    plateau_indices = [
        index for index, value in enumerate(progress) if 0.20 <= value <= 0.80
    ]
    plateau_slope = _linear_slope(
        [progress[index] for index in plateau_indices],
        [voltage[index] for index in plateau_indices],
    )
    derivatives = [
        (voltage[index + 1] - voltage[index - 1])
        / (progress[index + 1] - progress[index - 1])
        for index in range(1, len(voltage) - 1)
        if 0.10 <= progress[index] <= 0.90
    ]
    derivative_median = statistics.median(derivatives)
    slope_variability = 1.4826 * statistics.median(
        abs(value - derivative_median) for value in derivatives
    )

    return {
        "mean_discharge_temperature_native": statistics.mean(
            float(value) for value in step4_temperature
        ),
        "discharge_temperature_rise_native": max(
            float(value) for value in step4_temperature
        )
        - float(step4_temperature[0]),
        "discharge_plateau_slope_v_per_progress": plateau_slope,
        "discharge_slope_variability_v_per_progress": slope_variability,
        "mean_absolute_discharge_current_native": statistics.mean(
            abs(float(value)) for value in step4_current
        ),
    }


class CurveEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        if curve.ndim != 2 or curve.shape[1] != 128:
            raise ValueError(
                f"Expected curve input with shape (batch, 128), got {tuple(curve.shape)}"
            )
        return self.layers(curve.unsqueeze(1))


class ThreeEncoderSoH(nn.Module):
    def __init__(
        self,
        n_physics_features: int = len(PHYSICS_FEATURE_NAMES),
        freeze_discharge: bool = False,
    ) -> None:
        super().__init__()
        if n_physics_features != len(PHYSICS_FEATURE_NAMES):
            raise ValueError(
                f"Expected {len(PHYSICS_FEATURE_NAMES)} physics features, "
                f"got {n_physics_features}"
            )
        self.discharge_encoder = CurveEncoder()
        self.charge_encoder = CurveEncoder()
        self.physics_encoder = nn.Sequential(
            nn.Linear(n_physics_features, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
        )
        self.regression_head = nn.Sequential(
            nn.Linear(96, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        if freeze_discharge:
            for parameter in self.discharge_encoder.parameters():
                parameter.requires_grad = False

    def initialize_discharge_encoder(
        self, checkpoint_state: Mapping[str, torch.Tensor]
    ) -> tuple[str, ...]:
        target = self.discharge_encoder.state_dict()
        keys = (
            "layers.0.weight",
            "layers.0.bias",
            "layers.3.weight",
            "layers.3.bias",
        )
        with torch.no_grad():
            for key in keys:
                source_value = checkpoint_state.get(key)
                if source_value is None or source_value.shape != target[key].shape:
                    raise ValueError(
                        f"Historical CNN checkpoint is incompatible at {key}."
                    )
                target[key].copy_(source_value.detach().to(target[key].device))
        return keys

    def forward(
        self,
        discharge: torch.Tensor,
        charge: torch.Tensor,
        physics: torch.Tensor,
    ) -> torch.Tensor:
        if charge.ndim != 2 or charge.shape[1] != 128:
            raise ValueError(
                f"Expected charge input with shape (batch, 128), got {tuple(charge.shape)}"
            )
        if physics.ndim != 2 or physics.shape[1] != len(PHYSICS_FEATURE_NAMES):
            raise ValueError(
                f"Expected physics input with shape "
                f"(batch, {len(PHYSICS_FEATURE_NAMES)}), got {tuple(physics.shape)}"
            )
        if not (discharge.shape[0] == charge.shape[0] == physics.shape[0]):
            raise ValueError("All modalities must have the same batch size.")
        encoded = torch.cat(
            (
                self.discharge_encoder(discharge),
                self.charge_encoder(charge),
                self.physics_encoder(physics),
            ),
            dim=1,
        )
        return self.regression_head(encoded)
