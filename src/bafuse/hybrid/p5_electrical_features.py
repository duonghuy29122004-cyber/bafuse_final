from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

from bafuse.hybrid.soh_fusion import resample_curve


P5_FEATURE_NAMES = (
    "mean_abs_discharge_current",
    "discharge_current_variability",
    "discharge_voltage_slope_20_80",
    "discharge_voltage_derivative_mad_10_90",
    "discharge_voltage_segment_contrast_10_30_vs_70_90",
)


def _linear_slope(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise ValueError("A linear slope requires two equally sized sequences.")
    x_mean = statistics.mean(x_values)
    y_mean = statistics.mean(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if denominator <= 0:
        raise ValueError("Slope progress values have no variation.")
    return sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values)
    ) / denominator


def build_p5_electrical_features(
    step4_voltage: Sequence[float],
    step4_current: Sequence[float],
) -> dict[str, float]:
    if len(step4_voltage) < 2 or len(step4_current) < 2:
        raise ValueError("P5 extraction requires at least two voltage and current samples.")
    if len(step4_voltage) != len(step4_current):
        raise ValueError("Step 4 voltage and current samples must be row-aligned.")
    if any(not math.isfinite(float(value)) for value in step4_current):
        raise ValueError("Step 4 current contains a non-finite value.")

    voltage = resample_curve(step4_voltage, n_points=128)
    current = [float(value) for value in step4_current]
    progress = [index / 127 for index in range(128)]

    central_indices = [
        index for index, value in enumerate(progress) if 0.20 <= value <= 0.80
    ]
    slope = _linear_slope(
        [progress[index] for index in central_indices],
        [voltage[index] for index in central_indices],
    )

    derivatives = [
        (voltage[index + 1] - voltage[index - 1])
        / (progress[index + 1] - progress[index - 1])
        for index in range(1, 127)
        if 0.10 <= progress[index] <= 0.90
    ]
    derivative_median = statistics.median(derivatives)
    derivative_mad = 1.4826 * statistics.median(
        abs(value - derivative_median) for value in derivatives
    )

    early_indices = [
        index for index, value in enumerate(progress) if 0.10 <= value <= 0.30
    ]
    late_indices = [
        index for index, value in enumerate(progress) if 0.70 <= value <= 0.90
    ]
    current_median = statistics.median(current)
    current_mad = 1.4826 * statistics.median(
        abs(value - current_median) for value in current
    )

    features = {
        "mean_abs_discharge_current": statistics.mean(abs(value) for value in current),
        "discharge_current_variability": current_mad,
        "discharge_voltage_slope_20_80": slope,
        "discharge_voltage_derivative_mad_10_90": derivative_mad,
        "discharge_voltage_segment_contrast_10_30_vs_70_90": (
            statistics.mean(voltage[index] for index in early_indices)
            - statistics.mean(voltage[index] for index in late_indices)
        ),
    }
    if tuple(features) != P5_FEATURE_NAMES:
        raise AssertionError("P5 feature order does not match the public schema.")
    if any(not math.isfinite(value) for value in features.values()):
        raise ValueError("P5 extraction produced a non-finite feature.")
    return features
