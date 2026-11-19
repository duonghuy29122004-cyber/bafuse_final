from __future__ import annotations

import math
import runpy
import statistics
from pathlib import Path

import numpy as np
import pytest

from bafuse.hybrid.p5_electrical_features import (
    P5_FEATURE_NAMES,
    build_p5_electrical_features,
)

P5_RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "soh_forecast"
    / "bafuse_architecture_benchmark_v1"
    / "bafuse_p5_electrical_v1_20261114"
    / "run_p5_dataset.py"
)


def test_p5_features_have_fixed_order_and_expected_linear_curve_values():
    voltage = [4.2 - 1.7 * index / 256 for index in range(257)]
    current = [-1.5] * 257

    first = build_p5_electrical_features(voltage, current)
    second = build_p5_electrical_features(voltage, current)

    assert tuple(first) == P5_FEATURE_NAMES
    assert first == second
    assert first["mean_abs_discharge_current"] == pytest.approx(1.5)
    assert first["discharge_current_variability"] == 0.0
    assert first["discharge_voltage_slope_20_80"] == pytest.approx(-1.7)
    assert first["discharge_voltage_derivative_mad_10_90"] == pytest.approx(
        0.0, abs=1e-12
    )
    progress = [index / 127 for index in range(128)]
    early = [value for value in progress if 0.10 <= value <= 0.30]
    late = [value for value in progress if 0.70 <= value <= 0.90]
    assert first["discharge_voltage_segment_contrast_10_30_vs_70_90"] == pytest.approx(
        1.7 * (statistics.mean(late) - statistics.mean(early))
    )
    assert all(math.isfinite(value) for value in first.values())


def test_p5_current_variability_uses_robust_mad_and_remains_nonnegative():
    current = [-1.5] * 99 + [-1.8]
    features = build_p5_electrical_features(
        [4.2 - index / 99 for index in range(100)],
        current,
    )

    assert features["discharge_current_variability"] == 0.0
    assert features["mean_abs_discharge_current"] > 1.5


def test_p5_rejects_misaligned_or_nonfinite_raw_measurements():
    with pytest.raises(ValueError, match="row-aligned"):
        build_p5_electrical_features([4.2, 3.9], [-1.5, -1.4, -1.3])

    with pytest.raises(ValueError, match="non-finite"):
        build_p5_electrical_features([4.2, float("nan")], [-1.5, -1.4])

    with pytest.raises(ValueError, match="non-finite"):
        build_p5_electrical_features([4.2, 3.9], [-1.5, float("inf")])


def test_p5_scaler_fits_only_train_and_npz_writer_is_deterministic(tmp_path):
    runner = runpy.run_path(str(P5_RUNNER_PATH))
    values = np.asarray(
        [[1.0, 10.0]] * 51
        + [[3.0, 14.0]] * 51
        + [[100.0, -50.0]]
    )
    scaled, means, scales, constant = runner["_fit_train_scaler"](
        values, ["train"] * 102 + ["external_test"]
    )

    assert means == pytest.approx([2.0, 12.0])
    assert scales == pytest.approx([1.0, 2.0])
    assert not constant.any()
    assert scaled[102].tolist() == pytest.approx([98.0, -31.0])

    arrays = {
        "X_discharge": np.zeros((2, 128), dtype=np.float32),
        "X_charge": np.ones((2, 128), dtype=np.float32),
        "X_physics": np.full((2, 5), 2.0, dtype=np.float32),
        "y_soh": np.full((2, 1), 95.0, dtype=np.float32),
    }
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    runner["_write_deterministic_npz"](first, arrays)
    runner["_write_deterministic_npz"](second, arrays)

    assert first.read_bytes() == second.read_bytes()
    with np.load(first, allow_pickle=False) as archive:
        assert archive["X_discharge"].shape == (2, 128)
        assert archive["X_physics"].shape == (2, 5)
