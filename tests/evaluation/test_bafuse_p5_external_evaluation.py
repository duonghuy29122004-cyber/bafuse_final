from __future__ import annotations

import copy

import numpy as np
import pytest

from bafuse.hybrid.p5_external_evaluation import (
    DEFAULT_CONFIG,
    _metrics,
    _residual_statistics,
    _validate_config,
    _read_json,
)


def test_external_protocol_locks_authorization_and_selection() -> None:
    config = _read_json(DEFAULT_CONFIG)

    _validate_config(config)

    changed = copy.deepcopy(config)
    changed["authorization"]["training_authorized"] = True
    with pytest.raises(ValueError, match="authorization flags changed"):
        _validate_config(changed)

    changed = copy.deepcopy(config)
    changed["predeclared_validation_selections"]["external_results_may_change_selection"] = True
    with pytest.raises(ValueError, match="selection protocol changed"):
        _validate_config(changed)


def test_external_metrics_use_percentage_point_errors_and_residual_direction() -> None:
    target = np.asarray([90.0, 100.0])
    prediction = np.asarray([92.0, 97.0])

    metrics = _metrics(target, prediction)
    residuals = _residual_statistics(target, prediction)

    assert metrics["MAE"] == pytest.approx(2.5)
    assert metrics["RMSE"] == pytest.approx(np.sqrt(6.5))
    assert metrics["R2"] == pytest.approx(0.74)
    assert residuals["mean_residual_pp"] == pytest.approx(-0.5)
    assert residuals["maximum_absolute_error_pp"] == pytest.approx(3.0)


def test_external_r2_rejects_zero_variance_reference() -> None:
    with pytest.raises(ValueError, match="zero variance"):
        _metrics(np.asarray([95.0, 95.0]), np.asarray([94.0, 96.0]))


def test_external_metric_recalculation_preserves_tensor_precision() -> None:
    target = np.asarray([89.123456, 95.234567, 100.0], dtype=np.float32)
    prediction = np.asarray([90.123451, 94.234563, 99.765432], dtype=np.float32)
    saved_target = np.asarray([str(value) for value in target], dtype=np.float32)
    saved_prediction = np.asarray([str(value) for value in prediction], dtype=np.float32)

    assert _metrics(saved_target, saved_prediction) == _metrics(target, prediction)
