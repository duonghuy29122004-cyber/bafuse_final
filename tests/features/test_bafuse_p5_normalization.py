import numpy as np
import pytest

from bafuse.hybrid.p5_normalization import (
    PopulationZScore,
    apply_voltage_normalization,
    fit_population_zscore,
)


def test_population_zscore_uses_train_population_statistics_and_small_scale_policy():
    train = np.asarray(
        [
            [1.0, 7.0, 3.0],
            [3.0, 7.0, 3.0000001],
            [5.0, 7.0, 2.9999999],
        ],
        dtype=np.float64,
    )

    scaler = fit_population_zscore(train, minimum_std=1e-6)

    np.testing.assert_allclose(scaler.mean, train.mean(axis=0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(scaler.observed_std, train.std(axis=0, ddof=0), atol=1e-12)
    np.testing.assert_allclose(scaler.scale, [np.sqrt(8 / 3), 1.0, 1.0], atol=1e-12)
    assert scaler.fit_row_count == 3
    assert scaler.small_std_indices == (1, 2)

    validation = np.asarray([[7.0, 7.0, 3.0]])
    expected = (validation - scaler.mean) / scaler.scale
    np.testing.assert_allclose(scaler.transform(validation), expected, atol=1e-6)


def test_target_inverse_transform_round_trips_with_train_parameters():
    train_target = np.asarray([[80.0], [90.0], [100.0]], dtype=np.float32)
    scaler = fit_population_zscore(train_target, minimum_std=1e-8)
    normalized = scaler.transform(train_target)

    np.testing.assert_allclose(
        scaler.inverse_transform(normalized),
        train_target,
        rtol=0,
        atol=1e-5,
    )


def test_discharge_and_charge_use_separate_scalers_and_physics_is_unchanged():
    discharge = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    charge = np.asarray([[10.0, 20.0], [14.0, 24.0]], dtype=np.float32)
    physics = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    discharge_scaler = fit_population_zscore(discharge, minimum_std=1e-6)
    charge_scaler = fit_population_zscore(charge, minimum_std=1e-6)

    result = apply_voltage_normalization(
        {
            "X_discharge": discharge,
            "X_charge": charge,
            "X_physics": physics,
        },
        discharge_scaler=discharge_scaler,
        charge_scaler=charge_scaler,
    )

    assert result["X_discharge"].shape == (2, 2)
    assert result["X_charge"].shape == (2, 2)
    np.testing.assert_allclose(result["X_discharge"].mean(axis=0), 0.0, atol=1e-7)
    np.testing.assert_allclose(result["X_charge"].mean(axis=0), 0.0, atol=1e-7)
    np.testing.assert_array_equal(result["X_physics"], physics)
    assert not np.shares_memory(result["X_physics"], physics)


def test_saved_scaler_parameters_reconstruct_identical_transform():
    train = np.asarray([[2.0, 10.0], [4.0, 14.0], [6.0, 18.0]])
    fitted = fit_population_zscore(train, minimum_std=1e-6)
    restored = PopulationZScore.from_dict(fitted.to_dict())

    np.testing.assert_array_equal(
        restored.transform(train),
        fitted.transform(train),
    )


def test_normalization_rejects_non_finite_data():
    with pytest.raises(ValueError, match="non-finite"):
        fit_population_zscore(np.asarray([[1.0], [np.nan]]), minimum_std=1e-6)
