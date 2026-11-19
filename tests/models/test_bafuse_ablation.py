import numpy as np
import pytest
import torch

from experiments.soh_forecast.run_ablation_phase3 import (
    AblationSoH,
    CONFIGURATIONS,
    PHYSICS_FEATURE_NAMES,
    _contribution_rows,
    _fit_scaler,
    _metrics,
    _seed_summary,
    _training_failure_rows,
    _train_one,
)


@pytest.mark.parametrize("config_id", tuple(CONFIGURATIONS))
def test_ablation_model_uses_only_configured_modalities(config_id):
    modalities = CONFIGURATIONS[config_id]
    model = AblationSoH(modalities)
    inputs = {
        name: torch.zeros(
            (2, len(PHYSICS_FEATURE_NAMES) if name == "physics" else 128)
        )
        for name in modalities
    }

    assert model(inputs).shape == (2, 1)
    assert tuple(model.encoders) == modalities

    with pytest.raises(ValueError, match="Expected inputs"):
        model({})


def test_overall_metrics_use_all_observations():
    actual = np.asarray([80.0, 100.0])
    predicted = np.asarray([82.0, 97.0])

    metrics = _metrics(actual, predicted)

    assert metrics["mae_soh_percentage_points"] == pytest.approx(2.5)
    assert metrics["rmse_soh_percentage_points"] == pytest.approx(np.sqrt(6.5))
    assert metrics["r2"] == pytest.approx(1.0 - 13.0 / 200.0)


def test_contribution_differences_are_paired_and_positive_for_better_full_model():
    rows = []
    for seed in (211, 212, 213):
        for partition in ("train", "validation", "external_test"):
            for config_id in ("M2", "M3", "M4", "M5"):
                for part, cell, rpt, actual in (
                    ("1", "CellA", "1", 80.0),
                    ("1", "CellA", "2", 100.0),
                ):
                    prediction = actual if config_id == "M4" else actual + 2.0
                    rows.append(
                        {
                            "model_configuration": config_id,
                            "training_seed": seed,
                            "partition": partition,
                            "dataset_part": part,
                            "physical_cell_id": cell,
                            "measurement_id": f"RPT{rpt}",
                            "reference_soh_percent": actual,
                            "predicted_soh_percent": prediction,
                        }
                    )

    contributions = _contribution_rows(rows)

    assert len(contributions) == 27
    assert all(
        row["delta_mae_soh_percentage_points"] == pytest.approx(2.0)
        for row in contributions
    )
    assert all(row["paired_observation_keys"] for row in contributions)


def test_seed_summary_uses_sample_standard_deviation():
    metrics = [
        {
            "model_configuration": "M1",
            "partition": "validation",
            "mae_soh_percentage_points": value,
            "rmse_soh_percentage_points": value * 2,
            "r2": 1.0 - value,
        }
        for value in (1.0, 3.0)
    ]

    summary = _seed_summary(metrics)
    row = next(
        item
        for item in summary
        if item["model_configuration"] == "M1"
        and item["partition"] == "validation"
    )

    assert row["successful_training_seeds"] == 2
    assert row["mean_mae_soh_percentage_points"] == pytest.approx(2.0)
    assert row["std_mae_soh_percentage_points"] == pytest.approx(np.sqrt(2.0))


def test_no_failures_are_reported_as_empty_success_status_not_as_a_failure():
    rows = _training_failure_rows([])

    assert len(rows) == 1
    assert rows[0]["error_message"] == "No failures"
    assert rows[0]["model_configuration"] == ""


def test_scaler_is_fit_on_supplied_training_rows_only():
    values = np.asarray([[1.0, 10.0], [3.0, 14.0]])

    mean, scale = _fit_scaler(values, "test")

    assert mean == pytest.approx([2.0, 12.0])
    assert scale == pytest.approx([1.0, 2.0])


def test_training_saves_validation_selected_checkpoint_and_train_only_scalers(tmp_path):
    records = []
    for index, (partition, target) in enumerate(
        (
            ("train", 80.0),
            ("train", 85.0),
            ("train", 90.0),
            ("train", 95.0),
            ("validation", 87.0),
            ("validation", 89.0),
            ("external_test", 88.0),
        )
    ):
        records.append(
            {
                "key": ("1", f"Cell{index}", str(index)),
                "part": "1",
                "physical_cell_id": f"Cell{index}",
                "rpt_index": index,
                "partition": partition,
                "soh_percent": target,
                "split_version": "test-split",
                "features": {
                    "discharge": np.arange(128, dtype=np.float32) * (index + 1),
                    "charge": np.arange(128, dtype=np.float32) * (index + 2),
                    "physics": np.asarray(
                        [index + 1, index + 2, index + 3, index + 4, index + 5],
                        dtype=np.float32,
                    ),
                },
            }
        )

    result = _train_one(
        records,
        "M7",
        ("physics",),
        211,
        tmp_path / "M7" / "seed_211",
        max_epochs=2,
        patience=2,
        batch_size=2,
        device=torch.device("cpu"),
    )
    checkpoint = torch.load(
        result["checkpoint_path"], map_location="cpu", weights_only=False
    )

    expected_mean = np.mean(
        [record["features"]["physics"] for record in records[:4]], axis=0
    )
    assert checkpoint["input_scalers"]["physics"]["mean"] == pytest.approx(
        expected_mean.tolist()
    )
    assert checkpoint["preprocessing_fit_scope"] == "training observations only"
    assert checkpoint["best_epoch"] in (1, 2)
    assert len(result["predictions"]) == len(records)
