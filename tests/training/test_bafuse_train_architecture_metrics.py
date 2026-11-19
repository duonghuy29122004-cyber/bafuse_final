import pytest
import torch

from bafuse.hybrid.architecture_models import ARCHITECTURE_NAMES
from bafuse.hybrid.train_architectures import (
    _prediction_metrics_by_cell,
    _summarize_runs,
)


def test_per_cell_metrics_include_macro_averages_and_defined_r2_cells():
    rows = [
        {"partition": "validation", "physical_cell_id": "CellA"},
        {"partition": "validation", "physical_cell_id": "CellA"},
        {"partition": "validation", "physical_cell_id": "CellB"},
    ]

    per_cell, macro = _prediction_metrics_by_cell(
        rows,
        torch.tensor([1.0, 3.0, 5.0]),
        torch.tensor([2.0, 2.0, 4.0]),
        architecture="test",
        seed=17,
    )

    assert [row["N"] for row in per_cell] == [2, 1]
    assert [row["N_cells"] for row in per_cell] == [1, 1]
    assert macro["validation"]["cells"] == 2
    assert macro["validation"]["macro_MAE"] == pytest.approx(1.0)
    assert macro["validation"]["macro_RMSE"] == pytest.approx(1.0)
    assert macro["validation"]["macro_R2"] == pytest.approx(0.0)
    assert macro["validation"]["macro_R2_cells"] == 1


def test_seed_aggregation_uses_mean_and_sample_standard_deviation():
    rows = []
    for architecture in ARCHITECTURE_NAMES:
        for seed, mae, rmse, r2, macro_mae in (
            (17, 1.0, 2.0, 0.5, 2.0),
            (23, 3.0, 4.0, 0.7, 4.0),
        ):
            rows.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    "partition": "validation",
                    "N": 12,
                    "N_cells": 3,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2,
                    "macro_cell_MAE": macro_mae,
                    "macro_cell_RMSE": macro_mae + 1.0,
                    "macro_cell_R2": r2 - 0.1,
                }
            )
            rows.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    "partition": "train",
                    "N": 30,
                    "N_cells": 9,
                    "MAE": mae + 1.0,
                    "RMSE": rmse + 1.0,
                    "R2": r2 - 0.1,
                    "macro_cell_MAE": macro_mae + 1.0,
                    "macro_cell_RMSE": macro_mae + 2.0,
                    "macro_cell_R2": r2 - 0.2,
                }
            )

    summary = _summarize_runs(rows, [17, 23], 0.0)
    first = summary["architecture_summary"][0]

    assert first["N"] == 12
    assert first["N_cells"] == 3
    assert first["validation_MAE_mean"] == pytest.approx(2.0)
    assert first["validation_MAE_stdev"] == pytest.approx(2**0.5)
    assert first["validation_RMSE_mean"] == pytest.approx(3.0)
    assert first["validation_R2_mean"] == pytest.approx(0.6)
    assert first["validation_R2_stdev"] == pytest.approx(0.1 * 2**0.5)
    assert first["validation_R2_seed_count"] == 2
    assert first["validation_macro_cell_MAE_mean"] == pytest.approx(3.0)
    assert first["validation_macro_cell_MAE_stdev"] == pytest.approx(2**0.5)
    assert first["validation_macro_cell_R2_stdev"] == pytest.approx(0.1 * 2**0.5)
    assert first["train_N"] == 30
    assert first["train_N_cells"] == 9
    assert first["train_MAE_mean"] == pytest.approx(3.0)
    assert summary["selection_rule"] == (
        "lowest mean Validation overall MAE across fixed seeds; RMSE tie-break"
    )
