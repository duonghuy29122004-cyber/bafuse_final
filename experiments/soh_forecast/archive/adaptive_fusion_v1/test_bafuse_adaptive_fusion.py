import csv

import numpy as np
import pytest
import torch

from experiments.soh_forecast import run_adaptive_fusion as adaptive


def test_preflight_matches_expected_modality_eligibility():
    result = adaptive.preflight()

    assert result["physical_cell_counts"] == {
        "train": 12,
        "validation": 4,
        "external_test": 12,
    }
    assert result["valid_label_counts"] == {
        "train": 113,
        "validation": 44,
        "external_test": 84,
    }
    assert result["eligibility"] == adaptive.EXPECTED_ELIGIBILITY
    assert result["g10_cell2_external_label_count"] == 10
    assert not result["recovered_g3_rpts_included"]
    assert not result["group5_group6_unlabeled_rows_included"]


def test_routing_is_deterministic_for_all_modality_patterns():
    expected = {
        frozenset({"D", "C", "P"}): ("M2", "M4"),
        frozenset({"D", "C"}): ("M2", "M2"),
        frozenset({"D", "P"}): ("M3", "M3"),
        frozenset({"C", "P"}): ("M5", "M5"),
        frozenset({"D"}): ("M1", "M1"),
        frozenset({"C"}): ("M6", "M6"),
        frozenset({"P"}): ("M7", "M7"),
        frozenset(): (None, None),
    }

    for bits, routes in expected.items():
        features = {
            name: np.asarray([1.0], dtype=np.float32)
            for name, code in adaptive.MODALITY_CODES.items()
            if code in bits
        }
        adaptive_route = adaptive._route_model(
            features, adaptive.ROUTING_POLICIES[0], "M2"
        )
        fixed_route = adaptive._route_model(
            features, adaptive.ROUTING_POLICIES[1], "M2"
        )
        assert (adaptive_route, fixed_route) == routes


@pytest.mark.parametrize(
    ("cell", "rpt"),
    (("G5-Cell1", "1"), ("G6-Cell1", "1"), ("G3-Cell14", "11"), ("G3-Cell14", "12")),
)
def test_validation_rejects_excluded_supervised_records(cell, rpt):
    record = {
        "key": ("1", cell, rpt),
        "physical_cell_id": cell,
        "partition": "train",
        "features": {"discharge": np.zeros(128, dtype=np.float32)},
    }

    with pytest.raises(ValueError):
        adaptive._validate_records([record], {cell: "train"}, {})


def test_csv_writer_handles_per_seed_and_seed_summary_schemas(tmp_path):
    path = tmp_path / "metrics.csv"
    adaptive._write_csv(
        path,
        [
            {"summary_statistic": "per_seed", "mae": 1.0},
            {
                "summary_statistic": "mean_and_sample_std_across_seeds",
                "mae": 1.5,
                "mae_std": 0.5,
                "successful_training_seeds": 3,
            },
        ],
    )

    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert rows[0]["mae_std"] == ""
    assert rows[0]["successful_training_seeds"] == ""
    assert rows[1]["mae_std"] == "0.5"


def test_training_scalers_use_training_observations_only(tmp_path, monkeypatch):
    monkeypatch.setattr(adaptive, "MAX_EPOCHS", 2)
    monkeypatch.setattr(adaptive, "PATIENCE", 2)

    records = []
    for index, (partition, cell, target, base) in enumerate(
        (
            ("train", "Train1", 80.0, 1.0),
            ("train", "Train2", 85.0, 3.0),
            ("train", "Train3", 90.0, 5.0),
            ("validation", "Validation1", 87.0, 1000.0),
            ("validation", "Validation2", 89.0, 2000.0),
            ("external_test", "External1", 88.0, -1000.0),
        )
    ):
        records.append(
            {
                "key": ("1", cell, str(index + 1)),
                "part": "1",
                "physical_cell_id": cell,
                "rpt_index": index + 1,
                "measurement_id": f"RPT{index + 1}",
                "partition": partition,
                "soh_percent": target,
                "split_version": "test-split",
                "features": {
                    "physics": np.asarray(
                        [base + offset for offset in range(5)],
                        dtype=np.float32,
                    )
                },
            }
        )

    result = adaptive._train_one(
        records,
        "M7",
        211,
        tmp_path / "M7" / "seed_211",
        torch.device("cpu"),
    )
    checkpoint = torch.load(
        result["checkpoint_path"],
        map_location="cpu",
        weights_only=False,
    )

    assert checkpoint["input_scalers"]["physics"]["mean"] == pytest.approx(
        [3.0, 4.0, 5.0, 6.0, 7.0]
    )
    assert checkpoint["preprocessing_fit_scope"] == "training observations only"
    assert checkpoint["training_observation_keys"] == [
        ("1", "Train1", "1"),
        ("1", "Train2", "2"),
        ("1", "Train3", "3"),
    ]
