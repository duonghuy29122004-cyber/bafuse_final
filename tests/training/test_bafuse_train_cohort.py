from __future__ import annotations

import csv

from bafuse.hybrid.soh_fusion import PHYSICS_FEATURE_NAMES
from bafuse.hybrid.train_architectures import (
    _tensor_cohort,
    load_internal_cohort,
)


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _feature_row(part, cell, rpt, partition, physics_value):
    row = {
        "dataset_part": part,
        "physical_cell_id": cell,
        "rpt_index": rpt,
        "partition": partition,
    }
    for index in range(128):
        row[f"discharge_v_{index:03d}"] = "3.7"
        row[f"charge_v_{index:03d}"] = "3.9"
    row.update({name: str(physics_value) for name in PHYSICS_FEATURE_NAMES})
    return row


def test_internal_cohort_excludes_review_records_and_external_cells(tmp_path):
    split = tmp_path / "split.csv"
    labels = tmp_path / "labels.csv"
    features = tmp_path / "features.csv"
    split_rows = [
        {"physical_cell_id": "train-cell", "partition": "train"},
        {"physical_cell_id": "validation-cell", "partition": "validation"},
        {"physical_cell_id": "external-cell", "partition": "external_test"},
    ]
    split_rows.extend(
        {"physical_cell_id": f"other-train-{index}", "partition": "train"}
        for index in range(8)
    )
    split_rows.extend(
        {"physical_cell_id": f"other-validation-{index}", "partition": "validation"}
        for index in range(2)
    )
    split_rows.extend(
        {"physical_cell_id": f"other-external-{index}", "partition": "external_test"}
        for index in range(11)
    )
    _write_csv(split, split_rows)
    label_rows = []
    for part, cell, partition, rpt, status, target in (
        ("1", "train-cell", "train", "1", "ACCEPTED", "95"),
        ("1", "validation-cell", "validation", "1", "ACCEPTED", "90"),
        ("3", "external-cell", "external_test", "1", "ACCEPTED", "85"),
        ("1", "validation-cell", "validation", "2", "REVIEW", ""),
    ):
        label_rows.append(
            {
                "dataset_part": part,
                "physical_cell_id": cell,
                "rpt_index": rpt,
                "partition": partition,
                "label_disposition": status,
                "final_eligibility_status": "ELIGIBLE_COMPLETE_THREE_MODALITY_RECORD",
                "reconstructed_soh_percent": target,
            }
        )
    _write_csv(labels, label_rows)
    _write_csv(
        features,
        [
            _feature_row("1", "train-cell", "1", "train", 1),
            _feature_row("1", "validation-cell", "1", "validation", 10),
            _feature_row("3", "external-cell", "1", "external_test", 100),
        ],
    )

    cohort, feature_map = load_internal_cohort(labels, features, split)

    assert len(cohort) == 2
    assert {row["partition"] for row in cohort} == {"train", "validation"}
    assert all("external-cell" not in key for key in feature_map)
    inputs, targets, _, preprocessing = _tensor_cohort(cohort, feature_map)
    assert targets.tolist() == [95.0, 90.0]
    assert inputs["physics"][0].tolist() == [0.0] * len(PHYSICS_FEATURE_NAMES)
    assert inputs["physics"][1].tolist() == [9.0] * len(PHYSICS_FEATURE_NAMES)
    assert preprocessing["fit_scope"] == "Train records only"
