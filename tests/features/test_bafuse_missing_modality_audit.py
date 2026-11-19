import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = (
    ROOT
    / "experiments"
    / "soh_forecast"
    / "bafuse_missing_modality_audit_v1"
)


def _read(name):
    with (OUTPUT / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def test_observation_level_coverage_reconciles():
    rows = _read("all_rpts_modality_availability.csv")
    assert len(rows) == 472
    assert sum(row["valid_soh_label"] == "TRUE" for row in rows) == 241
    assert sum(row["complete_DCP"] == "TRUE" for row in rows) == 220
    assert sum(
        row["valid_soh_label"] == "FALSE"
        and (row["D"], row["C"], row["P"]) == ("UNKNOWN",) * 3
        for row in rows
    ) == 231


def test_partition_and_modality_counts_reconcile():
    expected = {
        "train": (274, 113, 106, 161),
        "validation": (111, 44, 41, 67),
        "external_test": (87, 84, 73, 3),
    }
    rows = _read("modality_combination_summary.csv")
    for partition, (mapped, labeled, complete, unlabeled) in expected.items():
        selected = [row for row in rows if row["partition"] == partition]
        combinations = [
            row
            for row in selected
            if row["row_type"] == "valid_label_modality_combination"
        ]
        assert sum(int(row["observation_count"]) for row in combinations) == labeled
        assert next(
            int(row["observation_count"])
            for row in combinations
            if row["modality_combination"] == "D+C+P"
        ) == complete
        assert next(
            int(row["observation_count"])
            for row in selected
            if row["row_type"] == "missing_or_invalid_soh_label"
        ) == unlabeled
        assert next(
            int(row["observation_count"])
            for row in selected
            if row["row_type"] == "partition_total"
        ) == mapped


def test_missing_cells_are_labeled_as_unresolved_not_missing_modalities():
    rows = _read("missing_cells_summary.csv")
    assert {
        (row["partition"], row["physical_cell_id"])
        for row in rows
    } == {
        ("train", "G5-Cell16"),
        ("train", "G5-Cell6"),
        ("train", "G6-Cell1"),
        ("validation", "G5-Cell5"),
    }
    assert all(row["valid_soh_reference_rpts"] == "0" for row in rows)
    assert all(
        row["status_for_unlabeled_rpts"] == "not assessed; no validated SOH label"
        for row in rows
    )


def test_all_labeled_incomplete_rpts_have_explicit_reasons():
    rows = _read("missing_rpts_detailed.csv")
    assert len(rows) == 21
    assert all(row["exclusion_reason"] for row in rows)
    assert all(
        {row[name] for name in ("D", "C", "P")} <= {"TRUE", "FALSE"}
        for row in rows
    )
    assert sum(
        row["D"] == "TRUE"
        and row["C"] == "TRUE"
        and row["P"] == "FALSE"
        for row in rows
    ) == 4


def test_two_encoder_coverage_is_data_eligibility_only():
    rows = _read("potential_encoder_coverage.csv")
    expected_additions = {
        "M2": (0, 0, 4),
        "M3": (6, 2, 5),
        "M5": (0, 0, 0),
    }
    partitions = ("train", "validation", "external_test")
    for model, expected in expected_additions.items():
        selected = {
            row["partition"]: row
            for row in rows
            if row["model_configuration"] == model
        }
        assert tuple(
            int(selected[partition]["additional_observations_vs_M4"])
            for partition in partitions
        ) == expected
        assert all(
            "Data eligibility only" in row["interpretation"]
            for row in selected.values()
        )
