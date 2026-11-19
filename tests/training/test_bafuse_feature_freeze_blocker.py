import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "experiments" / "soh_forecast" / "bafuse_architecture_benchmark_v1"
BLOCKER_DIR = BENCHMARK / "bafuse_feature_freeze_blocker_v1_20261114"
LABEL_DIR = BENCHMARK / "bafuse_soh_trajectory_audit_v1_20261114"
FEATURE_DIR = BENCHMARK / "raw_soh_reconstruction_v1_20261113"
PHYSICS_FEATURES = {
    "mean_discharge_temperature_native",
    "discharge_temperature_rise_native",
    "discharge_plateau_slope_v_per_progress",
    "discharge_slope_variability_v_per_progress",
    "mean_absolute_discharge_current_native",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["dataset_part"], row["physical_cell_id"], row["rpt_index"]


def test_authoritative_labels_and_feature_rows_remain_complete_and_cell_isolated():
    labels = _read_csv(LABEL_DIR / "bafuse_soh_labels_203.csv")
    features = _read_csv(FEATURE_DIR / "raw_curve_feature_matrix.csv")
    label_keys = {_key(row) for row in labels}
    feature_keys = {_key(row) for row in features}

    assert len(labels) == len(label_keys) == 203
    assert len(features) == len(feature_keys) == 203
    assert label_keys == feature_keys
    assert Counter(row["partition"] for row in labels) == {
        "train": 102,
        "validation": 37,
        "external_test": 64,
    }
    assert {row["final_status"] for row in labels} == {"ACCEPTED"}
    assert {
        row["target_definition"] for row in labels
    } == {
        "100 * Q_Step4_discharge(RPT) / Q_Step4_discharge(same-cell BoL)"
    }
    assert all(math.isfinite(float(row["soh_percent"])) for row in labels)

    cells_by_split = {}
    for row in labels:
        cells_by_split.setdefault(row["partition"], set()).add(
            row["physical_cell_id"]
        )
    assert not (cells_by_split["train"] & cells_by_split["validation"])
    assert not (cells_by_split["train"] & cells_by_split["external_test"])
    assert not (cells_by_split["validation"] & cells_by_split["external_test"])
    assert all(
        row["external_test_only"] == "True"
        for row in labels
        if row["dataset_part"] == "3"
    )


def test_candidate_modalities_are_numeric_but_temperature_blocker_prevents_freeze():
    features = _read_csv(FEATURE_DIR / "raw_curve_feature_matrix.csv")
    discharge_columns = [
        column for column in features[0] if column.startswith("discharge_v_")
    ]
    charge_columns = [
        column for column in features[0] if column.startswith("charge_v_")
    ]
    assert len(discharge_columns) == 128
    assert len(charge_columns) == 128
    assert {
        column for column in features[0] if column in PHYSICS_FEATURES
    } == PHYSICS_FEATURES
    for row in features:
        for column in discharge_columns + charge_columns + list(PHYSICS_FEATURES):
            assert math.isfinite(float(row[column]))
        assert "normalized_row_progress" in row["feature_provenance"]
        assert "capacity" in row["feature_target_leakage_fields_excluded"]
        assert "future_data" in row["feature_target_leakage_fields_excluded"]

    alignment = _read_csv(BLOCKER_DIR / "bafuse_alignment_audit.csv")
    assert len(alignment) == 203
    assert all(row["step4_block_hash_audit_match"] == "True" for row in alignment)
    assert all(row["step10_block_hash_audit_match"] == "True" for row in alignment)
    assert all(row["same_rpt_step_order_valid"] == "True" for row in alignment)
    assert all(row["discharge_curve_finite_points"] == "128" for row in alignment)
    assert all(row["charge_curve_finite_points"] == "128" for row in alignment)
    assert all(row["physics_finite_features"] == "5" for row in alignment)
    low_temperature = [
        row for row in alignment if row["part3_lowest_temperature_observation"] == "True"
    ]
    assert [
        (row["physical_cell_id"], row["rpt_index"])
        for row in low_temperature
    ] == [("G8-Cell18", "1")]

    config = json.loads(
        (BLOCKER_DIR / "bafuse_feature_audit_config.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["dataset_freeze_status"] == "BLOCKED_NO_FREEZE"
    assert config["dataset_freeze_authorized"] is True
    assert config["dataset_frozen"] is False
    assert config["preprocessing_fit"] == "none"
    assert config["training_authorized"] is False
    assert config["external_prediction_evaluation_authorized"] is False
    assert not list(BLOCKER_DIR.glob("*.npz"))


def test_duplicate_timestamp_sensitivity_is_recorded_without_rewriting_features():
    rows = _read_csv(BLOCKER_DIR / "bafuse_sampling_sensitivity.csv")
    g2 = next(
        row
        for row in rows
        if row["physical_cell_id"] == "G2-Cell3" and row["rpt_index"] == "2"
    )
    assert g2["raw_rows"] == "6791"
    assert g2["exact_adjacent_duplicate_rows_removed"] == "26"
    assert float(g2["feature_matrix_recomputation_max_abs_delta"]) == 0.0
    assert float(g2["duplicate_collapse_curve_max_abs_delta_v"]) > 0.0

    model_status = json.loads(
        (BLOCKER_DIR / "bafuse_model_forward_pass_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert model_status["synthetic_forward_passes_run"] is False
    assert model_status["training_started"] is False
    assert model_status["external_prediction_evaluation_started"] is False


def test_all_versioned_audit_outputs_match_their_sha256_manifest():
    manifest = json.loads(
        (BLOCKER_DIR / "bafuse_feature_audit_sha256.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["dataset_freeze_status"] == "BLOCKED_NO_FREEZE"
    assert manifest["dataset_identifier"] is None
    assert manifest["training_authorized"] is False
    assert manifest["external_prediction_evaluation_authorized"] is False
    for relative_path, expected in manifest["outputs_sha256"].items():
        assert _sha256(BLOCKER_DIR / relative_path) == expected

    source_hashes = _read_csv(BLOCKER_DIR / "bafuse_raw_source_hashes.csv")
    assert len(source_hashes) == 228
    assert Counter(row["status"] for row in source_hashes) == {
        "MATCH": 62,
        "CURRENT_SNAPSHOT_ONLY": 166,
    }
