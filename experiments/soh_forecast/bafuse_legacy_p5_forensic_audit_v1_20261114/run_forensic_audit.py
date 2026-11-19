from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
LEGACY = ROOT / "experiments" / "soh_forecast" / "bafuse_three_encoder_v1"
LEGACY_AUDIT = LEGACY / "evaluation_audit_v1"
ABLATION = ROOT / "experiments" / "soh_forecast" / "bafuse_three_encoder_ablation_v1"
BENCHMARK = ROOT / "experiments" / "soh_forecast" / "bafuse_architecture_benchmark_v1"
P5 = BENCHMARK / "bafuse_p5_electrical_v1_20261114"
P5_TRAIN = BENCHMARK / "bafuse_p5_trainval_benchmark_v2_20261114"
P5_EXTERNAL = BENCHMARK / "bafuse_p5_external_evaluation_v1_20261114"
EXPECTED_OUTPUTS = (
    "legacy_experiment_discovery.md",
    "legacy_vs_p5_architecture_comparison.md",
    "legacy_vs_p5_dataset_comparison.md",
    "legacy_vs_p5_observation_alignment.csv",
    "legacy_vs_p5_label_comparison.csv",
    "legacy_vs_p5_feature_comparison.md",
    "legacy_vs_p5_preprocessing_comparison.md",
    "legacy_vs_p5_training_comparison.md",
    "legacy_leakage_audit.md",
    "legacy_vs_p5_common_cohort_metrics.csv",
    "legacy_checkpoint_reproducibility.md",
    "performance_degradation_hypotheses.md",
    "forensic_audit_summary.md",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _manifest_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _manifest_entries(
    manifest_path: Path,
) -> tuple[dict[Path, set[str]], dict[Path, str]]:
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    roles: dict[Path, set[str]] = defaultdict(set)
    expected: dict[Path, str] = {}
    for section in ("inputs", "code", "outputs"):
        entries = document.get(section, {})
        if not isinstance(entries, dict):
            continue
        for name, entry in entries.items():
            if not isinstance(entry, dict) or not entry.get("path"):
                continue
            path = _manifest_path(str(entry["path"])).resolve()
            roles[path].add(f"{manifest_path.name}:{section}:{name}")
            declared = entry.get("sha256")
            if declared:
                previous = expected.get(path)
                if previous is not None and previous != declared:
                    raise ValueError(f"Conflicting manifest hashes for {path}")
                expected[path] = str(declared)
    return roles, expected


def _key(part: Any, cell: Any, rpt: Any) -> tuple[str, str, str]:
    part_text = str(part).strip()
    if part_text.startswith("Part "):
        part_text = part_text[5:]
    part_text = str(int(float(part_text)))
    cell_text = str(cell).strip()
    rpt_text = str(rpt).strip()
    if rpt_text.upper().startswith("RPT"):
        rpt_text = rpt_text[3:]
    return part_text, cell_text, str(int(float(rpt_text)))


def _row_key(row: dict[str, str]) -> tuple[str, str, str]:
    part = row.get("dataset_part", row.get("part", ""))
    rpt = row.get("rpt_index", row.get("measurement_id", ""))
    return _key(part, row["physical_cell_id"], rpt)


def _f(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Expected finite number, got {value!r}")
    return number


def _norm_path(value: str) -> str:
    return value.replace("\\", "/").strip().casefold().lstrip("./")


def _paths_match(left: str, right: str) -> bool:
    a = _norm_path(left)
    b = _norm_path(right)
    return a == b or a.endswith("/" + b) or b.endswith("/" + a)


def _metrics(actual: list[float], predicted: list[float]) -> dict[str, float | int]:
    y = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1 or y.size == 0:
        raise ValueError("Metrics require non-empty, aligned one-dimensional arrays.")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Metrics received non-finite values.")
    residual = p - y
    denominator = float(np.square(y - y.mean()).sum())
    r2 = 1.0 - float(np.square(residual).sum()) / denominator if denominator else float("nan")
    return {
        "n": int(y.size),
        "mae_soh_percentage_points": float(np.abs(residual).mean()),
        "rmse_soh_percentage_points": float(np.sqrt(np.square(residual).mean())),
        "r2": r2,
    }


def _number(value: float | int | None, digits: int = 8) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}g}"


def _tomatrix(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _assert_unique(rows: list[dict[str, str]], name: str) -> dict[tuple[str, str, str], dict[str, str]]:
    result: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = _row_key(row)
        if key in result:
            raise ValueError(f"Duplicate {name} observation key: {key}")
        result[key] = row
    return result


def _partition_cells(
    rows_by_key: dict[tuple[str, str, str], dict[str, str]],
    partition_by_key: dict[tuple[str, str, str], str],
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for key in rows_by_key:
        result[partition_by_key[key]].add(key[1])
    return result


def _overlap_counts(cells: dict[str, set[str]]) -> dict[str, int]:
    names = ("train", "validation", "external_test")
    return {
        f"{left}_vs_{right}": len(cells.get(left, set()) & cells.get(right, set()))
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    }


def _collect_sources() -> tuple[dict[Path, set[str]], dict[Path, str], list[Path]]:
    manifests = [
        P5 / "bafuse_p5_sha256_manifest.json",
        P5_TRAIN / "benchmark_sha256_manifest.json",
        P5_EXTERNAL / "external_sha256_manifest.json",
    ]
    roles: dict[Path, set[str]] = defaultdict(set)
    expected: dict[Path, str] = {}
    for manifest_path in manifests:
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        roles[manifest_path.resolve()].add("verified artifact manifest")
        manifest_roles, manifest_expected = _manifest_entries(manifest_path)
        for path, path_roles in manifest_roles.items():
            roles[path].update(path_roles)
        for path, digest in manifest_expected.items():
            prior = expected.get(path)
            if prior is not None and prior != digest:
                raise ValueError(f"Conflicting source manifests for {path}")
            expected[path] = digest

    explicit = {
        LEGACY / "run_phase2.py": "legacy training and inference source",
        LEGACY / "fusion_model_config.json": "legacy model configuration",
        LEGACY / "fusion_checkpoint.pt": "legacy saved fusion checkpoint",
        LEGACY / "fusion_predictions.csv": "legacy saved predictions",
        LEGACY / "fusion_soh_evaluation.csv": "legacy evaluation",
        LEGACY / "fusion_training_history.csv": "legacy training history",
        LEGACY / "multimodal_feature_manifest.csv": "legacy observation manifest",
        LEGACY / "discharge_features.csv": "legacy discharge features",
        LEGACY / "charge_features.csv": "legacy charge features",
        LEGACY / "physics_features.csv": "legacy physics features",
        LEGACY_AUDIT / "audit_soh_evaluation.py": "supplemental inference source",
        LEGACY_AUDIT / "supplemental_g10_cell2_labels.csv": "supplemental label provenance",
        LEGACY_AUDIT / "supplemental_g10_cell2_predictions.csv": "supplemental saved-checkpoint predictions",
        LEGACY_AUDIT / "soh_overall_metrics.csv": "corrected 73-observation metrics",
        LEGACY_AUDIT / "corrected_soh_evaluation_report.md": "corrected cohort report",
        LEGACY_AUDIT / "shared_subset_model_comparison.csv": "shared-key audit",
        LEGACY_AUDIT / "generalization_feature_shift.csv": "legacy feature-shift audit",
        ABLATION / "ablation_input_manifest.csv": "legacy 220-observation cohort",
        ABLATION / "historical_reference_metrics.csv": "legacy historical metric record",
        ABLATION / "run_metadata.json": "legacy ablation run metadata",
        ABLATION / "input_checksums.csv": "legacy run input checksums",
        ABLATION / "protected_artifact_integrity.csv": "legacy protected-artifact verification",
        ROOT / "src" / "bafuse" / "hybrid" / "soh_fusion.py": "legacy model and feature code",
        ROOT / "experiments" / "soh_forecast" / "run_ablation_phase3.py": "legacy cohort assembly and run source",
        ROOT / "src" / "bafuse" / "hybrid" / "p5_benchmark.py": "P5 training configuration and runner",
        ROOT / "src" / "bafuse" / "hybrid" / "p5_benchmark_models.py": "P5 model definitions",
        ROOT / "src" / "bafuse" / "hybrid" / "p5_electrical_features.py": "P5 feature definitions",
        ROOT / "src" / "bafuse" / "hybrid" / "p5_external_evaluation.py": "P5 saved-checkpoint evaluation source",
        P5 / "bafuse_p5_feature_schema.json": "P5 feature schema",
        P5 / "bafuse_p5_preprocessing_config.json": "P5 preprocessing parameters",
        P5 / "bafuse_p5_split_manifest.csv": "P5 locked split",
        P5 / "bafuse_soh_labels_frozen.csv": "P5 frozen target table",
        P5 / "bafuse_p5_feature_audit.csv": "P5 row-level source and feature audit",
        P5 / "bafuse_train_p5.npz": "P5 train tensors",
        P5 / "bafuse_validation_p5.npz": "P5 validation tensors",
        P5 / "bafuse_external_p5.npz": "P5 external tensors",
        P5_TRAIN / "benchmark_config.json": "P5 training configuration",
        P5_TRAIN / "benchmark_run_metrics.csv": "P5 saved training metrics",
        P5_TRAIN / "train_validation_predictions.csv": "P5 saved train/validation predictions",
        P5_TRAIN / "benchmark_training_report.md": "P5 training report",
        P5_EXTERNAL / "external_predictions.csv": "P5 saved external predictions",
        P5_EXTERNAL / "external_evaluation_report.md": "P5 external evaluation report",
        ROOT / "data" / "processed" / "label_soh_part1_2" / "run_007_part1_capacity_qc" / "labels_soh.csv": "historical Part 1 label table",
        ROOT / "data" / "processed" / "label_soh_part1_2" / "run_010_part2_soh_candidates" / "labels_soh_part2.csv": "historical Part 2 label table",
        ROOT / "experiments" / "results" / "data_audits" / "part3_soh_candidates" / "labels_soh_candidates.csv": "historical Part 3 label table",
    }
    for path, role in explicit.items():
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Required audit input is missing: {resolved}")
        roles[resolved].add(role)
    roles[Path(__file__).resolve()].add("read-only forensic audit code")
    paths = sorted(roles, key=lambda value: _relative(value))
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Manifest-referenced source is missing: {path}")
    return roles, expected, manifests


def _verify_sources(
    roles: dict[Path, set[str]], expected: dict[Path, str]
) -> dict[Path, str]:
    hashes = {path: _sha256(path) for path in roles}
    for path, digest in expected.items():
        if path not in hashes:
            hashes[path] = _sha256(path)
        if hashes[path] != digest:
            raise ValueError(
                f"Source hash mismatch for {_relative(path)}: "
                f"expected {digest}, observed {hashes[path]}"
            )
    return hashes


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    result = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    result.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(result)


def main() -> None:
    existing = [name for name in EXPECTED_OUTPUTS + ("forensic_audit_sha256_manifest.json",) if (OUT / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing audit outputs: {existing}")

    roles, declared_hashes, verified_manifests = _collect_sources()
    source_hashes_before = _verify_sources(roles, declared_hashes)
    p5_manifest = json.loads((P5 / "bafuse_p5_sha256_manifest.json").read_text(encoding="utf-8"))
    p5_benchmark_manifest = json.loads((P5_TRAIN / "benchmark_sha256_manifest.json").read_text(encoding="utf-8"))
    p5_external_manifest = json.loads((P5_EXTERNAL / "external_sha256_manifest.json").read_text(encoding="utf-8"))
    if p5_manifest.get("dataset_identifier") != "bafuse-p5-electrical-v1-20261114":
        raise ValueError("Unexpected frozen P5 dataset identifier.")

    legacy_rows = _read_csv(ABLATION / "ablation_input_manifest.csv")
    p5_split_rows = _read_csv(P5 / "bafuse_p5_split_manifest.csv")
    p5_label_rows = _read_csv(P5 / "bafuse_soh_labels_frozen.csv")
    p5_feature_rows = _read_csv(P5 / "bafuse_p5_feature_audit.csv")
    legacy_by_key = _assert_unique(legacy_rows, "legacy manifest")
    p5_split_by_key = _assert_unique(p5_split_rows, "P5 split")
    p5_label_by_key = _assert_unique(p5_label_rows, "P5 label")
    p5_feature_by_key = _assert_unique(p5_feature_rows, "P5 feature audit")
    if len(legacy_by_key) != 220 or len(p5_split_by_key) != 203 or len(p5_label_by_key) != 203:
        raise ValueError(
            "Cohort size changed: expected legacy 220 and P5 203, got "
            f"{len(legacy_by_key)}, {len(p5_split_by_key)}, {len(p5_label_by_key)}"
        )
    if set(p5_split_by_key) != set(p5_label_by_key) or set(p5_split_by_key) != set(p5_feature_by_key):
        raise ValueError("P5 labels, split, and feature audit do not have identical keys.")
    if not set(p5_split_by_key).issubset(legacy_by_key):
        raise ValueError("P5 observation keys are not a subset of the historical cohort.")

    old_counts = Counter(row["partition"] for row in legacy_rows)
    p5_counts = Counter(row["partition"] for row in p5_split_rows)
    expected_old_counts = {"train": 106, "validation": 41, "external_test": 73}
    expected_p5_counts = {"train": 102, "validation": 37, "external_test": 64}
    if dict(old_counts) != expected_old_counts or dict(p5_counts) != expected_p5_counts:
        raise ValueError(f"Unexpected split counts: legacy={dict(old_counts)}, P5={dict(p5_counts)}")

    old_partition = {key: row["partition"] for key, row in legacy_by_key.items()}
    p5_partition = {key: row["partition"] for key, row in p5_split_by_key.items()}
    old_cells = _partition_cells(legacy_by_key, old_partition)
    p5_cells = _partition_cells(p5_split_by_key, p5_partition)
    if any(_overlap_counts(old_cells).values()) or any(_overlap_counts(p5_cells).values()):
        raise ValueError("A physical cell overlaps partitions in one of the cohorts.")

    old_keys, p5_keys = set(legacy_by_key), set(p5_split_by_key)
    common_keys = old_keys & p5_keys
    legacy_only = old_keys - p5_keys
    p5_only = p5_keys - old_keys
    transition_counts = Counter((old_partition[key], p5_partition[key]) for key in common_keys)
    changed_split_count = sum(old_partition[key] != p5_partition[key] for key in common_keys)

    label_sources = (
        ROOT / "data" / "processed" / "label_soh_part1_2" / "run_007_part1_capacity_qc" / "labels_soh.csv",
        ROOT / "data" / "processed" / "label_soh_part1_2" / "run_010_part2_soh_candidates" / "labels_soh_part2.csv",
        ROOT / "experiments" / "results" / "data_audits" / "part3_soh_candidates" / "labels_soh_candidates.csv",
    )
    raw_label_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for path in label_sources:
        for row in _read_csv(path):
            if row.get("condition") != "RPT" or not row.get("measurement_id", "").upper().startswith("RPT"):
                continue
            key = _row_key(row)
            if key in raw_label_by_key:
                raise ValueError(f"Duplicate original raw-label key {key}")
            raw_label_by_key[key] = row
    if not p5_keys.issubset(raw_label_by_key):
        raise ValueError(f"Original raw-label sources are missing {len(p5_keys - set(raw_label_by_key))} P5 keys.")

    label_comparison: list[dict[str, Any]] = []
    baseline_matches = 0
    native_qc_exact_matches = 0
    label_deltas: list[float] = []
    abs_label_deltas: list[float] = []
    for key in sorted(p5_keys, key=lambda item: (int(item[0]), item[1], int(item[2]))):
        old_row = legacy_by_key[key]
        p5_row = p5_label_by_key[key]
        raw_row = raw_label_by_key[key]
        old_target = _f(old_row["soh_percent"])
        p5_target = _f(p5_row["soh_percent"])
        raw_target = _f(raw_row["soh_percent"])
        native_qc = _f(p5_row["soh_native_amphr_qc_percent"])
        delta = p5_target - old_target
        label_deltas.append(delta)
        abs_label_deltas.append(abs(delta))
        baseline = _paths_match(
            raw_row["baseline_source_file"], p5_row["bol_step4_source_file"]
        )
        baseline_matches += int(baseline)
        native_exact = native_qc == old_target
        native_qc_exact_matches += int(native_exact)
        label_comparison.append(
            {
                "dataset_part": key[0],
                "physical_cell_id": key[1],
                "rpt_id": f"RPT{key[2]}",
                "legacy_soh_percent": old_target,
                "p5_soh_percent": p5_target,
                "p5_minus_legacy_soh_pp": delta,
                "absolute_delta_pp": abs(delta),
                "historical_raw_label_percent": raw_target,
                "p5_native_amphr_qc_percent": native_qc,
                "native_qc_equals_legacy_label": native_exact,
                "historical_bol_source": raw_row["baseline_source_file"],
                "p5_bol_step4_source": p5_row["bol_step4_source_file"],
                "bol_source_path_match": baseline,
                "p5_step4_source": p5_row["rpt_step4_source_file"],
                "p5_step4_segment_ref": p5_row["rpt_step4_segment_ref"],
                "p5_step4_block_sha256": p5_row["rpt_step4_block_sha256"],
                "p5_bol_step4_block_sha256": p5_row["bol_step4_block_sha256"],
            }
        )

    if baseline_matches != 203 or native_qc_exact_matches != 203:
        raise ValueError(
            "Historical BoL mapping or native-Amphr label cross-check failed: "
            f"baseline matches={baseline_matches}, exact native-QC matches={native_qc_exact_matches}"
        )

    p5_predictions: dict[tuple[str, str, str], dict[str, str]] = {}
    p5_train_rows = _read_csv(P5_TRAIN / "train_validation_predictions.csv")
    p5_external_rows = _read_csv(P5_EXTERNAL / "external_predictions.csv")
    for row in p5_train_rows + p5_external_rows:
        if row.get("architecture") != "BaFuse_CNN1D_P5" or row.get("seed") != "123":
            continue
        key = _row_key(row)
        if key in p5_predictions:
            raise ValueError(f"Duplicate saved P5 seed-123 prediction for {key}")
        p5_predictions[key] = row
    if set(p5_predictions) != p5_keys:
        raise ValueError(
            f"P5 seed-123 saved predictions do not match frozen labels: "
            f"missing={len(p5_keys - set(p5_predictions))}, extra={len(set(p5_predictions) - p5_keys)}"
        )

    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "experiments" / "soh_forecast"))
    import run_ablation_phase3 as legacy_runner
    from bafuse.hybrid.p5_benchmark_models import build_model, trainable_parameter_count
    from bafuse.hybrid.soh_fusion import ThreeEncoderSoH

    records, split_version, _ = legacy_runner._load_common_cohort()
    if len(records) != 220:
        raise ValueError(f"Read-only legacy cohort loader returned {len(records)} rows, expected 220.")
    if any(record["part"] == "3" and record["partition"] != "external_test" for record in records):
        raise ValueError("Part 3 records entered a non-external legacy partition.")

    legacy_checkpoint = torch.load(
        LEGACY / "fusion_checkpoint.pt", map_location="cpu", weights_only=False
    )
    if legacy_checkpoint.get("split_version") != split_version:
        raise ValueError("Legacy checkpoint split does not match the assembled cohort.")
    fusion = ThreeEncoderSoH()
    fusion.load_state_dict(legacy_checkpoint["model_state_dict"], strict=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    fusion.to(device).eval()
    discharge_mean = _tomatrix(legacy_checkpoint["discharge_mean"])
    discharge_scale = _tomatrix(legacy_checkpoint["discharge_scale"])
    charge_mean = _tomatrix(legacy_checkpoint["charge_mean"])
    charge_scale = _tomatrix(legacy_checkpoint["charge_scale"])
    physics_mean = _tomatrix(legacy_checkpoint["physics_mean"])
    physics_scale = _tomatrix(legacy_checkpoint["physics_scale"])
    target_mean = float(legacy_checkpoint["target_mean"])
    target_scale = float(legacy_checkpoint["target_scale"])
    if target_scale <= 0 or not math.isfinite(target_scale):
        raise ValueError("Legacy checkpoint target scale is invalid.")

    reproduced: dict[tuple[str, str, str], float] = {}
    with torch.inference_mode():
        for record in records:
            key = record["key"]
            values = record["features"]
            normalized = (
                (np.asarray(values["discharge"], dtype=np.float32) - discharge_mean)
                / discharge_scale,
                (np.asarray(values["charge"], dtype=np.float32) - charge_mean)
                / charge_scale,
                (np.asarray(values["physics"], dtype=np.float32) - physics_mean)
                / physics_scale,
            )
            inputs = tuple(
                torch.from_numpy(value.astype(np.float32, copy=False)[None, :]).to(device)
                for value in normalized
            )
            output = fusion(*inputs)
            prediction = float(output.reshape(-1)[0].item() * target_scale + target_mean)
            if not math.isfinite(prediction):
                raise ValueError(f"Non-finite reproduced legacy prediction for {key}")
            reproduced[key] = prediction
    if set(reproduced) != old_keys:
        raise ValueError("Reproduced legacy predictions do not match the 220-row historical manifest.")

    saved_legacy_predictions: dict[tuple[str, str, str], float] = {}
    for row in _read_csv(LEGACY / "fusion_predictions.csv"):
        key = _row_key(row)
        saved_legacy_predictions[key] = _f(row["fusion_v2_predicted_soh_percent"])
    for row in _read_csv(LEGACY_AUDIT / "supplemental_g10_cell2_predictions.csv"):
        if row.get("fusion_eligible", "").casefold() != "true":
            continue
        key = _row_key(row)
        saved_legacy_predictions[key] = _f(row["fusion_v2_predicted_soh_percent"])
    if set(saved_legacy_predictions) != old_keys:
        raise ValueError(
            f"Saved legacy predictions do not match the 220-row manifest: "
            f"saved={len(saved_legacy_predictions)}, expected={len(old_keys)}"
        )
    reproduction_deltas = {
        key: reproduced[key] - saved_legacy_predictions[key] for key in old_keys
    }
    maximum_reproduction_delta = max(abs(value) for value in reproduction_deltas.values())
    exact_reproduction_count = sum(value == 0.0 for value in reproduction_deltas.values())

    alignment_rows: list[dict[str, Any]] = []
    for key in sorted(old_keys | p5_keys, key=lambda item: (int(item[0]), item[1], int(item[2]))):
        old = legacy_by_key.get(key)
        new = p5_split_by_key.get(key)
        label = p5_label_by_key.get(key)
        feature = p5_feature_by_key.get(key)
        alignment_rows.append(
            {
                "dataset_part": key[0],
                "physical_cell_id": key[1],
                "rpt_id": f"RPT{key[2]}",
                "legacy_partition": old["partition"] if old else "",
                "p5_partition": new["partition"] if new else "",
                "present_in_legacy": old is not None,
                "present_in_p5": new is not None,
                "split_changed_for_common_key": bool(old and new and old["partition"] != new["partition"]),
                "legacy_soh_percent": _f(old["soh_percent"]) if old else "",
                "p5_soh_percent": _f(label["soh_percent"]) if label else "",
                "legacy_source_file": old["source_file"] if old else "",
                "p5_step4_source_file": label["rpt_step4_source_file"] if label else "",
                "p5_step4_segment_ref": label["rpt_step4_segment_ref"] if label else "",
                "p5_step10_source_file": feature["step10_source_file"] if feature else "",
                "p5_step10_segment_ref": feature["step10_segment_ref"] if feature else "",
            }
        )

    metrics_rows: list[dict[str, Any]] = []

    def add_metric(
        population: str,
        model_name: str,
        keys: list[tuple[str, str, str]],
        reference_name: str,
        actual_map: dict[tuple[str, str, str], float],
        prediction_map: dict[tuple[str, str, str], float],
        note: str,
    ) -> None:
        if not keys:
            raise ValueError(f"Empty metric population for {population}/{model_name}")
        metrics = _metrics(
            [actual_map[key] for key in keys],
            [prediction_map[key] for key in keys],
        )
        metrics_rows.append(
            {
                "population": population,
                "model": model_name,
                "reference": reference_name,
                "observation_count": metrics["n"],
                "physical_cell_count": len({key[1] for key in keys}),
                "mae_soh_percentage_points": metrics["mae_soh_percentage_points"],
                "rmse_soh_percentage_points": metrics["rmse_soh_percentage_points"],
                "r2": metrics["r2"],
                "note": note,
            }
        )

    legacy_targets = {key: _f(legacy_by_key[key]["soh_percent"]) for key in old_keys}
    p5_targets = {key: _f(p5_label_by_key[key]["soh_percent"]) for key in p5_keys}
    p5_prediction_values = {
        key: _f(row["predicted_soh_percent"]) for key, row in p5_predictions.items()
    }
    external_common = sorted(
        [key for key in common_keys if p5_partition[key] == "external_test"],
        key=lambda item: (int(item[0]), item[1], int(item[2])),
    )
    legacy_external_73 = sorted(
        [key for key in old_keys if old_partition[key] == "external_test"],
        key=lambda item: (int(item[0]), item[1], int(item[2])),
    )
    add_metric(
        "common_external_64",
        "legacy_three_encoder_seed_211_reproduced",
        external_common,
        "P5 frozen discharge-retention labels",
        p5_targets,
        reproduced,
        "Exact same 64 Part 3 keys and P5 target values; inference only.",
    )
    add_metric(
        "common_external_64",
        "P5_CNN1D_seed_123_saved",
        external_common,
        "P5 frozen discharge-retention labels",
        p5_targets,
        p5_prediction_values,
        "Existing saved P5 prediction output; no new P5 inference or selection.",
    )
    add_metric(
        "common_external_64",
        "legacy_three_encoder_seed_211_reproduced",
        external_common,
        "Historical legacy manifest labels",
        legacy_targets,
        reproduced,
        "Same 64 keys as above; isolates the small label-construction difference.",
    )
    add_metric(
        "common_external_64",
        "P5_CNN1D_seed_123_saved",
        external_common,
        "Historical legacy manifest labels",
        legacy_targets,
        p5_prediction_values,
        "Same 64 keys as above; existing P5 predictions, alternate reference only.",
    )
    add_metric(
        "legacy_external_73_historical",
        "legacy_three_encoder_seed_211_reproduced",
        legacy_external_73,
        "Historical legacy manifest labels",
        legacy_targets,
        reproduced,
        "Reconstructs the saved 73-row report population, including nine supplemental G10-Cell2 observations.",
    )

    reference_metrics = _read_csv(ABLATION / "historical_reference_metrics.csv")
    recorded_external = next(
        row
        for row in reference_metrics
        if row["model"] == "three_encoder_fusion"
        and row["partition"] == "external_test"
    )
    reproduced_legacy_external = _metrics(
        [legacy_targets[key] for key in legacy_external_73],
        [reproduced[key] for key in legacy_external_73],
    )
    p5_saved_external_metrics = _metrics(
        [p5_targets[key] for key in external_common],
        [p5_prediction_values[key] for key in external_common],
    )
    legacy_common_p5_label_metrics = _metrics(
        [p5_targets[key] for key in external_common],
        [reproduced[key] for key in external_common],
    )

    feature_schema = json.loads((P5 / "bafuse_p5_feature_schema.json").read_text(encoding="utf-8"))
    preprocessing = json.loads((P5 / "bafuse_p5_preprocessing_config.json").read_text(encoding="utf-8"))
    old_model_config = json.loads((LEGACY / "fusion_model_config.json").read_text(encoding="utf-8"))
    p5_benchmark_config = json.loads((P5_TRAIN / "benchmark_config.json").read_text(encoding="utf-8"))
    p5_architecture = build_model("BaFuse_CNN1D_P5")
    legacy_architecture = ThreeEncoderSoH()
    legacy_parameter_count = sum(parameter.numel() for parameter in legacy_architecture.parameters() if parameter.requires_grad)
    p5_parameter_count = trainable_parameter_count(p5_architecture)

    p5_audit = p5_feature_by_key
    if any(row.get("predictive_label_or_bol_feature", "").casefold() != "false" for row in p5_feature_rows):
        raise ValueError("P5 feature audit marks a label/BoL-derived predictor.")
    if any(row.get("future_rpt_feature", "").casefold() != "false" for row in p5_feature_rows):
        raise ValueError("P5 feature audit marks a future-RPT feature.")
    if any(row.get("curve_progress_basis") != "normalized source-row index" for row in p5_feature_rows):
        raise ValueError("P5 curve-progress basis differs from the audited source-row index.")

    max_step4_curve_delta = max(_f(row["old_step4_curve_max_abs_delta_v"]) for row in p5_feature_rows)
    max_step10_curve_delta = max(_f(row["old_step10_curve_max_abs_delta_v"]) for row in p5_feature_rows)
    raw_current_train = [
        _f(row["p5_raw_mean_abs_discharge_current"])
        for key, row in p5_audit.items()
        if p5_partition[key] == "train"
    ]
    raw_current_external = [
        _f(row["p5_raw_mean_abs_discharge_current"])
        for key, row in p5_audit.items()
        if p5_partition[key] == "external_test"
    ]
    p5_z_external = [
        _f(row["p5_scaled_mean_abs_discharge_current"])
        for key, row in p5_audit.items()
        if p5_partition[key] == "external_test"
    ]
    old_current_mean = float(physics_mean.reshape(-1)[4])
    old_current_scale = float(physics_scale.reshape(-1)[4])
    if old_current_scale <= 0:
        raise ValueError("Legacy mean-current scaler has non-positive scale.")
    old_z_external = [(value - old_current_mean) / old_current_scale for value in raw_current_external]
    current_external_over_three = sum(abs(value) > 3.0 for value in p5_z_external)
    old_current_external_over_three = sum(abs(value) > 3.0 for value in old_z_external)

    p5_feature_names = list(
        feature_schema.get("modalities", {})
        .get("X_physics", {})
        .get("feature_names_in_order", [])
    )
    if not p5_feature_names:
        p5_feature_names = list(
            feature_schema.get("modalities", {})
            .get("X_physics", {})
            .get("feature_names", [])
        )

    old_label_deltas = np.asarray(abs_label_deltas, dtype=np.float64)
    label_delta_mean = float(old_label_deltas.mean())
    label_delta_median = float(np.median(old_label_deltas))
    label_delta_max = float(old_label_deltas.max())
    signed_label_delta_median = float(np.median(np.asarray(label_deltas, dtype=np.float64)))
    label_delta_exact = sum(value == 0.0 for value in abs_label_deltas)

    legacy_current = {
        "legacy": {
            "mean": old_current_mean,
            "scale": old_current_scale,
            "external_z_mean": float(np.mean(old_z_external)),
            "external_z_min": float(np.min(old_z_external)),
            "external_z_max": float(np.max(old_z_external)),
            "external_abs_z_gt_3": old_current_external_over_three,
        },
        "p5": {
            "external_z_mean": float(np.mean(p5_z_external)),
            "external_z_min": float(np.min(p5_z_external)),
            "external_z_max": float(np.max(p5_z_external)),
            "external_abs_z_gt_3": current_external_over_three,
        },
    }

    # Read the P5 benchmark parameters as recorded; do not fit or update any transform.
    p5_training = p5_benchmark_config.get("training", {})
    p5_scaler_fit_partition = preprocessing.get("fit_partition")
    p5_scaler_fit_rows = preprocessing.get("fit_row_count")
    if p5_scaler_fit_partition != "train" or int(p5_scaler_fit_rows) != 102:
        raise ValueError("P5 frozen preprocessing is not recorded as Train-only on 102 observations.")

    original_metrics_delta = {
        "mae_abs_difference": abs(
            reproduced_legacy_external["mae_soh_percentage_points"]
            - _f(recorded_external["mae_soh_percentage_points"])
        ),
        "rmse_abs_difference": abs(
            reproduced_legacy_external["rmse_soh_percentage_points"]
            - _f(recorded_external["rmse_soh_percentage_points"])
        ),
        "r2_abs_difference": abs(reproduced_legacy_external["r2"] - _f(recorded_external["r2"])),
    }

    runtime = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "not available",
        "utc_created_at": datetime.now(timezone.utc).isoformat(),
    }

    alignment_fields = [
        "dataset_part", "physical_cell_id", "rpt_id", "legacy_partition", "p5_partition",
        "present_in_legacy", "present_in_p5", "split_changed_for_common_key",
        "legacy_soh_percent", "p5_soh_percent", "legacy_source_file",
        "p5_step4_source_file", "p5_step4_segment_ref", "p5_step10_source_file",
        "p5_step10_segment_ref",
    ]
    label_fields = [
        "dataset_part", "physical_cell_id", "rpt_id", "legacy_soh_percent", "p5_soh_percent",
        "p5_minus_legacy_soh_pp", "absolute_delta_pp", "historical_raw_label_percent",
        "p5_native_amphr_qc_percent", "native_qc_equals_legacy_label",
        "historical_bol_source", "p5_bol_step4_source", "bol_source_path_match",
        "p5_step4_source", "p5_step4_segment_ref", "p5_step4_block_sha256",
        "p5_bol_step4_block_sha256",
    ]
    metric_fields = [
        "population", "model", "reference", "observation_count", "physical_cell_count",
        "mae_soh_percentage_points", "rmse_soh_percentage_points", "r2", "note",
    ]

    legacy_architecture_text = json.dumps(old_model_config["architecture"], indent=2)
    alignment_counts_text = "\n".join(
        f"- `{legacy_split} → {p5_split}`: {count} common observations"
        for (legacy_split, p5_split), count in sorted(transition_counts.items())
    )
    legacy_only_text = "\n".join(
        f"- Part {key[0]} {key[1]} RPT{key[2]}"
        for key in sorted(legacy_only, key=lambda item: (int(item[0]), item[1], int(item[2])))
    )
    source_manifest_names = ", ".join(f"`{path.name}`" for path in verified_manifests)

    reports: dict[str, str] = {}
    reports["legacy_experiment_discovery.md"] = f"""# Legacy experiment discovery

## Finding

The high historical three-encoder result is the saved Phase 2 fusion checkpoint
(`bafuse-three-encoder-v1`, seed 211, best epoch {legacy_checkpoint.get("best_epoch")}),
not a result from the later multi-seed M4 ablation. The checkpoint is
`../bafuse_three_encoder_v1/fusion_checkpoint.pt`; its model, split, scaler,
and training metadata were read without modification.

The 73-row historical reference is a corrected/reconstructed evaluation
population: 64 observations from the original fusion cohort plus nine
supplemental G10-Cell2 observations. The supplemental outputs declare
`prediction_source=saved_checkpoints_inference_only` and
`model_retrained=False`. The corrected audit documents that G10-Cell2 identity
was user-confirmed because the Guide omits an explicit cell number; those nine
rows therefore have a distinct provenance caveat.

## Recorded results and cohorts

| Cohort/model | Train | Validation | External | External MAE (pp) | RMSE (pp) | R² |
|---|---:|---:|---:|---:|---:|---:|
| Legacy Phase 2 fusion | 106 | 41 | 64 original | 1.4867 | 1.8779 | 0.7760 |
| Legacy corrected reference | 106 | 41 | 73 | {_number(_f(recorded_external["mae_soh_percentage_points"]), 7)} | {_number(_f(recorded_external["rmse_soh_percentage_points"]), 7)} | {_number(_f(recorded_external["r2"]), 7)} |
| Frozen P5 cohort | 102 | 37 | 64 | {_number(p5_saved_external_metrics["mae_soh_percentage_points"], 7)} | {_number(p5_saved_external_metrics["rmse_soh_percentage_points"], 7)} | {_number(p5_saved_external_metrics["r2"], 7)} |

The P5 comparison uses the already-saved seed-123 CNN1D predictions; this audit
does not train or select a model. The exact common-external comparison is in
[`legacy_vs_p5_common_cohort_metrics.csv`](./legacy_vs_p5_common_cohort_metrics.csv).

## Reproducibility and boundaries

This audit independently loaded the saved legacy checkpoint and regenerated
predictions for all {len(reproduced)} historical cohort rows from the archived
feature tables plus the audited supplemental raw curves. The maximum absolute
prediction difference against the saved legacy outputs was
`{maximum_reproduction_delta:.12g}` SoH percentage points
({exact_reproduction_count}/{len(reproduced)} exact scalar matches). No training,
tuning, data/label edits, checkpoint edits, or external-based model selection
was performed.

Verified artifact manifests: {source_manifest_names}. See
[`forensic_audit_sha256_manifest.json`](./forensic_audit_sha256_manifest.json)
for exact source, code, configuration, and output hashes.

## Evidence sources

- [Phase 2 training and checkpoint creation](../bafuse_three_encoder_v1/run_phase2.py#L760) (`_train`)
- [Supplemental checkpoint inference](../bafuse_three_encoder_v1/evaluation_audit_v1/audit_soh_evaluation.py#L391) (`_infer_cell2`)
- [Corrected 73-row report](../bafuse_three_encoder_v1/evaluation_audit_v1/corrected_soh_evaluation_report.md)
- [Historical reference metrics](../bafuse_three_encoder_ablation_v1/historical_reference_metrics.csv)
"""

    reports["legacy_vs_p5_architecture_comparison.md"] = f"""# Legacy versus P5 architecture comparison

Both models use independent discharge and charge encoders, an independent
five-feature physics encoder, 32-dimensional modality embeddings, a 96-wide
concatenation, and a 64→32→1 fusion head. They have the same measured
trainable-parameter count ({legacy_parameter_count:,} for legacy and
{p5_parameter_count:,} for P5 CNN1D), but are not architecturally identical.

| Component | Legacy three-encoder | P5 CNN1D |
|---|---|---|
| Curve stem | Conv1D 1→16, k=9, GELU, max-pool 2; Conv1D 16→32, k=5, GELU; adaptive average pool | Same channel/kernel/pooling layout, ReLU instead of GELU |
| Physics branch | Linear 5→32→32 with GELU | Linear 5→32→32 with ReLU |
| Fusion | Linear 96→64, GELU, Dropout(0.2), Linear 64→32, GELU, Linear 32→1 | Linear 96→64→32→1 with ReLU; no dropout |
| Output | One continuous SoH value; no output activation | One continuous SoH value; no output activation |
| Parameter count | {legacy_parameter_count:,} | {p5_parameter_count:,} |

Legacy architecture configuration:

```json
{legacy_architecture_text}
```

P5 architecture configurations are defined in
[`p5_benchmark_models.py`](../../../src/bafuse/hybrid/p5_benchmark_models.py#L23)
(`CNNVoltageEncoder`, `P5MultimodalSoH`, `build_model`). The four benchmark
families share the same `(N,128)`, `(N,128)`, `(N,5)` inputs; this comparison
uses only the CNN1D implementation.

Legacy implementation: [`soh_fusion.py`](../../../src/bafuse/hybrid/soh_fusion.py#L112)
(`CurveEncoder`, `ThreeEncoderSoH`); P5 implementation:
[`p5_benchmark_models.py`](../../../src/bafuse/hybrid/p5_benchmark_models.py#L134).
Similar dimensions do not make the models equivalent: activation, dropout,
physics features, preprocessing, target scaling, and encoder initialization
also differ.
"""

    reports["legacy_vs_p5_dataset_comparison.md"] = f"""# Legacy versus P5 dataset comparison

## Cohort sizes and overlap

| Split | Legacy historical cohort | Frozen P5 cohort |
|---|---:|---:|
| Train | 106 | 102 |
| Validation | 41 | 37 |
| Part 3 External Test | 73 | 64 |
| Total | 220 | 203 |

All {len(p5_keys)} P5 keys occur in the legacy 220-key manifest. There are
{len(legacy_only)} legacy-only keys and {len(p5_only)} P5-only keys. P5 is
therefore a subset, not the same cohort. The legacy-only set includes nine
supplemental G10-Cell2 RPTs and these additional observations:

{legacy_only_text}

Among the {len(common_keys)} common keys, partition transitions are:

{alignment_counts_text}

{changed_split_count} of the 139 common internal keys changed between Train and
Validation. The 64 common Part 3 external keys remain External in both.
Neither cohort splits a physical cell across partitions:

| Cohort | Train/Validation overlap | Train/External overlap | Validation/External overlap |
|---|---:|---:|---:|
| Legacy | {_overlap_counts(old_cells)["train_vs_validation"]} | {_overlap_counts(old_cells)["train_vs_external_test"]} | {_overlap_counts(old_cells)["validation_vs_external_test"]} |
| P5 | {_overlap_counts(p5_cells)["train_vs_validation"]} | {_overlap_counts(p5_cells)["train_vs_external_test"]} | {_overlap_counts(p5_cells)["validation_vs_external_test"]} |

The P5 physical-cell split has 12 internal cells and 11 external cells. The
legacy corrected cohort has 12 internal cells and 12 external cells, including
G10-Cell2. Every key and source reference is recorded in
[`legacy_vs_p5_observation_alignment.csv`](./legacy_vs_p5_observation_alignment.csv).

## Target alignment

For all 203 common observations, the P5 frozen BoL Step 4 source matches the
historical raw label table's same-cell baseline path ({baseline_matches}/203).
The P5 `soh_native_amphr_qc_percent` exactly matches the legacy manifest label
for {native_qc_exact_matches}/203. The P5 primary value differs slightly
because it uses time-integrated current rather than the historical native
`Amphr` span: mean absolute difference {_number(label_delta_mean, 7)} pp,
median signed change (P5 − legacy) {_number(signed_label_delta_median, 7)} pp,
median absolute difference {_number(label_delta_median, 7)} pp, maximum
absolute difference {_number(label_delta_max, 7)} pp, and exact primary-value
matches {label_delta_exact}/203. These are reconstructed same-cell
Step 4-discharge retention labels, not official Oxford ground truth.

See [`legacy_vs_p5_label_comparison.csv`](./legacy_vs_p5_label_comparison.csv)
for observation-level values, BoL paths, and Step 4 provenance.
"""

    reports["legacy_vs_p5_feature_comparison.md"] = f"""# Legacy versus P5 feature comparison

## Curve modalities

Both models consume same-RPT Step 4 discharge and Step 10 charge voltage
curves, each resampled to 128 values. P5's row-level audit records normalized
source-row index as the progress coordinate. It does not normalize the curve
axis by same-RPT capacity. Maximum absolute difference between the audited
legacy and P5 resampled curves is {max_step4_curve_delta:.9g} V for Step 4 and
{max_step10_curve_delta:.9g} V for Step 10 across the 203 common observations.
These small differences support matching curve inputs, not identity of all
downstream processing.

## Physics branch

| Legacy five-feature schema | P5 five-feature schema |
|---|---|
| Mean Step 4 temperature (native units) | {p5_feature_names[0] if len(p5_feature_names) > 0 else "Mean absolute Step 4 current"} |
| Step 4 temperature rise (native units) | {p5_feature_names[1] if len(p5_feature_names) > 1 else "Step 4 current variability"} |
| Voltage slope over 20–80% progress | {p5_feature_names[2] if len(p5_feature_names) > 2 else "Step 4 voltage slope"} |
| Robust spread of voltage derivative over 10–90% | {p5_feature_names[3] if len(p5_feature_names) > 3 else "Step 4 voltage derivative MAD"} |
| Mean absolute Step 4 current (native units) | {p5_feature_names[4] if len(p5_feature_names) > 4 else "Step 4 voltage segment contrast"} |

P5 excludes temperature because temperature semantics and cross-Part
availability were not established consistently; it does not silently impute
temperature. P5 also replaces the old temperature summaries with a
current-variability summary and voltage-segment contrast. No plateau fraction
is present in the P5 five-feature schema.

Legacy code defines its five features in
[`soh_fusion.py`](../../../src/bafuse/hybrid/soh_fusion.py#L61)
(`build_physics_features`). P5 feature calculations and names are in
[`p5_electrical_features.py`](../../../src/bafuse/hybrid/p5_electrical_features.py).
These physics features are engineered summaries of the same-RPT electrical
signals; they are not independent sensors or separate ground truth.

## Current-feature domain shift

For mean absolute current, the P5 audit's raw Train mean is
{float(np.mean(raw_current_train)):.9g} native units and External mean is
{float(np.mean(raw_current_external)):.9g}; the P5 standardized External
values have mean {legacy_current["p5"]["external_z_mean"]:.4g}, range
[{legacy_current["p5"]["external_z_min"]:.4g},
{legacy_current["p5"]["external_z_max"]:.4g}], and
{current_external_over_three}/64 have absolute z-score above 3. The old
checkpoint uses a similarly small current-feature scale
({old_current_scale:.9g}); applying its saved Train-only scaler to the same
External currents gives mean z={legacy_current["legacy"]["external_z_mean"]:.4g},
range [{legacy_current["legacy"]["external_z_min"]:.4g},
{legacy_current["legacy"]["external_z_max"]:.4g}], with
{old_current_external_over_three}/64 above |z|=3. This is a descriptive
distribution-shift diagnostic, not a physical threshold or proof of a cause.
Because the standardized shift also exists for the legacy model, it is not by
itself an explanation for the P5-versus-legacy performance gap.

The P5 row-level provenance and feature audit is available in
`../bafuse_architecture_benchmark_v1/bafuse_p5_electrical_v1_20261114/bafuse_p5_feature_audit.csv`.
"""

    reports["legacy_vs_p5_preprocessing_comparison.md"] = f"""# Legacy versus P5 preprocessing comparison

| Stage | Legacy three-encoder | P5 CNN1D |
|---|---|---|
| Curve resampling | 128-point linear resampling over normalized source-row progress | Same 128-point linear representation; progress basis explicitly recorded as normalized source-row index |
| Curve amplitude | Standardized each voltage point using the legacy Train fit | Leaves voltage in volts |
| Physics | Standardized five legacy physics features using Train fit | Standardizes five P5 physics features using Train fit |
| Target | Train-standardized SoH percentage; inverse-transformed at prediction | No target transform; raw SoH percentage points |
| Fit scope | Legacy training partition (106 observations) | Frozen Train partition only ({p5_scaler_fit_rows} observations) |
| Validation/External | No scaler refit reported | Frozen Train parameters unchanged; no Validation or External fitting |

P5's saved preprocessing declares `fit_partition={p5_scaler_fit_partition}`,
`fit_row_count={p5_scaler_fit_rows}`, and
`validation_and_external_fit_or_update={preprocessing.get("validation_and_external_fit_or_update")}`.
The benchmark config records target transform
`{p5_training.get("target_transform")}`. No scaler or feature transformation
was refit during this audit.

Legacy scaler fitting and application are in
[`run_phase2.py`](../bafuse_three_encoder_v1/run_phase2.py#L760)
(`_train`, scaler construction/application); P5's frozen checks are in
[`p5_benchmark.py`](../../../src/bafuse/hybrid/p5_benchmark.py#L168)
(`_validate_config`) and its training runner. The P5 curve-progress basis is
not target-derived; the audit records no target/BoL-derived predictors.
"""

    reports["legacy_vs_p5_training_comparison.md"] = f"""# Legacy versus P5 training comparison

| Setting | Legacy Phase 2 fusion | P5 CNN1D benchmark |
|---|---|---|
| Data | 106 Train / 41 Validation; 73 in corrected External report | 102 Train / 37 Validation; 64 External |
| Random seeds | One saved run, seed {legacy_checkpoint.get("training_seed")} | Seeds {", ".join(map(str, p5_training.get("seeds", [])))} |
| Max epochs / patience | {old_model_config.get("max_epochs")} / {old_model_config.get("early_stopping_patience")} | {p5_training.get("max_epochs")} / {p5_training.get("early_stopping_patience")} |
| Best epoch / selection | Epoch {legacy_checkpoint.get("best_epoch")}; minimum Validation loss | Validation RMSE checkpoint selection |
| Batch / optimizer | {old_model_config.get("batch_size")}; {old_model_config.get("optimizer")} | {p5_training.get("batch_size")}; {p5_training.get("optimizer")} |
| Learning rate / weight decay | 0.001 / 0.0001 | {p5_training.get("learning_rate")} / {p5_training.get("weight_decay")} |
| Target/loss | Train-standardized target, MSE | Raw SoH percent, MSE |
| Initialization | Discharge encoder initialized from historical Part 1 CNN1D; fusion branches then trained | Fresh model initialization per seed |

The P5 benchmark was an existing saved run. This forensic pass did not retrain,
tune, change a checkpoint, or use External results for selection. The two
experiments differ in split membership, input features, architecture
activations/dropout, input/target scaling, initialization, seeds, and selection
procedure. This is not a controlled experiment that isolates a single cause.

Source references: legacy [`run_phase2.py`](../bafuse_three_encoder_v1/run_phase2.py#L760)
(`_train`) and P5 [`p5_benchmark.py`](../../../src/bafuse/hybrid/p5_benchmark.py#L682)
(`_train_one_run`).
"""

    reports["legacy_leakage_audit.md"] = f"""# Legacy and P5 leakage audit

## Findings

- **No direct target or BoL leakage was established in reviewed model inputs.**
  The legacy model receives only same-RPT discharge voltage, charge voltage,
  and five derived physics values. Its configuration explicitly excludes
  Step 4 discharge duration because duration can reconstruct same-RPT
  discharge capacity.
- **P5's row-level audit marks all 203 observations**
  `predictive_label_or_bol_feature=false` and `future_rpt_feature=false`.
  The curve coordinate is normalized source-row index, not discharged
  capacity; capacity, native `Amphr`, integrated capacity, BoL capacity, SoH,
  cell ID, RPT ID, and Part ID are not model tensors.
- **No future-RPT input was found.** Charge and discharge are from the same
  RPT. This is same-time SOH estimation, not future forecasting.
- **No physical-cell split overlap was found** in either cohort. However, the
  internal cell assignments differ substantially between the historical and
  P5 cohorts; these assignments are not interchangeable.
- **Potential confounding remains, not confirmed leakage.** The legacy model
  uses temperature summaries whose sensor semantics and cross-Part context are
  uncertain. Part, instrument/protocol, and temperature can be associated
  with one another; the model has no explicit Part-ID field, but observed
  signals may still carry domain information.
- **External evaluation boundary:** P5 predictions in this report are existing
  saved outputs. External values were not used to fit preprocessing, select
  features, tune architecture, train, or choose checkpoints in this audit.

## Code and evidence

Legacy input exclusion is documented in
[`soh_fusion.py`](../../../src/bafuse/hybrid/soh_fusion.py#L20)
(`EXCLUDED_FEATURES`). The actual feature builder and encoders are at
[`build_physics_features`](../../../src/bafuse/hybrid/soh_fusion.py#L61) and
[`ThreeEncoderSoH`](../../../src/bafuse/hybrid/soh_fusion.py#L133). The Phase 2
trainer applies the input/target scalers and sends only the three modalities
to the model in [`run_phase2.py`](../bafuse_three_encoder_v1/run_phase2.py#L801).
Supplemental G10 predictions were generated by the saved checkpoint in
`_infer_cell2`, which records `saved_checkpoints_inference_only` and
`model_retrained=False` in
[`audit_soh_evaluation.py`](../bafuse_three_encoder_v1/evaluation_audit_v1/audit_soh_evaluation.py#L391).

P5's persisted `bafuse_p5_feature_audit.csv` gives same-RPT Step 4/Step 10
segment references, raw-block hashes, feature values, curve coordinate, and
the leakage/future flags for every row. It records no positive target/BoL or
future flags. This evidence does not prove that every undocumented raw signal
has known physical semantics; it supports only the specific reviewed pipeline.

## Conclusion

No confirmed direct leakage explains the performance difference. The
temperature/Part/instrument association is a domain-confounding concern that
requires a controlled, training-authorized experiment to quantify. No
corrective data or model change is made here.
"""

    _write_csv(OUT / "legacy_vs_p5_observation_alignment.csv", alignment_fields, alignment_rows)
    _write_csv(OUT / "legacy_vs_p5_label_comparison.csv", label_fields, label_comparison)
    _write_csv(OUT / "legacy_vs_p5_common_cohort_metrics.csv", metric_fields, metrics_rows)

    reports["legacy_checkpoint_reproducibility.md"] = f"""# Legacy checkpoint reproducibility

## Procedure

Loaded the saved `fusion_checkpoint.pt` with the exact
`bafuse-three-encoder-v1` architecture, strict state-dict loading, saved
discharge/charge/physics scalers, and saved target inverse transform. Inference
ran in evaluation/inference mode; no optimizer, gradient, training loop, or
checkpoint write was used. Features came from the archived legacy feature
tables for the historical cohort plus the verified raw Step 4/Step 10
segments for the nine supplemental G10-Cell2 rows.

## Results

- Reproduced rows: {len(reproduced)} of 220.
- Original Phase 2 predictions: {sum(key in _assert_unique(_read_csv(LEGACY / "fusion_predictions.csv"), "legacy saved predictions") for key in old_keys)} keyed observations.
- Supplemental saved-checkpoint predictions: 9 eligible G10-Cell2 rows.
- Exact scalar prediction matches: {exact_reproduction_count}/{len(reproduced)}.
- Maximum absolute difference from saved outputs:
  `{maximum_reproduction_delta:.12g}` SoH percentage points.
- Reproduced corrected 73-row external metrics: MAE
  `{reproduced_legacy_external["mae_soh_percentage_points"]:.12g}`, RMSE
  `{reproduced_legacy_external["rmse_soh_percentage_points"]:.12g}`, R²
  `{reproduced_legacy_external["r2"]:.12g}`.
- Recorded historical 73-row reference: MAE
  `{_f(recorded_external["mae_soh_percentage_points"]):.12g}`, RMSE
  `{_f(recorded_external["rmse_soh_percentage_points"]):.12g}`, R²
  `{_f(recorded_external["r2"]):.12g}`.
- Absolute metric differences from the recorded row: MAE
  `{original_metrics_delta["mae_abs_difference"]:.12g}`, RMSE
  `{original_metrics_delta["rmse_abs_difference"]:.12g}`, R²
  `{original_metrics_delta["r2_abs_difference"]:.12g}`.

The exact common 64-key comparison and alternate-reference checks are in
[`legacy_vs_p5_common_cohort_metrics.csv`](./legacy_vs_p5_common_cohort_metrics.csv).
The 73-row result is a corrected historical cohort, not the original 64-row
Phase 2 External population.
"""

    hypotheses = [
        (
            "Different cohort/split",
            "CONFIRMED DIFFERENCE; NOT A FULL EXPLANATION",
            "Legacy has 73 corrected External rows; P5 has 64. All 64 P5 External keys are common and the legacy checkpoint still outperforms P5 on those exact keys under the same P5 labels. Internal split membership differs for 77 common keys.",
        ),
        (
            "Target/label construction",
            "DIFFERENT CONSTRUCTION; NOT SUPPORTED AS MAIN CAUSE",
            f"Same-cell BoL paths match {baseline_matches}/203; the legacy-native Amphr QC value exactly matches legacy labels for {native_qc_exact_matches}/203. Primary integrated-current versus native-Amphr values differ by mean absolute {label_delta_mean:.6g} pp and maximum {label_delta_max:.6g} pp.",
        ),
        (
            "Physics feature redesign",
            "CONFIRMED DIFFERENCE; CONTRIBUTION UNIDENTIFIED",
            "Legacy uses two temperature features; P5 excludes temperature and uses current variability plus voltage-segment contrast. The feature sets differ materially, but no controlled ablation was run.",
        ),
        (
            "Curve preprocessing",
            "CONFIRMED DIFFERENCE; CONTRIBUTION UNIDENTIFIED",
            f"Legacy standardizes every curve point; P5 leaves curves in volts. Raw resampled curves are very close (max Step 4 {max_step4_curve_delta:.3g} V, Step 10 {max_step10_curve_delta:.3g} V), so the scaling difference remains a plausible model-input difference.",
        ),
        (
            "External current distribution shift",
            "CONFIRMED DESCRIPTIVE SHIFT; INSUFFICIENT AS SOLE EXPLANATION",
            f"External current has large standardized values under both P5 ({current_external_over_three}/64 above |z|=3) and legacy ({old_current_external_over_three}/64); the legacy model performs well despite the same signal shift.",
        ),
        (
            "Architecture/regularization",
            "CONFIRMED DIFFERENCE; CONTRIBUTION UNIDENTIFIED",
            "GELU plus Dropout(0.2) in the legacy fusion branch versus ReLU without dropout in P5; equal parameter counts do not remove this difference.",
        ),
        (
            "Training procedure",
            "CONFIRMED DIFFERENCE; CONTRIBUTION UNIDENTIFIED",
            f"Legacy is one seed ({legacy_checkpoint.get('training_seed')}) with target standardization and a Part 1 discharge-encoder warm start; P5 CNN1D uses saved seed 123 among three fresh-init runs and raw-percent targets.",
        ),
        (
            "Direct target/future leakage",
            "NOT SUPPORTED BY REVIEWED PIPELINE",
            "Capacity, BoL, SoH, identifiers, and future-RPT features are excluded; Step 4 duration is explicitly excluded. P5 row-level audits mark no target/BoL-derived or future features.",
        ),
        (
            "Part/instrument domain effects",
            "PRESENT AS CONTEXT; CAUSAL ROLE UNIDENTIFIED",
            "Part 3 uses a different instrument/protocol and the README temperature context differs/omits detail. Both models are compared on the same 64 Part 3 keys, so domain shift alone does not explain the between-model gap.",
        ),
    ]
    reports["performance_degradation_hypotheses.md"] = "# Performance-gap hypotheses\n\n"
    reports["performance_degradation_hypotheses.md"] += _format_table(
        ["Hypothesis", "Status", "Evidence / limitation"],
        [[name, status, evidence] for name, status, evidence in hypotheses],
    )
    reports["performance_degradation_hypotheses.md"] += """

## Interpretation

The legacy and P5 checkpoints are not a controlled A/B experiment. Differences
in partition membership, features, per-point scaling, target scaling,
activation/dropout, warm start, seed count, and checkpoint selection are
confounded. The clear falsifications are that the 9 extra historical External
rows do not account for the difference on the common 64 keys, and small target
numerical differences do not plausibly account for the full gap. The present
evidence does not justify attributing the result to one architectural or
feature choice. No new training or tuning was authorized or performed.
"""

    reports["forensic_audit_summary.md"] = f"""# Forensic audit summary

## Scope and outcome

Read-only comparison of the historical three-encoder CNN1D and frozen P5
CNN1D, including saved-checkpoint inference to reproduce legacy outputs.
Training, tuning, external model selection, dataset/label edits, and checkpoint
edits were not performed.

- Historical legacy cohort: 220 observations (106 Train, 41 Validation, 73
  corrected External).
- Frozen P5 cohort: 203 observations (102 Train, 37 Validation, 64 External)
  across 23 physical cells.
- Overlap: {len(common_keys)} common keys; {len(legacy_only)} legacy-only;
  {len(p5_only)} P5-only.
- Internal split: {changed_split_count} common internal observations changed
  Train/Validation assignment; physical-cell overlap is zero in both cohorts.
- Label provenance: same-cell historical/P5 BoL paths match
  {baseline_matches}/203; P5 native-Amphr QC matches the historical label for
  {native_qc_exact_matches}/203; primary label differences are small
  (mean absolute {_number(label_delta_mean, 7)} pp, max
  {_number(label_delta_max, 7)} pp).
- Legacy checkpoint reproduction: {len(reproduced)}/220 rows, maximum saved
  prediction difference `{maximum_reproduction_delta:.12g}` pp.
- Same 64 External keys, same P5 target: legacy MAE
  `{legacy_common_p5_label_metrics["mae_soh_percentage_points"]:.6g}` pp,
  RMSE `{legacy_common_p5_label_metrics["rmse_soh_percentage_points"]:.6g}`
  pp, R² `{legacy_common_p5_label_metrics["r2"]:.6g}`; P5 CNN1D seed 123 MAE
  `{p5_saved_external_metrics["mae_soh_percentage_points"]:.6g}` pp, RMSE
  `{p5_saved_external_metrics["rmse_soh_percentage_points"]:.6g}` pp, R²
  `{p5_saved_external_metrics["r2"]:.6g}`.
- Direct leakage: none confirmed in the reviewed feature/target flow; domain
  confounding remains possible.
- Primary explanation: not identifiable from these non-controlled runs.

## Findings

1. The `.81` R²/1.79 pp RMSE record is the legacy checkpoint's corrected
   73-row External re-evaluation, not the original 64-row Phase 2 result and
   not the later multi-seed M4 ablation.
2. The common external population is exactly 64 observations across 11
   physical cells. Legacy performance remains much better than the saved P5
   seed-123 prediction on these same observations and P5 labels.
3. Target values are aligned in definition and same-cell BoL provenance.
   The P5-native-Amphr QC labels exactly match the legacy values; the primary
   P5 integral differs only slightly.
4. The models differ in physics feature semantics, voltage scaling, target
   scaling, activation/dropout, warm-start, seeds, and validation selection.
   Those effects cannot be separated from the present artifacts.
5. No direct target leakage, future-RPT input, or cell split overlap was
   established. Temperature/Part/instrument association is an unresolved
   domain-confounding concern, not proof of leakage.

## Detailed artifacts

- [Experiment discovery](./legacy_experiment_discovery.md)
- [Architecture comparison](./legacy_vs_p5_architecture_comparison.md)
- [Dataset comparison](./legacy_vs_p5_dataset_comparison.md)
- [Observation alignment](./legacy_vs_p5_observation_alignment.csv)
- [Label comparison](./legacy_vs_p5_label_comparison.csv)
- [Feature comparison](./legacy_vs_p5_feature_comparison.md)
- [Preprocessing comparison](./legacy_vs_p5_preprocessing_comparison.md)
- [Training comparison](./legacy_vs_p5_training_comparison.md)
- [Leakage audit](./legacy_leakage_audit.md)
- [Common-cohort metrics](./legacy_vs_p5_common_cohort_metrics.csv)
- [Checkpoint reproducibility](./legacy_checkpoint_reproducibility.md)
- [Performance-gap hypotheses](./performance_degradation_hypotheses.md)
- [SHA-256 manifest](./forensic_audit_sha256_manifest.json)

## Runtime and authorization

- Runtime: Python {runtime["python"]}, NumPy {runtime["numpy"]}, PyTorch
  {runtime["torch"]}, device `{runtime["device"]}` ({runtime["cuda_device"]}).
- `training_authorized: false`; `external_prediction_evaluation_authorized: false`.
- Existing saved P5 predictions were read only. Legacy checkpoint inference was
  performed only for the explicit reproducibility comparison.
- This report does not authorize or recommend training. A causal attribution
  would require a separately authorized controlled experiment.
"""

    for name in EXPECTED_OUTPUTS:
        if name.endswith(".md"):
            _write_text(OUT / name, reports[name])

    source_hashes_after = _verify_sources(roles, declared_hashes)
    changed_sources = [
        _relative(path)
        for path in source_hashes_before
        if source_hashes_before[path] != source_hashes_after[path]
    ]
    if changed_sources:
        raise RuntimeError(f"Audit inputs changed during this read-only run: {changed_sources}")

    output_hashes = {
        name: _sha256(OUT / name)
        for name in EXPECTED_OUTPUTS
    }
    manifest_document = {
        "manifest_version": 1,
        "audit_identifier": "bafuse-legacy-p5-forensic-audit-v1-20261114",
        "generated_at_utc": runtime["utc_created_at"],
        "scope": {
            "training_performed": False,
            "tuning_performed": False,
            "dataset_or_label_modified": False,
            "checkpoint_modified": False,
            "p5_predictions_rerun": False,
            "legacy_checkpoint_inference_for_reproducibility": True,
            "external_model_selection_performed": False,
        },
        "verified_artifact_manifests": [
            {
                "path": _relative(path),
                "sha256": source_hashes_before[path.resolve()],
            }
            for path in verified_manifests
        ],
        "runtime": runtime,
        "inputs": [
            {
                "path": _relative(path),
                "sha256": source_hashes_before[path],
                "roles": sorted(roles[path]),
                "declared_manifest_sha256": declared_hashes.get(path),
            }
            for path in sorted(roles, key=lambda value: _relative(value))
            if path != Path(__file__).resolve()
        ],
        "code": {
            "path": _relative(Path(__file__)),
            "sha256": source_hashes_before[Path(__file__).resolve()],
        },
        "outputs": output_hashes,
        "manifest_self_hash": "excluded; calculate after writing this file",
        "summary_checks": {
            "legacy_observations": len(legacy_by_key),
            "p5_observations": len(p5_keys),
            "common_observations": len(common_keys),
            "legacy_only_observations": len(legacy_only),
            "p5_only_observations": len(p5_only),
            "baseline_path_matches": baseline_matches,
            "native_amphr_exact_label_matches": native_qc_exact_matches,
            "legacy_checkpoint_reproduced_observations": len(reproduced),
            "legacy_checkpoint_max_abs_prediction_delta_pp": maximum_reproduction_delta,
            "same_external_keys": len(external_common),
            "train_only_p5_scaler_rows": int(p5_scaler_fit_rows),
            "physical_cell_split_overlap": {
                "legacy": _overlap_counts(old_cells),
                "p5": _overlap_counts(p5_cells),
            },
        },
    }
    _write_json(OUT / "forensic_audit_sha256_manifest.json", manifest_document)
    manifest_sha = _sha256(OUT / "forensic_audit_sha256_manifest.json")
    print(
        json.dumps(
            {
                "audit_identifier": manifest_document["audit_identifier"],
                "outputs": sorted(list(output_hashes) + ["forensic_audit_sha256_manifest.json"]),
                "manifest_sha256": manifest_sha,
                "legacy_reproduction_max_abs_delta_pp": maximum_reproduction_delta,
                "legacy_reproduction_exact_matches": exact_reproduction_count,
                "common_external_metrics": {
                    "legacy_mae": legacy_common_p5_label_metrics["mae_soh_percentage_points"],
                    "legacy_rmse": legacy_common_p5_label_metrics["rmse_soh_percentage_points"],
                    "legacy_r2": legacy_common_p5_label_metrics["r2"],
                    "p5_mae": p5_saved_external_metrics["mae_soh_percentage_points"],
                    "p5_rmse": p5_saved_external_metrics["rmse_soh_percentage_points"],
                    "p5_r2": p5_saved_external_metrics["r2"],
                },
                "input_hashes_unchanged": not changed_sources,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
