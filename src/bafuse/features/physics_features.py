from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Sequence

import numpy as np


PHYSICS_FEATURE_NAMES = (
    "step4_temp1_mean_C",
    "step4_temp1_max_C",
    "step4_temp1_range_C",
    "step4_temp1_mean_minus_24C",
    "step6_temp1_mean_C",
    "step6_temp1_max_C",
    "step6_temp1_range_C",
    "step6_temp1_mean_minus_24C",
)
PHYSICS_ENCODER_INPUT_COUNT = len(PHYSICS_FEATURE_NAMES) * 2


def _normalize_source(value: str) -> str:
    return PureWindowsPath(value.strip()).as_posix().casefold()


def _parse_feature(value: Any, name: str, source: str) -> float | None:
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return None
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"Invalid {name} value for {source}: {value!r}") from error
    return parsed if math.isfinite(parsed) else None


def _parse_availability(value: Any, source: str) -> bool | None:
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"Invalid physics availability value for {source}: {value!r}")


def build_physics_feature_rows(
    source_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create one target-free Temp1 feature row per physical-cell RPT source."""
    by_source: dict[tuple[str, str], dict[str, Any]] = {}
    for source_row in source_rows:
        cell = str(source_row.get("physical_cell_id", "")).strip()
        source = str(source_row.get("source_file", "")).strip()
        measurement = str(source_row.get("measurement_id", "")).strip()
        if not cell or not source:
            raise ValueError("Physics source rows require physical_cell_id and source_file.")

        values = {
            name: _parse_feature(source_row.get(name), name, source)
            for name in PHYSICS_FEATURE_NAMES
        }
        source_available = _parse_availability(
            source_row.get("physics_temp1_available"), source
        )
        available_count = sum(value is not None for value in values.values())
        if source_available is True and available_count != len(PHYSICS_FEATURE_NAMES):
            raise ValueError(f"Source marks incomplete Temp1 features available: {source}")
        if source_available is False and available_count:
            raise ValueError(f"Source marks populated Temp1 features unavailable: {source}")

        key = (cell.casefold(), _normalize_source(source))
        existing = by_source.get(key)
        if existing is not None:
            if existing["measurement_id"] != measurement:
                raise ValueError(f"Repeated source rows disagree on measurement_id: {source}")
            for name, value in values.items():
                if existing[name] != value:
                    raise ValueError(f"Repeated source rows disagree on {name}: {source}")
            continue

        status = (
            "available"
            if available_count == len(PHYSICS_FEATURE_NAMES)
            else "partial"
            if available_count
            else "missing"
        )
        row: dict[str, Any] = {
            "physical_cell_id": cell,
            "measurement_id": measurement,
            "source_file": source,
            "physics_feature_status": status,
        }
        for name, value in values.items():
            row[name] = "" if value is None else value
            row[f"{name}_available"] = value is not None
        by_source[key] = row

    if not by_source:
        raise ValueError("No physics feature rows were available in the source manifest.")
    return [
        by_source[key]
        for key in sorted(by_source)
    ]


def physics_feature_matrix(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    if not rows:
        raise ValueError("Cannot build a physics feature matrix from zero rows.")
    matrix = []
    for row in rows:
        source = str(row.get("source_file", ""))
        values = []
        for name in PHYSICS_FEATURE_NAMES:
            value = _parse_feature(row.get(name), name, source)
            values.append(np.nan if value is None else value)
        matrix.append(values)
    return np.asarray(matrix, dtype=np.float64)


@dataclass(frozen=True)
class PhysicsFeatureScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, training_values: np.ndarray) -> PhysicsFeatureScaler:
        values = np.asarray(training_values, dtype=np.float64)
        expected_shape = (len(training_values), len(PHYSICS_FEATURE_NAMES))
        if values.ndim != 2 or values.shape != expected_shape:
            raise ValueError(
                f"Expected training features with shape [rows, {len(PHYSICS_FEATURE_NAMES)}]."
            )
        finite = np.isfinite(values)
        counts = finite.sum(axis=0)
        missing_columns = [
            PHYSICS_FEATURE_NAMES[index]
            for index, count in enumerate(counts)
            if count == 0
        ]
        if missing_columns:
            raise ValueError(
                "Training split has no finite observations for: "
                + ", ".join(missing_columns)
            )
        mean = np.where(finite, values, 0.0).sum(axis=0) / counts
        centered = np.where(finite, values - mean, 0.0)
        scale = np.sqrt((centered * centered).sum(axis=0) / counts)
        scale[scale <= np.finfo(np.float64).eps] = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        raw = np.asarray(values, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != len(PHYSICS_FEATURE_NAMES):
            raise ValueError(
                f"Expected features with shape [rows, {len(PHYSICS_FEATURE_NAMES)}]."
            )
        if self.mean.shape != (len(PHYSICS_FEATURE_NAMES),):
            raise ValueError("Physics scaler mean does not match the feature schema.")
        if self.scale.shape != self.mean.shape or not np.isfinite(self.scale).all():
            raise ValueError("Physics scaler scale is invalid.")
        if np.any(self.scale <= 0):
            raise ValueError("Physics scaler scale must be positive.")
        missing = ~np.isfinite(raw)
        imputed = np.where(missing, self.mean, raw)
        standardized = (imputed - self.mean) / self.scale
        return np.concatenate(
            (standardized, missing.astype(np.float64)), axis=1
        ).astype(np.float32)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_physics_feature_artifacts(
    output_dir: Path,
    source_rows: Sequence[dict[str, Any]],
    source_manifest: Path,
) -> dict[str, Any]:
    rows = build_physics_feature_rows(source_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "physics_features.csv"
    fields = list(rows[0])
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)

    feature_availability = {
        name: sum(bool(row[f"{name}_available"]) for row in rows)
        for name in PHYSICS_FEATURE_NAMES
    }
    manifest = {
        "status": "conditional_ready_not_trained",
        "dataset_scope": "Oxford Path Dependent Battery Degradation Dataset Part 1",
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_rows": len(source_rows),
        "unique_cell_rpt_rows": len(rows),
        "independent_cells": len({row["physical_cell_id"] for row in rows}),
        "feature_names": list(PHYSICS_FEATURE_NAMES),
        "feature_availability_counts": feature_availability,
        "encoder_input_count": PHYSICS_ENCODER_INPUT_COUNT,
        "encoder_input_layout": (
            "Train-fold standardized values for the eight Temp1 summaries, "
            "followed by eight per-feature missingness indicators."
        ),
        "preprocessing_scope": (
            "Fit means and population standard deviations on training cells only; "
            "impute missing values with the corresponding training mean."
        ),
        "alignment": "One row per physical cell and source RPT CSV; direction duplicates are collapsed after equality checks.",
        "target_columns_included": False,
        "non_feature_columns": [
            "physical_cell_id",
            "measurement_id",
            "source_file",
            "physics_feature_status",
            *(f"{name}_available" for name in PHYSICS_FEATURE_NAMES),
        ],
        "limitations": [
            "Temp1 sensor position and calibration are not documented locally.",
            "The field names use degC inferred from the 24 degC protocol setpoint; treat values as conditional candidates.",
            "These are same-RPT operational summaries, not future-time inputs or independent physical ground truth.",
            "The feature table has not been used to train or evaluate the physics encoder.",
        ],
    }
    (output_dir / "physics_feature_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a target-free thermal feature table for the physics encoder."
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.source_manifest.is_file():
        raise FileNotFoundError(f"Source manifest not found: {args.source_manifest}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output}")
    source_rows = _read_csv(args.source_manifest)
    manifest = write_physics_feature_artifacts(
        args.output, source_rows, args.source_manifest
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
