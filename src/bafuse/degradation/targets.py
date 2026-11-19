from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from bafuse.labeling.pydma_policy import pydma_fit_passes_gate
from bafuse.degradation.models import DegradationPredictions


TARGET_NAMES = ("soh", "lli", "lam_anode", "lam_cathode")
PYDMA_ACCEPTED_STATUS = "diagnostic_estimate_accepted"
PYDMA_REFERENCE_STATUS = "verified"


@dataclass(frozen=True)
class LabelProvenance:
    split: str
    dataset_part: str
    group_id: str
    physical_cell_id: str
    rpt: str
    source_artifact: str
    source_row_id: str
    procedure: str
    source_file: str
    mapping_status: str
    curve_status: str
    soh_label_status: str
    diagnostic_label_status: str
    direction: str | None
    direction_policy_id: str | None
    reference_source: str | None
    reference_status: str | None
    fit_rmse_mV: float | None
    accepted_runs: int | None
    fit_is_accepted: bool


@dataclass(frozen=True)
class TargetMaskAudit:
    masks: np.ndarray
    exclusion_reasons: tuple[tuple[str, ...], ...]
    splits: tuple[str, ...]


@dataclass(frozen=True)
class TargetStandardizer:
    mean: np.ndarray
    observed_std: np.ndarray
    scale: np.ndarray
    fit_counts: np.ndarray
    minimum_std: float

    def transform(
        self,
        values: np.ndarray,
        masks: np.ndarray,
    ) -> np.ndarray:
        array = _target_matrix(values, "values")
        valid = _target_mask(masks, array.shape[0])
        _require_finite_observed(array, valid)
        transformed = np.zeros(array.shape, dtype=np.float64)
        transformed[valid] = (
            array[valid] - np.broadcast_to(self.mean, array.shape)[valid]
        ) / np.broadcast_to(self.scale, array.shape)[valid]
        if not np.isfinite(transformed[valid]).all():
            raise ValueError("Target transform produced non-finite observed values.")
        return transformed.astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        array = _target_matrix(values, "values")
        restored = array * self.scale + self.mean
        if not np.isfinite(restored).all():
            raise ValueError("Target inverse transform produced non-finite values.")
        return restored.astype(np.float32)

    def to_dict(self) -> dict[str, object]:
        return {
            "fit_partition": "train",
            "target_names": list(TARGET_NAMES),
            "mean": self.mean.tolist(),
            "observed_population_std": self.observed_std.tolist(),
            "effective_scale": self.scale.tolist(),
            "fit_counts": self.fit_counts.tolist(),
            "minimum_std": self.minimum_std,
            "small_std_policy": "Use effective scale 1.0 below minimum_std.",
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> TargetStandardizer:
        if value.get("fit_partition") != "train":
            raise ValueError("Target normalization must be fitted on Train.")
        if tuple(value.get("target_names", ())) != TARGET_NAMES:
            raise ValueError("Saved target names do not match the V2 target schema.")
        mean = np.asarray(value["mean"], dtype=np.float64)
        observed_std = np.asarray(value["observed_population_std"], dtype=np.float64)
        scale = np.asarray(value["effective_scale"], dtype=np.float64)
        fit_counts = np.asarray(value["fit_counts"], dtype=np.int64)
        if any(array.shape != (len(TARGET_NAMES),) for array in (
            mean, observed_std, scale, fit_counts
        )):
            raise ValueError("Saved target-normalization shapes do not match.")
        minimum_std = float(value["minimum_std"])
        if not (
            np.isfinite(mean).all()
            and np.isfinite(observed_std).all()
            and np.isfinite(scale).all()
            and np.all(scale > 0)
            and np.all(fit_counts > 0)
            and math.isfinite(minimum_std)
            and minimum_std > 0
        ):
            raise ValueError("Saved target-normalization parameters are invalid.")
        return cls(mean, observed_std, scale, fit_counts, minimum_std)


def build_target_masks(
    targets: np.ndarray,
    provenance: Sequence[LabelProvenance],
    *,
    approved_direction_policy_ids: Collection[str],
) -> TargetMaskAudit:
    values = _target_matrix(targets, "targets")
    if len(provenance) != values.shape[0]:
        raise ValueError("Target rows and provenance rows must have the same length.")

    masks = np.zeros(values.shape, dtype=bool)
    row_reasons: list[tuple[str, ...]] = []
    approved_policies = set(approved_direction_policy_ids)
    for row_index, record in enumerate(provenance):
        reasons: list[str] = []
        reasons.extend(_common_provenance_reasons(record))

        if record.soh_label_status != "ACCEPTED":
            reasons.append("soh_label_not_accepted")
        elif not _is_finite_number(values[row_index, 0]):
            reasons.append("soh_target_missing_or_nonfinite")
        elif not _common_provenance_reasons(record):
            masks[row_index, 0] = True

        diagnostic_reasons = _diagnostic_provenance_reasons(
            record, approved_policies
        )
        reasons.extend(diagnostic_reasons)
        if not diagnostic_reasons:
            for target_index, target_name in enumerate(TARGET_NAMES[1:], start=1):
                if _is_finite_number(values[row_index, target_index]):
                    masks[row_index, target_index] = True
                else:
                    reasons.append(f"{target_name}_target_missing_or_nonfinite")
        row_reasons.append(tuple(dict.fromkeys(reasons)))

    return TargetMaskAudit(
        masks=masks,
        exclusion_reasons=tuple(row_reasons),
        splits=tuple(record.split for record in provenance),
    )


def fit_target_standardizer(
    targets: np.ndarray,
    audit: TargetMaskAudit,
    *,
    minimum_std: float = 1e-8,
) -> TargetStandardizer:
    values = _target_matrix(targets, "targets")
    valid = _target_mask(audit.masks, values.shape[0])
    if len(audit.splits) != values.shape[0]:
        raise ValueError("Target rows and provenance splits must have the same length.")
    if not math.isfinite(minimum_std) or minimum_std <= 0:
        raise ValueError("minimum_std must be a positive finite value.")
    train_rows = np.asarray(
        [str(split).strip().casefold() == "train" for split in audit.splits],
        dtype=bool,
    )
    if not train_rows.any():
        raise ValueError("Target normalization requires Train rows.")
    train_mask = valid & train_rows[:, None]
    _require_finite_observed(values, train_mask)

    mean = np.empty(len(TARGET_NAMES), dtype=np.float64)
    observed_std = np.empty(len(TARGET_NAMES), dtype=np.float64)
    scale = np.empty(len(TARGET_NAMES), dtype=np.float64)
    fit_counts = train_mask.sum(axis=0).astype(np.int64)
    for target_index, target_name in enumerate(TARGET_NAMES):
        selected = values[train_mask[:, target_index], target_index]
        if selected.size == 0:
            raise ValueError(
                f"Train has no accepted observations for target {target_name!r}."
            )
        mean[target_index] = selected.mean()
        observed_std[target_index] = selected.std(ddof=0)
        scale[target_index] = (
            observed_std[target_index]
            if observed_std[target_index] >= minimum_std
            else 1.0
        )
    return TargetStandardizer(mean, observed_std, scale, fit_counts, minimum_std)


def masked_training_mse(
    predictions: DegradationPredictions,
    targets: torch.Tensor,
    masks: torch.Tensor,
    provenance: Sequence[LabelProvenance],
    *,
    target_names: Sequence[str] = TARGET_NAMES,
) -> torch.Tensor:
    if targets.ndim != 2 or targets.shape[1] != len(TARGET_NAMES):
        raise ValueError(f"Expected target tensor with shape (batch, {len(TARGET_NAMES)}).")
    if masks.shape != targets.shape or masks.dtype != torch.bool:
        raise ValueError("Target masks must be boolean and match the target tensor.")
    if len(provenance) != targets.shape[0]:
        raise ValueError("Target rows and provenance rows must have the same length.")
    if any(record.split.strip().casefold() != "train" for record in provenance):
        raise ValueError("Training loss accepts Train rows only; external labels are blocked.")
    if tuple(target_names) != tuple(dict.fromkeys(target_names)):
        raise ValueError("Training target names must be unique.")
    unknown = set(target_names) - set(TARGET_NAMES)
    if unknown:
        raise ValueError(f"Unknown training targets: {sorted(unknown)}.")
    if targets.device != masks.device:
        raise ValueError("Targets and masks must be on the same device.")

    prediction_values = predictions.as_tensor()
    if prediction_values.shape != targets.shape:
        raise ValueError("Predictions and targets must have matching shapes.")
    if prediction_values.device != targets.device:
        raise ValueError("Predictions and targets must be on the same device.")
    losses: list[torch.Tensor] = []
    for target_name in target_names:
        target_index = TARGET_NAMES.index(target_name)
        observed = masks[:, target_index]
        if not observed.any():
            continue
        target_values = targets[observed, target_index]
        predicted_values = prediction_values[observed, target_index]
        if not torch.isfinite(target_values).all():
            raise ValueError(f"Observed target {target_name!r} is non-finite.")
        if not torch.isfinite(predicted_values).all():
            raise ValueError(f"Prediction for {target_name!r} is non-finite.")
        losses.append(torch.mean((predicted_values - target_values) ** 2))
    if not losses:
        raise ValueError("Training batch contains no observed targets; refusing no-op loss.")
    return torch.stack(losses).mean()


def _common_provenance_reasons(record: LabelProvenance) -> list[str]:
    reasons: list[str] = []
    required = {
        "split": record.split,
        "dataset_part": record.dataset_part,
        "group_id": record.group_id,
        "physical_cell_id": record.physical_cell_id,
        "rpt": record.rpt,
        "source_artifact": record.source_artifact,
        "source_row_id": record.source_row_id,
        "procedure": record.procedure,
        "source_file": record.source_file,
    }
    if any(not str(value).strip() for value in required.values()):
        reasons.append("provenance_incomplete")
    if _part_number(record.dataset_part) not in {"1", "2", "3"}:
        reasons.append("dataset_part_invalid")
    if record.mapping_status != "verified":
        reasons.append("mapping_not_verified")
    if record.curve_status != "verified":
        reasons.append("curve_not_verified")
    if record.reference_status != PYDMA_REFERENCE_STATUS or not str(
        record.reference_source or ""
    ).strip():
        reasons.append("reference_not_verified")
    if _is_excluded_part2_group(record.dataset_part, record.group_id):
        reasons.append("part2_group5_6_excluded")
    return reasons


def _diagnostic_provenance_reasons(
    record: LabelProvenance,
    approved_direction_policy_ids: set[str],
) -> list[str]:
    reasons = _common_provenance_reasons(record)
    if record.diagnostic_label_status != PYDMA_ACCEPTED_STATUS:
        reasons.append("diagnostic_label_not_accepted")
    if str(record.direction or "").strip().casefold() not in ("charge", "discharge"):
        reasons.append("direction_missing_or_invalid")
    if (
        not record.direction_policy_id
        or record.direction_policy_id not in approved_direction_policy_ids
    ):
        reasons.append("direction_policy_not_approved")
    if (
        not _is_finite_number(record.fit_rmse_mV)
        or not pydma_fit_passes_gate(
            rmse_mV=record.fit_rmse_mV,
            accepted_runs=record.accepted_runs,
            is_accepted=record.fit_is_accepted,
        )
    ):
        reasons.append("pydma_fit_gate_failed")
    return reasons


def _is_excluded_part2_group(dataset_part: str, group_id: str) -> bool:
    part_number = _part_number(dataset_part)
    group_number = str(group_id).strip().casefold().removeprefix("group").strip()
    return part_number == "2" and group_number in {"5", "6"}


def _part_number(dataset_part: str) -> str:
    return str(dataset_part).strip().casefold().replace("part", "").strip()


def _target_matrix(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(TARGET_NAMES):
        raise ValueError(
            f"{name} must have shape (rows, {len(TARGET_NAMES)}), got {array.shape}."
        )
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    return array


def _target_mask(values: np.ndarray, rows: int) -> np.ndarray:
    mask = np.asarray(values)
    if mask.shape != (rows, len(TARGET_NAMES)) or mask.dtype != np.bool_:
        raise ValueError(
            f"Target mask must be boolean with shape (rows, {len(TARGET_NAMES)})."
        )
    return mask


def _require_finite_observed(values: np.ndarray, mask: np.ndarray) -> None:
    if not np.isfinite(values[mask]).all():
        raise ValueError("Observed target values must be finite.")


def _is_finite_number(value: object) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
