from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from bafuse.hybrid.final2_dataset import PHYSICS_FEATURE_NAMES
from bafuse.hybrid.final2_models import (
    FINAL2_MODEL_NAMES,
    build_final2_model,
    forward_final2_model,
)
from bafuse.hybrid.p5_normalization import PopulationZScore
from bafuse.hybrid.final2_experiment import Step4CNN1DBaseline
from app.part3_review_preprocessing import (
    RESOLUTION_PATH as PART3_REVIEW_RESOLUTION_PATH,
    load_review_resolution_index,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "experiments" / "final2" / "training_6_models" / "run_001"
CHECKPOINT_MANIFEST = RUN_DIR / "checkpoint_manifest.json"
NORMALIZATION_PATH = RUN_DIR / "normalization_parameters.json"
TRAINING_CONFIG_PATH = ROOT / "experiments" / "final2" / "training_6_models" / "training_config_v2.json"
SELECTION_PATH = RUN_DIR / "validation_selection.json"
METRICS_PATH = RUN_DIR / "selected_validation_metrics.csv"
PART3_ELIGIBILITY_PATH = ROOT / "app" / "part3_modality_eligibility.csv"
H2_CHARGE_PATH = ROOT / "experiments" / "pydma_hybrid_direction_v3" / "h2_hybrid_v3.csv"
PART3_RUN009_CHARGE_PATH = (
    ROOT
    / "data"
    / "processed"
    / "label_pydma_fit_part3"
    / "run_009_part3_pydma"
    / "pydma_results.csv"
)
G10_CHARGE_PATH = (
    ROOT
    / "experiments"
    / "final2"
    / "phase_c2_g10_cell2_charge"
    / "charge_degradation_results.csv"
)
G10_RPT10_SOURCE_FILE = (
    "data/raw/oxford/part3/Group_10/Group 10/TPG10.19-Cell2.csv"
)
G10_RPT10_BOL_SOURCE_FILE = (
    "data/raw/oxford/part3/Group_10/Group 10/TPG10.0-Cell2.csv"
)
EXTERNAL_METRICS_PATH = (
    ROOT
    / "experiments"
    / "final2"
    / "training_6_models"
    / "external_evaluation"
    / "external_metrics.json"
)

STEP4_BASELINE = "CNN1D-Step4-Baseline"
MODEL_NAMES = (STEP4_BASELINE, *FINAL2_MODEL_NAMES)
MODEL_ARCHITECTURES = {
    STEP4_BASELINE: (
        "Step4CNN1DBaseline: single CNNVoltageEncoder + 32→32→1 regression head"
    ),
    "CNN-3M": (
        "BaFuse_CNN1D_P5: independent Step 4/Step 10 Conv1D encoders, "
        "five-feature MLP, fusion head"
    ),
    "CNN-2M": (
        "CNN1D2MSoH: independent Step 4/Step 10 Conv1D encoders and fusion head"
    ),
    "TCN-2M": (
        "TCN2MSoH: independent causal TCN encoders (receptive field 13) "
        "and fusion head"
    ),
    "ResNet-2M": (
        "ResNet1D-2M: independent residual 1D voltage encoders and fusion head"
    ),
    "DenseNet-2M": (
        "DenseNet2MSoH: independent DenseNet1D voltage encoders and fusion head"
    ),
    "ConvNeXt-2M": (
        "ConvNeXt2MSoH: independent ConvNeXt1D voltage encoders and fusion head"
    ),
}
MODEL_MODALITIES = {
    STEP4_BASELINE: ("step4_voltage",),
    "CNN-3M": ("step4_voltage", "step10_voltage", "p5_physics"),
    **{
        name: ("step4_voltage", "step10_voltage")
        for name in FINAL2_MODEL_NAMES
        if name != "CNN-3M"
    },
}
_TRUE_VALUES = {"1", "true", "yes", "y", "available", "eligible", "accepted"}
_FALSE_VALUES = {"0", "false", "no", "n", "unavailable", "ineligible", "missing"}
_RPT_PATTERN = re.compile(r"RPT\s*([0-9]+)", re.IGNORECASE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _truth(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().casefold()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    return None


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _rpt_number(value: Any) -> str:
    match = _RPT_PATTERN.fullmatch(str(value).strip())
    if match:
        return str(int(match.group(1)))
    digits = re.search(r"\d+", str(value).strip())
    if digits:
        return str(int(digits.group(0)))
    raise ValueError(f"Invalid RPT/checkpoint value: {value}")


def _integer_id(value: Any, label: str) -> str:
    match = re.search(r"\d+", str(value).strip())
    if not match:
        raise ValueError(f"Invalid {label} value: {value}")
    return str(int(match.group(0)))


def _identity_key(part: Any, group: Any, cell: Any, rpt: Any) -> tuple[str, str, str, str]:
    return (
        _integer_id(part, "part"),
        _integer_id(group, "group"),
        str(cell).strip(),
        _rpt_number(rpt),
    )


def _norm_path(value: Any) -> str:
    return str(value).strip().replace("\\", "/").casefold()


def _part3_cell_number(value: Any) -> str:
    match = re.search(r"cell\D*(\d+)$", str(value).strip(), re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid Part 3 cell value: {value}")
    return str(int(match.group(1)))


def _row_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    part = row.get("part", row.get("dataset_part"))
    group = row.get("group", row.get("group_id"))
    cell = row.get("physical_cell_id", row.get("cell_id"))
    rpt = row.get("rpt_index", row.get("rpt", row.get("checkpoint")))
    if None in (part, group, cell, rpt):
        raise ValueError("Exact part/group/cell/RPT identity is required.")
    return _identity_key(part, group, cell, rpt)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _first_field(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in row and str(row[name]).strip() != "":
            return row[name]
    return None


def _eligibility_flag(
    row: dict[str, Any], names: tuple[str, ...]
) -> tuple[bool | None, str]:
    value = _first_field(row, names)
    if value is None:
        return None, "eligibility_column_missing"
    parsed = _truth(value)
    if parsed is None:
        return None, f"unrecognized_eligibility_value:{value}"
    return parsed, str(value)


class Final2ModelRuntime:
    """Loads only frozen Final2 checkpoints and their saved Train-only scalers."""

    def __init__(
        self,
        run_dir: Path = RUN_DIR,
        eligibility_path: Path = PART3_ELIGIBILITY_PATH,
        review_resolution_path: Path = PART3_REVIEW_RESOLUTION_PATH,
    ) -> None:
        self.run_dir = run_dir
        self.eligibility_path = eligibility_path
        self.review_resolution = load_review_resolution_index(review_resolution_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: dict[str, tuple[torch.nn.Module, dict[str, Any], str]] = {}
        self.load_errors: dict[str, str] = {}
        self.manifest: dict[str, Any] = {}
        self.training_config: dict[str, Any] = {}
        self.validation_selection: dict[str, Any] = {}
        self.normalization_sha256 = ""
        self.training_config_sha256 = ""
        self.scalers: dict[str, PopulationZScore] = {}
        self.target_scaler: PopulationZScore | None = None
        self.validation_metrics: dict[str, dict[str, Any]] = {}
        self.external_metrics: dict[str, dict[str, Any]] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        try:
            self.manifest = json.loads(
                (self.run_dir / CHECKPOINT_MANIFEST.name).read_text(encoding="utf-8")
            )
            config_path = self.run_dir.parent / TRAINING_CONFIG_PATH.name
            self.training_config = json.loads(config_path.read_text(encoding="utf-8"))
            self.training_config_sha256 = _sha256(config_path)
            if (
                self.training_config.get("dataset_identifier")
                != "bafuse-final2-p5-g10-guide-verified-v2"
            ):
                raise ValueError("Training config does not match frozen Final2 v2.")
            scaler_data = json.loads(
                (self.run_dir / NORMALIZATION_PATH.name).read_text(encoding="utf-8")
            )
            self.normalization_sha256 = _sha256(
                self.run_dir / NORMALIZATION_PATH.name
            )
            if scaler_data.get("dataset_identifier") != "bafuse-final2-p5-g10-guide-verified-v2":
                raise ValueError("Scaler dataset identifier does not match frozen Final2 v2.")
            if scaler_data.get("fit_partition") != "train":
                raise ValueError("Saved normalization parameters were not fitted on Train.")
            for name in ("discharge", "charge", "physics", "target"):
                self.scalers[name] = PopulationZScore.from_dict(scaler_data[name])
            self.target_scaler = self.scalers["target"]
            metrics_path = self.run_dir / METRICS_PATH.name
            for record in _read_csv(metrics_path):
                if record["model"] not in MODEL_NAMES:
                    continue
                self.validation_metrics[record["model"]] = {
                    "validation_mae_pp": float(record["validation_MAE"]),
                    "validation_rmse_pp": float(record["validation_RMSE"]),
                    "validation_r2": float(record["validation_R2"]),
                    "seed": int(record["seed"]),
                    "best_epoch": int(record["best_epoch"]),
                }
            self.validation_selection = json.loads(
                (self.run_dir / SELECTION_PATH.name).read_text(encoding="utf-8")
            )
            try:
                external = json.loads(
                    EXTERNAL_METRICS_PATH.read_text(encoding="utf-8")
                )
                if external.get("dataset_identifier") == self.training_config.get(
                    "dataset_identifier"
                ):
                    metrics_by_model = external.get("metrics_by_model", {})
                    for name in MODEL_NAMES:
                        metric = metrics_by_model.get(name)
                        if not isinstance(metric, dict):
                            continue
                        values = {
                            "external_mae_pp": _number(metric.get("MAE")),
                            "external_rmse_pp": _number(metric.get("RMSE")),
                            "external_r2": _number(metric.get("R2")),
                            "external_sample_count": _number(metric.get("N")),
                            "external_cell_count": _number(metric.get("N_cells")),
                            "source": "saved_external_metrics_json",
                        }
                        if all(
                            values[key] is not None
                            for key in ("external_mae_pp", "external_rmse_pp", "external_r2")
                        ):
                            self.external_metrics[name] = values
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                self.external_metrics = {}
            if self.validation_selection.get("dataset_identifier") != self.training_config.get(
                "dataset_identifier"
            ):
                raise ValueError("Validation selection dataset identifier is inconsistent.")
            if (
                self.validation_selection.get("training_config_sha256")
                != self.training_config_sha256
            ):
                raise ValueError("Validation selection references a different training config.")
            if (
                self.validation_selection.get("normalization_parameters_sha256")
                != self.normalization_sha256
            ):
                raise ValueError("Validation selection references different saved scalers.")
            selected_seeds = self.validation_selection.get(
                "selected_best_seed_by_model", {}
            )
            for name in MODEL_NAMES:
                selected = selected_seeds.get(name)
                metric = self.validation_metrics.get(name)
                if (
                    not selected
                    or metric is None
                    or int(selected["seed"]) != metric["seed"]
                    or int(selected["best_epoch"]) != metric["best_epoch"]
                    or selected["checkpoint_sha256"]
                    != self.manifest.get(name, {}).get("best_sha256")
                ):
                    raise ValueError(
                        f"Validation metadata does not match the selected {name} checkpoint."
                    )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.load_errors = {name: f"model_metadata_unavailable:{error}" for name in MODEL_NAMES}

    def default_model(self) -> str:
        try:
            selection = self.validation_selection
            selected = str(selection.get("selected_primary_model", ""))
            selected_seed = int(selection.get("selected_primary_seed"))
            selected_rmse = float(selection.get("selected_primary_validation_rmse_pp"))
            metric = self.validation_metrics.get(selected, {})
            entry = self.manifest.get(selected, {})
            checkpoint_path = self.run_dir / str(entry.get("best_path", ""))
            supported = (
                bool(selection.get("selection_frozen"))
                and selected == "TCN-2M"
                and "TCN-2M" in self.training_config.get(
                    "primary_comparison_models", ()
                )
                and selected_seed == int(entry.get("seed", -1))
                and math.isclose(
                    selected_rmse,
                    float(metric.get("validation_rmse_pp", math.nan)),
                    rel_tol=0,
                    abs_tol=1e-12,
                )
                and checkpoint_path.is_file()
                and _sha256(checkpoint_path) == entry.get("best_sha256")
            )
            if supported and self._checkpoint_status(selected)["available"]:
                return selected
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
        if self._checkpoint_status(STEP4_BASELINE)["available"]:
            return STEP4_BASELINE
        return next(
            (
                name
                for name in MODEL_NAMES
                if self._checkpoint_status(name)["available"]
            ),
            STEP4_BASELINE,
        )

    def model_catalog(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "architecture": MODEL_ARCHITECTURES[name],
                "modalities": list(MODEL_MODALITIES[name]),
                "checkpoint_id": self.manifest.get(name, {}).get("best_path"),
                "checkpoint_path": (
                    str(
                        (self.run_dir / str(self.manifest[name]["best_path"]))
                        .resolve()
                    )
                    if self.manifest.get(name, {}).get("best_path")
                    else None
                ),
                "dataset_identifier": self.training_config.get(
                    "dataset_identifier"
                ),
                "validation_metrics": self.validation_metrics.get(name),
                "external_metrics": self.external_metrics.get(name),
                "checkpoint_status": self._checkpoint_status(name),
                "default": name == self.default_model(),
            }
            for name in MODEL_NAMES
        ]

    def _checkpoint_status(self, name: str) -> dict[str, Any]:
        if name in self.load_errors:
            return {"available": False, "reason": self.load_errors[name]}
        entry = self.manifest.get(name)
        if not entry:
            return {"available": False, "reason": "checkpoint_manifest_entry_missing"}
        path = self.run_dir / str(entry.get("best_path", ""))
        if not path.is_file():
            return {"available": False, "reason": f"checkpoint_missing:{path}"}
        try:
            digest = _sha256(path)
        except OSError as error:
            return {"available": False, "reason": f"checkpoint_unreadable:{error}"}
        if digest != entry.get("best_sha256"):
            return {"available": False, "reason": "checkpoint_sha256_mismatch"}
        if name not in self.validation_metrics:
            return {"available": False, "reason": "selected_validation_metric_missing"}
        return {"available": True, "reason": "frozen_validation_selected_checkpoint"}

    def _load_model(
        self, name: str
    ) -> tuple[torch.nn.Module, dict[str, Any], str] | None:
        if name in self.models:
            return self.models[name]
        status = self._checkpoint_status(name)
        if not status["available"]:
            self.load_errors[name] = status["reason"]
            return None
        entry = self.manifest[name]
        path = self.run_dir / entry["best_path"]
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if (
                payload.get("model") != name
                or int(payload.get("seed", -1)) != int(entry["seed"])
                or int(payload.get("best_epoch", -1)) != int(entry["best_epoch"])
                or payload.get("dataset_identifier")
                != self.training_config.get("dataset_identifier")
                or payload.get("training_config_sha256")
                != self.training_config_sha256
            ):
                raise ValueError("Checkpoint metadata does not match its frozen manifest.")
            model = (
                Step4CNN1DBaseline()
                if name == STEP4_BASELINE
                else build_final2_model(name)
            )
            model.load_state_dict(payload["model_state_dict"], strict=True)
            model.to(self.device)
            model.eval()
            loaded = (model, entry, str(path))
            self.models[name] = loaded
            return loaded
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
            self.load_errors[name] = f"checkpoint_load_failed:{error}"
            return None

    def _part3_eligibility(self, row: dict[str, Any]) -> dict[str, Any] | None:
        key = _row_key(row)
        if key[0] != "3":
            return None
        if (
            str(row.get("inventory_scope", "")) == "VERIFIED_DEMO_ONLY"
            and _truth(row.get("source_checksum_verified")) is True
        ):
            flags = {
                "step4_voltage": row.get("demo_step4_valid") is True,
                "step10_voltage": row.get("demo_step10_valid") is True,
                "p5_physics": row.get("demo_p5_valid") is True,
            }
            reasons = {
                name: "verified_demo_segment_unavailable"
                for name, available in flags.items()
                if not available
            }
            return {
                "blocked": False,
                "reason": "verified_demo_only",
                "modalities": flags,
                "modality_reasons": reasons,
            }
        resolution = self.review_resolution.get(key)
        if resolution is not None:
            expected_source = _norm_path(
                (ROOT / Path(resolution["raw_source_path"])).resolve()
            )
            actual_source_path = Path(
                str(row.get("source_file", "")).replace("/", "\\")
            )
            if not actual_source_path.is_absolute():
                actual_source_path = ROOT / actual_source_path
            actual_source = _norm_path(actual_source_path.resolve())
            if (
                actual_source != expected_source
                or str(row.get("source_sha256", "")).casefold()
                != resolution["source_sha256"].casefold()
                or _truth(row.get("source_checksum_verified")) is not True
                or _truth(resolution.get("guide_mapping_verified")) is not True
            ):
                return {
                    "blocked": True,
                    "reason": "review_resolution_source_or_mapping_not_verified",
                }
            flags = {
                "step4_voltage": _truth(resolution.get("baseline_eligible")),
                "step10_voltage": _truth(resolution.get("step10_valid")),
                "p5_physics": _truth(resolution.get("physics5_valid")),
            }
            reasons = {}
            if flags["step4_voltage"] is not True:
                reasons["step4_voltage"] = resolution.get(
                    "step4_review_reason", "step4_not_validated"
                )
            if flags["step10_voltage"] is not True:
                reasons["step10_voltage"] = resolution.get(
                    "step10_review_reason", "step10_not_validated"
                )
            if flags["p5_physics"] is not True:
                reasons["p5_physics"] = resolution.get(
                    "remaining_issue", "p5_physics_not_validated"
                )
            return {
                "blocked": False,
                "reason": "versioned_review_resolution",
                "modalities": flags,
                "modality_reasons": reasons,
            }
        if not self.eligibility_path.is_file():
            return {"blocked": True, "reason": f"eligibility_manifest_missing:{self.eligibility_path}"}
        try:
            records = _read_csv(self.eligibility_path)
        except (OSError, csv.Error, UnicodeDecodeError) as error:
            return {"blocked": True, "reason": f"eligibility_manifest_unreadable:{error}"}
        exact = []
        for record in records:
            try:
                if _row_key(record) == key:
                    exact.append(record)
            except (TypeError, ValueError):
                continue
        if not exact:
            return {"blocked": True, "reason": "no_exact_part3_modality_eligibility_row"}
        if len(exact) != 1:
            return {"blocked": True, "reason": "duplicate_exact_part3_modality_eligibility_rows"}
        record = exact[0]
        flag_specs = {
            "step4_voltage": (
                "step4_curve_available",
                "step4_available",
                "step4_valid",
                "step4_eligible",
                "step4_curve_eligible",
                "step4_modality_eligible",
                "step4_inference_eligible",
                "step4_status",
                "discharge_voltage_available",
            ),
            "step10_voltage": (
                "step10_curve_available",
                "step10_available",
                "step10_valid",
                "step10_eligible",
                "step10_curve_eligible",
                "step10_modality_eligible",
                "step10_inference_eligible",
                "step10_status",
                "charge_voltage_available",
            ),
            "p5_physics": (
                "p5_physics_available",
                "physics_available",
                "physics_inference_eligible",
                "p5_physics_eligible",
                "physics_eligible",
                "p5_eligible",
                "p5_status",
            ),
        }
        flags = {}
        modality_reasons = {}
        reasons = []
        for modality, names in flag_specs.items():
            flag, detail = _eligibility_flag(record, names)
            flags[modality] = flag
            if flag is not True:
                modality_reasons[modality] = detail
                reasons.append(f"{modality}:{detail}")
        return {
            "blocked": False,
            "reason": ";".join(reasons) if reasons else "eligible",
            "modalities": flags,
            "modality_reasons": modality_reasons,
        }

    @staticmethod
    def _row_modality_present(row: dict[str, Any], modality: str) -> bool:
        if modality == "step4_voltage":
            names = [f"step4_voltage_progress_{index:03d}" for index in range(128)]
            return all(name in row and _number(row[name]) is not None for name in names)
        if modality == "step10_voltage":
            names = [f"step10_voltage_progress_{index:03d}" for index in range(128)]
            return all(name in row and _number(row[name]) is not None for name in names)
        if modality == "p5_physics":
            values = row.get("p5_physics_features")
            if isinstance(values, (list, tuple, np.ndarray)):
                return len(values) == len(PHYSICS_FEATURE_NAMES) and all(
                    _number(value) is not None for value in values
                )
            return all(
                _number(row.get(name)) is not None for name in PHYSICS_FEATURE_NAMES
            )
        return False

    def _normalized_inputs(self, row: dict[str, Any], model_name: str) -> dict[str, torch.Tensor]:
        discharge = np.asarray(
            [[float(row[f"step4_voltage_progress_{i:03d}"]) for i in range(128)]],
            dtype=np.float32,
        )
        discharge = self.scalers["discharge"].transform(discharge)
        inputs = {
            "discharge": torch.as_tensor(discharge, dtype=torch.float32, device=self.device)
        }
        if model_name == STEP4_BASELINE:
            return inputs
        charge = np.asarray(
            [[float(row[f"step10_voltage_progress_{i:03d}"]) for i in range(128)]],
            dtype=np.float32,
        )
        charge = self.scalers["charge"].transform(charge)
        inputs["charge"] = torch.as_tensor(charge, dtype=torch.float32, device=self.device)
        if model_name == "CNN-3M":
            values = row.get("p5_physics_features")
            if not isinstance(values, (list, tuple, np.ndarray)):
                values = [row[name] for name in PHYSICS_FEATURE_NAMES]
            physics = np.asarray([values], dtype=np.float32)
            physics = self.scalers["physics"].transform(physics)
            inputs["physics"] = torch.as_tensor(physics, dtype=torch.float32, device=self.device)
        return inputs

    def preprocess_rows(
        self, rows: list[dict[str, Any]], model_name: str | None = None
    ) -> dict[str, Any]:
        selected = model_name or self.default_model()
        if selected not in MODEL_NAMES:
            raise ValueError(f"Unknown model: {selected}")
        results = []
        for row in rows:
            identity = {
                "part": str(row.get("part", row.get("dataset_part", ""))),
                "group": str(row.get("group", row.get("group_id", ""))),
                "physical_cell_id": str(
                    row.get("physical_cell_id", row.get("cell_id", ""))
                ),
                "rpt_index": str(row.get("rpt_index", row.get("rpt", ""))),
                "inventory_scope": str(row.get("inventory_scope", "")),
                "procedure": str(row.get("procedure", "")),
                "guide_stage": str(row.get("guide_stage", "")),
                "mapping_status": str(
                    row.get("mapping_status", row.get("guide_mapping_status", ""))
                ),
                "mapping_provenance": str(row.get("mapping_provenance", "")),
                "mapping_evidence": str(row.get("mapping_evidence", "")),
                "source_file": str(row.get("source_file", "")),
                "source_sha256": str(row.get("source_sha256", "")),
                "source_checksum_verified": (
                    _truth(row.get("source_checksum_verified")) is True
                ),
                "is_eol_procedure": _truth(row.get("is_eol_procedure")) is True,
            }
            missing = [
                modality
                for modality in MODEL_MODALITIES[selected]
                if not self._row_modality_present(row, modality)
            ]
            eligibility = self._part3_eligibility(row)
            gate_reason = ""
            if eligibility and eligibility["blocked"]:
                gate_reason = eligibility["reason"]
            elif eligibility:
                denied = [
                    modality
                    for modality in MODEL_MODALITIES[selected]
                    if eligibility["modalities"].get(modality) is not True
                ]
                if denied:
                    gate_reason = "part3_modality_not_eligible:" + ",".join(
                        f"{modality}({eligibility['modality_reasons'].get(modality, 'unavailable')})"
                        for modality in denied
                    )
            checkpoint = self._load_model(selected)
            if missing:
                reason = "missing_required_modalities:" + ",".join(missing)
            elif gate_reason:
                reason = gate_reason
            elif checkpoint is None:
                reason = self.load_errors.get(selected, "checkpoint_unavailable")
            else:
                reason = ""
            prediction = None
            if not reason:
                model, entry, path = checkpoint
                inputs = self._normalized_inputs(row, selected)
                with torch.inference_mode():
                    if selected == STEP4_BASELINE:
                        normalized_prediction = model(inputs["discharge"])
                    else:
                        normalized_prediction = forward_final2_model(
                            model,
                            selected,
                            inputs["discharge"],
                            inputs["charge"],
                            inputs.get("physics"),
                        )
                restored = self.target_scaler.inverse_transform(
                    normalized_prediction.detach().to("cpu").numpy().reshape(1, -1)
                )
                prediction = float(restored[0, 0])
                if not math.isfinite(prediction):
                    prediction = None
                    reason = "model_returned_nonfinite_prediction"
                else:
                    entry_metrics = self.validation_metrics[selected]
                    try:
                        displayed_path = str(Path(path).relative_to(ROOT))
                    except ValueError:
                        displayed_path = str(path)
                    results.append(
                        {
                            **identity,
                            "predicted_soh_percent": prediction,
                            "soh_status": "frozen_checkpoint_inference",
                            "model_name": selected,
                            "checkpoint_path": displayed_path,
                            "checkpoint_sha256": entry["best_sha256"],
                            "seed": int(entry["seed"]),
                            "best_epoch": int(entry["best_epoch"]),
                            "validation_metrics": entry_metrics,
                            "device": str(self.device),
                            "dataset_identifier": self.training_config[
                                "dataset_identifier"
                            ],
                            "normalization_parameters_sha256": self.normalization_sha256,
                            "preprocessing_fit_scope": "accepted Final2 Train observations only",
                            "modalities": list(MODEL_MODALITIES[selected]),
                            "eligibility_status": (
                                eligibility["reason"] if eligibility else "not_applicable"
                            ),
                            "modality_eligibility": (
                                eligibility.get("modalities", {}) if eligibility else {}
                            ),
                            "modality_eligibility_reasons": (
                                eligibility.get("modality_reasons", {}) if eligibility else {}
                            ),
                            "normalized_inputs": {
                                "step4_voltage_normalized": inputs["discharge"]
                                .detach()
                                .to("cpu")
                                .numpy()[0]
                                .astype(float)
                                .tolist(),
                                "step10_voltage_normalized": (
                                    inputs["charge"].detach().to("cpu").numpy()[0].astype(float).tolist()
                                    if "charge" in inputs
                                    else None
                                ),
                                "physics_features_normalized": (
                                    inputs["physics"].detach().to("cpu").numpy()[0].astype(float).tolist()
                                    if "physics" in inputs
                                    else None
                                ),
                            },
                        }
                    )
                    continue
            results.append(
                {
                    **identity,
                    "predicted_soh_percent": None,
                    "soh_status": "prediction_unavailable",
                    "model_name": selected,
                    "reason": reason,
                    "modalities": list(MODEL_MODALITIES[selected]),
                    "eligibility_status": (
                        eligibility["reason"] if eligibility else "not_applicable"
                    ),
                    "modality_eligibility": (
                        eligibility.get("modalities", {}) if eligibility else {}
                    ),
                    "modality_eligibility_reasons": (
                        eligibility.get("modality_reasons", {}) if eligibility else {}
                    ),
                }
            )
        return {"model_name": selected, "results": results}


def _charge_result_rows(reader: csv.DictReader, source_name: str) -> list[dict[str, str]]:
    """Project a saved artifact to the Charge inputs used by the runtime."""
    rows = list(reader)
    if any(None in row for row in rows):
        raise csv.Error("row_has_extra_values")
    if source_name == "part3_run009_charge":
        return [row for row in rows if str(row.get("direction", "")).strip().casefold() == "charge"]
    if source_name == "h2_hybrid_v3":
        identity = {"part", "group", "physical_cell_id", "rpt_index", "source_file"}
        columns = [name for name in reader.fieldnames or () if name in identity or name.startswith("charge_")]
        return [{name: row[name] for name in columns} for row in rows]
    return rows


def _load_part3_run009_charge(
    key: tuple[str, str, str, str], path: Path
) -> dict[str, Any]:
    required_fields = {
        "group_id",
        "cell_id",
        "checkup",
        "direction",
        "source_file",
        "fit_rmse_mV",
        "fit_accepted_by_pydma_15mV",
        "fit_threshold_mV",
        "accepted_runs",
        "total_runs",
        "diagnostic_label_status",
        "LLI_percent_vs_BoL",
        "LAM_anode_percent_vs_BoL",
        "LAM_cathode_percent_vs_BoL",
    }
    if not path.is_file():
        return {
            "status": "charge_artifact_missing",
            "value_percent": None,
            "reason": "expected_saved_charge_artifact_missing",
            "source_artifact": None,
            "artifact_errors": [],
            "source_errors": [],
            "missing_artifacts": [str(path)],
        }
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            missing_fields = required_fields - set(reader.fieldnames or ())
            if missing_fields:
                raise ValueError(
                    "missing_required_columns:" + ",".join(sorted(missing_fields))
                )
            records = _charge_result_rows(reader, "part3_run009_charge")
    except (OSError, csv.Error, UnicodeDecodeError, ValueError) as error:
        return {
            "status": "charge_artifact_parse_failed",
            "value_percent": None,
            "reason": f"{type(error).__name__}:{error}",
            "source_artifact": None,
            "artifact_errors": [{"artifact": str(path), "reason": str(error)}],
            "source_errors": [],
            "missing_artifacts": [],
        }

    group_key, cell_key, rpt_key = key[1], _part3_cell_number(key[2]), key[3]
    target_rows: list[dict[str, str]] = []
    bol_rows: list[dict[str, str]] = []
    for row in records:
        try:
            if _integer_id(row.get("group_id"), "group") != group_key:
                continue
            if _part3_cell_number(row.get("cell_id")) != cell_key:
                continue
            checkup = str(row.get("checkup", "")).strip()
            if checkup.casefold() == "bol":
                bol_rows.append(row)
            elif _rpt_number(checkup) == rpt_key:
                target_rows.append(row)
        except (TypeError, ValueError):
            continue

    try:
        artifact_name = path.relative_to(ROOT).as_posix()
    except ValueError:
        artifact_name = str(path)

    source_errors: list[dict[str, str]] = []
    if len(target_rows) > 1:
        source_errors.append(
            {
                "artifact": str(path),
                "reason": "part3_run009_duplicate_exact_charge_rows",
            }
        )
    elif len(bol_rows) != 1:
        source_errors.append(
            {
                "artifact": str(path),
                "reason": "part3_run009_same_cell_charge_bol_missing_or_ambiguous",
            }
        )
    if source_errors:
        return {
            "status": "charge_artifact_source_mismatch",
            "value_percent": None,
            "reason": source_errors[0]["reason"],
            "source_artifact": artifact_name,
            "artifact_errors": [],
            "source_errors": source_errors,
            "missing_artifacts": [],
        }
    if not target_rows:
        return {
            "status": "accepted_charge_pydma_unavailable_for_exact_part_group_cell_rpt",
            "value_percent": None,
            "reason": "no_existing_charge_fit_passes_rmse15_and_minimum_three_runs",
            "source_artifact": artifact_name,
            "artifact_errors": [],
            "source_errors": [],
            "missing_artifacts": [],
        }

    target, bol = target_rows[0], bol_rows[0]
    target_source = target["source_file"].strip()
    bol_source = bol["source_file"].strip()
    if not target_source or not bol_source:
        source_errors.append(
            {
                "artifact": str(path),
                "reason": "part3_run009_target_or_bol_source_file_missing",
            }
        )
    if source_errors:
        return {
            "status": "charge_artifact_source_mismatch",
            "value_percent": None,
            "reason": source_errors[0]["reason"],
            "source_artifact": artifact_name,
            "source_file": target_source,
            "same_cell_bol_source": bol_source,
            "artifact_errors": [],
            "source_errors": source_errors,
            "missing_artifacts": [],
        }

    def gate_values(row: dict[str, str]) -> tuple[bool | None, float | None, float | None, float | None]:
        return (
            _truth(row.get("fit_accepted_by_pydma_15mV")),
            _number(row.get("fit_rmse_mV")),
            _number(row.get("accepted_runs")),
            _number(row.get("fit_threshold_mV")),
        )

    target_accepted, target_rmse, target_runs, target_threshold = gate_values(target)
    bol_accepted, bol_rmse, bol_runs, bol_threshold = gate_values(bol)
    if (
        target_rmse is None
        or target_runs is None
        or bol_rmse is None
        or bol_runs is None
        or target_threshold is None
        or bol_threshold is None
    ):
        return {
            "status": "charge_artifact_parse_failed",
            "value_percent": None,
            "reason": "run009_target_or_same_cell_bol_has_missing_or_invalid_qc_fields",
            "source_artifact": artifact_name,
            "artifact_errors": [
                {
                    "artifact": str(path),
                    "reason": "run009_target_or_same_cell_bol_has_missing_or_invalid_qc_fields",
                }
            ],
            "source_errors": source_errors,
            "missing_artifacts": [],
        }
    if (
        target_accepted is not True
        or target_rmse >= 15.0
        or target_runs < 3
    ):
        return {
            "status": "accepted_charge_pydma_unavailable_for_exact_part_group_cell_rpt",
            "value_percent": None,
            "reason": "run009_target_fit_did_not_pass_saved_rmse15_minimum_three_runs_gate",
            "source_artifact": artifact_name,
            "source_file": target_source,
            "quality_control_status": target.get("diagnostic_label_status", ""),
            "artifact_errors": [],
            "source_errors": source_errors,
            "missing_artifacts": [],
        }
    if bol_accepted is not True or bol_rmse >= 15.0 or bol_runs < 3:
        source_errors.append(
            {
                "artifact": str(path),
                "reason": "part3_run009_same_cell_bol_did_not_pass_rmse15_minimum_three_runs_gate",
            }
        )
        return {
            "status": "charge_artifact_source_mismatch",
            "value_percent": None,
            "reason": source_errors[0]["reason"],
            "source_artifact": artifact_name,
            "source_file": target_source,
            "same_cell_bol_source": bol_source,
            "quality_control_status": target.get("diagnostic_label_status", ""),
            "same_cell_bol_quality_control_status": bol.get(
                "diagnostic_label_status", ""
            ),
            "artifact_errors": [],
            "source_errors": source_errors,
            "missing_artifacts": [],
        }

    values = {
        "LLI_percent": _number(target.get("LLI_percent_vs_BoL")),
        "Anode_LAM_percent": _number(target.get("LAM_anode_percent_vs_BoL")),
        "Cathode_LAM_percent": _number(target.get("LAM_cathode_percent_vs_BoL")),
    }
    if any(value is None for value in values.values()):
        return {
            "status": "charge_artifact_parse_failed",
            "value_percent": None,
            "reason": "run009_accepted_target_has_missing_or_invalid_degradation_values",
            "source_artifact": artifact_name,
            "artifact_errors": [
                {
                    "artifact": str(path),
                    "reason": "run009_accepted_target_has_missing_or_invalid_degradation_values",
                }
            ],
            "source_errors": source_errors,
            "missing_artifacts": [],
        }

    return {
        "status": "accepted_charge_pydma_diagnostic_not_ground_truth",
        "reason": "",
        "value_percent": None,
        **values,
        "fit_rmse_mV": target_rmse,
        "accepted_runs": int(target_runs),
        "total_runs": _number(target.get("total_runs")),
        "fit_threshold_mV": target_threshold,
        "fit_accepted_by_pydma_15mV": target_accepted,
        "quality_control_status": target.get("diagnostic_label_status", ""),
        "source_file": target_source,
        "charge_source_file": target_source,
        "source_artifact": artifact_name,
        "source_name": "part3_run009_charge",
        "source_direction": "charge",
        "same_cell_bol_source": bol_source,
        "same_cell_bol_fit_accepted": bol_accepted,
        "same_cell_bol_status": "same_cell_bol_fit_accepted",
        "same_cell_bol_note": (
            "Part 3 run 009 records an accepted same-cell Charge BoL fit "
            "joined by exact group and cell."
        ),
        "same_cell_bol_fit_rmse_mV": bol_rmse,
        "same_cell_bol_accepted_runs": int(bol_runs),
        "same_cell_bol_total_runs": _number(bol.get("total_runs")),
        "same_cell_bol_fit_threshold_mV": bol_threshold,
        "same_cell_bol_quality_control_status": bol.get(
            "diagnostic_label_status", ""
        ),
        "artifact_errors": [],
        "source_errors": source_errors,
        "artifact_warnings": [],
    }


def load_shared_charge_degradation(
    part: Any,
    group: Any,
    physical_cell_id: Any,
    rpt_index: Any,
    *,
    h2_path: Path = H2_CHARGE_PATH,
    part3_path: Path = PART3_RUN009_CHARGE_PATH,
    g10_path: Path = G10_CHARGE_PATH,
) -> dict[str, Any]:
    """Read accepted charge-direction PyDMA results without running a fit."""
    key = _identity_key(part, group, physical_cell_id, rpt_index)
    if key[0] == "3" and key[:3] != ("3", "10", "G10-Cell2"):
        return _load_part3_run009_charge(key, part3_path)
    candidates: list[dict[str, Any]] = []
    parse_errors: list[dict[str, str]] = []
    source_errors: list[dict[str, str]] = []
    missing_artifacts: list[str] = []
    sources = []
    if key[0] == "3" and key[:3] == ("3", "10", "G10-Cell2"):
        sources.append(
            (
                g10_path,
                "phase_c2_charge",
                {
                    "physical_cell_id",
                    "rpt_index",
                    "source_file",
                    "fit_direction",
                    "step",
                    "fit_rmse_mV",
                    "lli_percent",
                    "lam_anode_percent",
                    "lam_cathode_percent",
                    "accepted_runs",
                    "quality_control_status",
                    "same_cell_bol_source",
                    "same_cell_bol_fit_accepted",
                },
            )
        )
    else:
        sources.append(
            (
                h2_path,
                "h2_hybrid_v3",
                {
                    "part",
                    "group",
                    "physical_cell_id",
                    "rpt_index",
                    "source_file",
                    "charge_source_file",
                    "charge_accepted",
                    "charge_rmse_mV",
                    "charge_accepted_runs",
                    "charge_LLI_percent",
                    "charge_Anode_LAM_percent",
                    "charge_Cathode_LAM_percent",
                },
            )
        )
    for path, source_name, required_fields in sources:
        if not path.is_file():
            missing_artifacts.append(str(path))
            continue
        try:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream, strict=True)
                fields = set(reader.fieldnames or ())
                missing_fields = required_fields - fields
                if missing_fields:
                    parse_errors.append(
                        {
                            "artifact": str(path),
                            "reason": "missing_required_columns:"
                            + ",".join(sorted(missing_fields)),
                        }
                    )
                    continue
                records = _charge_result_rows(reader, source_name)
        except (OSError, csv.Error, UnicodeDecodeError) as error:
            parse_errors.append(
                {"artifact": str(path), "reason": f"{type(error).__name__}:{error}"}
            )
            continue
        for row_index, row in enumerate(records, start=2):
            try:
                if source_name == "h2_hybrid_v3":
                    row_key = _identity_key(
                        row["part"], row["group"], row["physical_cell_id"], row["rpt_index"]
                    )
                    rmse = _number(row.get("charge_rmse_mV"))
                    accepted_runs = _number(row.get("charge_accepted_runs"))
                    accepted = _truth(row.get("charge_accepted")) is True
                    values = {
                        "LLI_percent": _number(row.get("charge_LLI_percent")),
                        "Anode_LAM_percent": _number(row.get("charge_Anode_LAM_percent")),
                        "Cathode_LAM_percent": _number(row.get("charge_Cathode_LAM_percent")),
                    }
                    source_file = row["source_file"]
                    charge_source_file = row["charge_source_file"]
                    source_path_matches = (
                        bool(source_file.strip())
                        and bool(charge_source_file.strip())
                        and Path(source_file.replace("\\", "/")).name.casefold()
                        == Path(charge_source_file.replace("\\", "/")).name.casefold()
                    )
                    if (
                        row_key == key
                        and _truth(row.get("charge_accepted")) is True
                        and not source_path_matches
                    ):
                        source_errors.append(
                            {
                                "artifact": str(path),
                                "reason": "charge_source_file_does_not_match_observation_source",
                            }
                        )
                        continue
                    if row_key != key:
                        continue
                    same_cell_bol_source = None
                    same_cell_bol_fit_accepted = None
                    same_cell_bol_status = "not_recorded_in_h2_v3_artifact"
                    same_cell_bol_note = (
                        "H2 v3 does not record a same-cell BoL source or BoL-fit "
                        "acceptance; same-cell BoL provenance is not asserted."
                    )
                else:
                    row_key = _identity_key(
                        3, 10, row["physical_cell_id"], row["rpt_index"]
                    )
                    source_file = row["source_file"].strip()
                    if row_key == key and not source_file:
                        source_errors.append(
                            {
                                "artifact": str(path),
                                "reason": "phase_c2_target_source_file_missing",
                            }
                        )
                        continue
                    if (
                        row_key == key
                        and key[:3] == ("3", "10", "G10-Cell2")
                        and key[3] == "10"
                        and _norm_path(source_file) != _norm_path(G10_RPT10_SOURCE_FILE)
                    ):
                        source_errors.append(
                            {
                                "artifact": str(path),
                                "reason": "phase_c2_source_file_does_not_match_rpt10_source",
                            }
                        )
                        continue
                    same_cell_bol_source = row["same_cell_bol_source"].strip() or None
                    same_cell_bol_fit_accepted = _truth(
                        row.get("same_cell_bol_fit_accepted")
                    )
                    same_cell_bol_status = (
                        "same_cell_bol_fit_accepted"
                        if same_cell_bol_source and same_cell_bol_fit_accepted is True
                        else "same_cell_bol_fit_not_accepted"
                        if same_cell_bol_fit_accepted is False
                        else "same_cell_bol_provenance_incomplete"
                    )
                    same_cell_bol_note = (
                        "Phase C2 saved same-cell BoL provenance."
                        if same_cell_bol_status == "same_cell_bol_fit_accepted"
                        else "Phase C2 did not confirm an accepted same-cell BoL fit."
                    )
                    if (
                        row_key == key
                        and key[:3] == ("3", "10", "G10-Cell2")
                        and (
                            _norm_path(same_cell_bol_source or "")
                            != _norm_path(G10_RPT10_BOL_SOURCE_FILE)
                            or same_cell_bol_fit_accepted is not True
                        )
                    ):
                        source_errors.append(
                            {
                                "artifact": str(path),
                                "reason": "phase_c2_same_cell_bol_provenance_mismatch",
                            }
                        )
                        continue
                    rmse = _number(row.get("fit_rmse_mV"))
                    accepted_runs = _number(row.get("accepted_runs"))
                    quality_status = str(
                        row.get("quality_control_status", "")
                    ).casefold()
                    accepted = (
                        str(row.get("fit_direction", "")).casefold() == "charge"
                        and quality_status
                        in {
                            "diagnostic_estimate_accepted",
                            "diagnostic_only_final2_rejected",
                        }
                    )
                    values = {
                        "LLI_percent": _number(row.get("lli_percent")),
                        "Anode_LAM_percent": _number(row.get("lam_anode_percent")),
                        "Cathode_LAM_percent": _number(row.get("lam_cathode_percent")),
                    }
                quality_status = str(row.get("quality_control_status", "")).strip()
                fit_threshold = _number(row.get("fit_threshold_mV"))
                total_runs = _number(
                    row.get(
                        "charge_total_runs"
                        if source_name == "h2_hybrid_v3"
                        else "total_runs"
                    )
                )
                fit_accepted_flag = (
                    _truth(row.get("fit_accepted_by_pydma_15mV"))
                    if source_name == "part3_run009_charge"
                    else accepted
                )
                if row_key != key:
                    continue
                if accepted and (
                    rmse is None
                    or accepted_runs is None
                    or any(value is None for value in values.values())
                ):
                    parse_errors.append(
                        {
                            "artifact": str(path),
                            "reason": "accepted_charge_row_has_missing_or_invalid_numeric_fields",
                        }
                    )
                    continue
                if (
                    not accepted
                    or rmse is None
                    or rmse >= 15.0
                    or accepted_runs is None
                    or accepted_runs < 3
                    or any(value is None for value in values.values())
                ):
                    continue
                try:
                    artifact_name = path.relative_to(ROOT).as_posix()
                except ValueError:
                    artifact_name = str(path)
                candidates.append(
                    {
                        **values,
                        "fit_rmse_mV": rmse,
                        "accepted_runs": int(accepted_runs),
                        "source_file": source_file,
                        "quality_control_status": quality_status,
                        "fit_threshold_mV": fit_threshold,
                        "fit_accepted_by_pydma_15mV": fit_accepted_flag,
                        "total_runs": total_runs,
                        "charge_source_file": (
                            charge_source_file
                            if source_name == "h2_hybrid_v3"
                            else source_file
                        ),
                        "source_artifact": artifact_name,
                        "source_name": source_name,
                        "same_cell_bol_source": same_cell_bol_source,
                        "same_cell_bol_fit_accepted": same_cell_bol_fit_accepted,
                        "same_cell_bol_status": same_cell_bol_status,
                        "same_cell_bol_note": same_cell_bol_note,
                    }
                )
            except (KeyError, TypeError, ValueError) as error:
                try:
                    raw_rpt = re.search(r"\d+", str(row.get("rpt_index", "")))
                    if source_name == "h2_hybrid_v3":
                        raw_identity_matches = (
                            _integer_id(row.get("part"), "part") == key[0]
                            and _integer_id(row.get("group"), "group") == key[1]
                            and str(row.get("physical_cell_id", "")).strip().casefold()
                            == key[2].casefold()
                            and raw_rpt is not None
                            and raw_rpt.group(0) == key[3]
                        )
                    else:
                        raw_identity_matches = (
                            key[:3] == ("3", "10", "G10-Cell2")
                            and str(row.get("physical_cell_id", "")).strip().casefold()
                            == key[2].casefold()
                            and raw_rpt is not None
                            and raw_rpt.group(0) == key[3]
                        )
                except (KeyError, TypeError, ValueError):
                    raw_identity_matches = False
                if raw_identity_matches:
                    parse_errors.append(
                        {
                            "artifact": str(path),
                            "reason": (
                                f"invalid_exact_identity_row:{row_index}:"
                                f"{type(error).__name__}:{error}"
                            ),
                        }
                    )
                continue
    if not candidates:
        if source_errors:
            status = "charge_artifact_source_mismatch"
            reason = source_errors[0]["reason"]
        elif parse_errors:
            status = "charge_artifact_parse_failed"
            reason = parse_errors[0]["reason"]
        elif missing_artifacts:
            status = "charge_artifact_missing"
            reason = "expected_saved_charge_artifact_missing"
        else:
            status = "accepted_charge_pydma_unavailable_for_exact_part_group_cell_rpt"
            reason = "no_existing_charge_fit_passes_rmse15_and_minimum_three_runs"
        return {
            "status": status,
            "value_percent": None,
            "reason": reason,
            "source_artifact": None,
            "artifact_errors": parse_errors,
            "source_errors": source_errors,
            "missing_artifacts": missing_artifacts,
        }
    preferred = [candidate for candidate in candidates if candidate["source_name"] == "phase_c2_charge"]
    chosen = preferred[0] if preferred else candidates[0]
    duplicates = [
        candidate
        for candidate in candidates
        if any(candidate[name] != chosen[name] for name in ("LLI_percent", "Anode_LAM_percent", "Cathode_LAM_percent"))
    ]
    if duplicates:
        return {
            "status": "accepted_charge_pydma_ambiguous_conflicting_artifacts",
            "value_percent": None,
            "reason": "conflicting_accepted_charge_results_for_exact_identity",
            "source_artifact": [candidate["source_artifact"] for candidate in candidates],
            "artifact_errors": parse_errors,
            "source_errors": source_errors,
        }
    return {
        "status": (
            "accepted_charge_fit_final2_observation_rejected_diagnostic_only"
            if chosen["source_name"] == "phase_c2_charge"
            and key[0:3] == ("3", "10", "G10-Cell2")
            and key[3] == "10"
            else "accepted_charge_pydma_diagnostic_not_ground_truth"
        ),
        "reason": "",
        "value_percent": None,
        **{key: chosen[key] for key in ("LLI_percent", "Anode_LAM_percent", "Cathode_LAM_percent")},
        "fit_rmse_mV": chosen["fit_rmse_mV"],
        "accepted_runs": chosen["accepted_runs"],
        "total_runs": chosen["total_runs"],
        "fit_threshold_mV": chosen["fit_threshold_mV"],
        "fit_accepted_by_pydma_15mV": chosen["fit_accepted_by_pydma_15mV"],
        "quality_control_status": chosen["quality_control_status"],
        "source_file": chosen["source_file"],
        "charge_source_file": chosen["charge_source_file"],
        "source_artifact": chosen["source_artifact"],
        "source_name": chosen["source_name"],
        "source_direction": "charge",
        "same_cell_bol_source": chosen["same_cell_bol_source"],
        "same_cell_bol_fit_accepted": chosen["same_cell_bol_fit_accepted"],
        "same_cell_bol_status": chosen["same_cell_bol_status"],
        "same_cell_bol_note": chosen["same_cell_bol_note"],
        "artifact_errors": parse_errors,
        "source_errors": source_errors,
        "artifact_warnings": [
            *(
                [{"reason": "expected_saved_charge_artifact_missing", "artifact": p}
                 for p in missing_artifacts]
            ),
            *parse_errors,
            *source_errors,
        ],
    }
