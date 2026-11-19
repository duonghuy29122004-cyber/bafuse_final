from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pandas as pd


_MATLAB_V5 = b"MATLAB 5.0 MAT-file"
_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
_MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none"}
_DATA_FILE_PATTERN = re.compile(
    r"TPG(?P<group>\d+)(?:[._](?P<file>\d+))?\s*[-_ ]*Cell\s*(?P<cell>\d+)",
    re.IGNORECASE,
)
_EIS_PATTERN = re.compile(
    r"Group(?P<group>\d+)_(?P<stage>BoL|MoL)_Cell(?P<cell>\d+)_SOC(?P<soc>\d+)",
    re.IGNORECASE,
)
_PROCEDURE_PATTERN = re.compile(r"TPG(?P<group>\d+)(?:\.(?P<file>\d+))?$", re.IGNORECASE)
_COLUMN_UNITS: dict[str, tuple[str, str]] = {
    "volts": ("V", "Inferred from column name and protocol voltage limits."),
    "voltage": ("V", "Inferred from column name and protocol voltage limits."),
    "amps": ("A", "Inferred from column name and protocol current values."),
    "current": ("A", "Inferred from column name and protocol current values."),
    "amphr": ("Unknown", "Column suggests ampere-hours; absolute unit is not explicit in the local guide."),
    "capacity": ("Unknown", "Unit is not established by the local guide."),
    "watthr": ("Unknown", "Unit is not established by the local guide."),
    "temp1": ("degC (inferred)", "Temperature scale inferred from the documented 24 degC chamber setting."),
    "rez_ohm": ("Ohm", "Explicit in the column name."),
    "re_z_ohm": ("Ohm", "Explicit in the column name."),
    "cyc": ("count/index", "Cycle counter semantics beyond the file are not documented."),
    "step": ("count/index", "Mapped to protocol actions only for documented RPT files."),
    "soc": ("percent or fraction (file-dependent)", "Range must be checked per file."),
}


@dataclass
class GuideEntry:
    group_id: str
    cell_id: str
    procedure: str
    stage: str
    file_id: int


@dataclass
class FileRecord:
    relative_path: str
    file_type: str
    size_bytes: int
    modified_ns: int
    format: str
    decode_status: str
    group_id: str = ""
    cell_id: str = ""
    physical_cell_id: str = ""
    file_id: int | None = None
    procedure: str = ""
    stage: str = ""
    modality: str = ""
    soc: str = ""
    condition: str = ""
    row_count: int | None = None
    columns_json: str = "[]"
    column_types_json: str = "{}"
    units_json: str = "{}"
    missing_counts_json: str = "{}"
    measurement_summary_json: str = "{}"
    schema_status: str = ""
    warnings: str = ""
    error: str = ""


@dataclass
class ColumnStats:
    non_missing: int = 0
    missing: int = 0
    integer_values: int = 0
    float_values: int = 0
    text_values: int = 0
    minimum: float | None = None
    maximum: float | None = None

    def observe(self, series: pd.Series) -> None:
        self.missing += int(series.isna().sum())
        values = series.dropna()
        self.non_missing += int(len(values))
        if values.empty:
            return
        if pd.api.types.is_bool_dtype(values.dtype):
            self.integer_values += int(len(values))
        elif pd.api.types.is_integer_dtype(values.dtype):
            self.integer_values += int(len(values))
            low, high = int(values.min()), int(values.max())
            self.minimum = low if self.minimum is None else min(self.minimum, low)
            self.maximum = high if self.maximum is None else max(self.maximum, high)
        elif pd.api.types.is_float_dtype(values.dtype):
            self.float_values += int(len(values))
            low, high = float(values.min()), float(values.max())
            if math.isfinite(low) and math.isfinite(high):
                self.minimum = low if self.minimum is None else min(self.minimum, low)
                self.maximum = high if self.maximum is None else max(self.maximum, high)
        else:
            converted = pd.to_numeric(values, errors="coerce")
            numeric_count = int(converted.notna().sum())
            self.text_values += int(len(values) - numeric_count)
            self.float_values += numeric_count
            numeric = converted.dropna()
            if not numeric.empty:
                low, high = float(numeric.min()), float(numeric.max())
                self.minimum = low if self.minimum is None else min(self.minimum, low)
                self.maximum = high if self.maximum is None else max(self.maximum, high)

    @property
    def dtype(self) -> str:
        kinds = sum(bool(value) for value in (self.integer_values, self.float_values, self.text_values))
        if kinds == 0:
            return "empty"
        if self.text_values and (self.integer_values or self.float_values):
            return "mixed"
        if self.text_values:
            return "string"
        if self.float_values:
            return "float"
        return "integer"


def _shared_strings(archive: zipfile.ZipFile, namespace: dict[str, str]) -> list[str]:
    member = "xl/sharedStrings.xml"
    if member not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read(member))
    return [
        "".join(text.text or "" for text in item.findall(".//m:t", namespace))
        for item in root.findall("m:si", namespace)
    ]


def read_guide_entries(guide_path: Path) -> list[GuideEntry]:
    if not guide_path.exists():
        return []
    ns = {
        "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }
    package_ns = {"p": "http://schemas.openxmlformats.org/package/2006/relationships"}
    entries: list[GuideEntry] = []
    with zipfile.ZipFile(guide_path) as archive:
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {
            item.attrib["Id"]: item.attrib["Target"]
            for item in relationships.findall("p:Relationship", package_ns)
        }
        strings = _shared_strings(archive, ns)
        for sheet in workbook.findall("m:sheets/m:sheet", ns):
            target = targets[sheet.attrib[f"{{{ns['r']}}}id"]].lstrip("/")
            if not target.startswith("xl/"):
                target = f"xl/{target}"
            root = ElementTree.fromstring(archive.read(target))
            worksheet_rows: list[dict[str, str]] = []
            for row in root.findall(".//m:sheetData/m:row", ns):
                values: dict[str, str] = {}
                for cell in row.findall("m:c", ns):
                    match = re.match(r"[A-Z]+", cell.attrib.get("r", ""))
                    if not match:
                        continue
                    value = cell.find("m:v", ns)
                    text = "" if value is None else value.text or ""
                    if cell.attrib.get("t") == "s" and text:
                        text = strings[int(text)]
                    values[match.group()] = text.strip()
                worksheet_rows.append(values)

            has_guide_header = any(
                re.sub(r"[^a-z0-9]", "", values.get("A", "").casefold()) == "cell"
                and re.sub(r"[^a-z0-9]", "", values.get("B", "").casefold())
                == "procedure"
                and re.sub(r"[^a-z0-9]", "", values.get("C", "").casefold())
                == "rptprofile"
                for values in worksheet_rows
            )
            if (
                "data file names" not in sheet.attrib.get("name", "").casefold()
                and not has_guide_header
            ):
                continue

            current_cell = ""
            for values in worksheet_rows:
                procedure = values.get("B", "")
                stage = values.get("C", "")
                first_column = values.get("A", "")
                if (
                    first_column.casefold().startswith("group ")
                    and not procedure
                    and not stage
                ):
                    current_cell = ""
                    continue
                if (
                    first_column.casefold() == "cell"
                    and procedure.casefold() == "procedure"
                ):
                    continue
                if first_column.isdigit() and procedure.startswith("TPG"):
                    current_cell = first_column
                if not procedure or not stage:
                    continue
                match = _PROCEDURE_PATTERN.fullmatch(procedure)
                if match is None or not current_cell:
                    continue
                file_id = int(match.group("file") or 1)
                entries.append(
                    GuideEntry(
                        group_id=match.group("group"),
                        cell_id=current_cell,
                        procedure=procedure,
                        stage=stage,
                        file_id=file_id,
                    )
                )
    return entries


def _identify_file(path: Path, dataset_root: Path) -> dict[str, Any]:
    relative = path.relative_to(dataset_root)
    lowered = str(relative).lower()
    metadata: dict[str, Any] = {
        "group_id": "",
        "cell_id": "",
        "physical_cell_id": "",
        "file_id": None,
        "procedure": "",
        "stage": "",
        "modality": "unknown",
        "soc": "",
        "condition": "",
    }
    if "half_cell" in lowered:
        metadata["modality"] = "half_cell_reference"
        metadata["condition"] = "pristine harvested electrode reference (documented)"
        return metadata
    eis_match = _EIS_PATTERN.search(path.stem)
    if eis_match:
        metadata.update(
            group_id=eis_match.group("group"),
            cell_id=eis_match.group("cell"),
            physical_cell_id=f"G{eis_match.group('group')}-Cell{eis_match.group('cell')}",
            stage=eis_match.group("stage"),
            modality="EIS",
            soc=eis_match.group("soc"),
            condition=f"{eis_match.group('stage')} EIS at SOC {eis_match.group('soc')}%",
        )
        return metadata
    match = _DATA_FILE_PATTERN.search(path.stem)
    if match:
        group_id = match.group("group")
        cell_id = match.group("cell")
        file_id = int(match.group("file") or 1)
        metadata.update(
            group_id=group_id,
            cell_id=cell_id,
            physical_cell_id=f"G{group_id}-Cell{cell_id}",
            file_id=file_id,
            procedure=f"TPG{group_id}" if file_id == 1 else f"TPG{group_id}.{file_id}",
            modality="cycling_and_RPT",
        )
        return metadata
    return metadata


def _read_mat_header(path: Path) -> tuple[str, str, str]:
    with path.open("rb") as stream:
        header = stream.read(520)
    if header.startswith(_MATLAB_V5):
        return "MATLAB v5/v7.2 binary", "header_only; variables not decoded", "MAT v5 header; MCOS structures may be opaque to Python readers."
    if header.startswith(b"MATLAB 7.3 MAT-file") or header[512:520] == _HDF5_SIGNATURE:
        return "MATLAB v7.3/HDF5", "header_only; HDF5 hierarchy not decoded", "MATLAB v7.3 container detected; table/object decoding not attempted."
    if header.startswith(_HDF5_SIGNATURE):
        return "HDF5 container (MAT compatibility unknown)", "header_only; HDF5 hierarchy not decoded", "HDF5 signature detected."
    return "unknown MAT-like binary", "header_only; variables not decoded", "MAT variables and internal structures are not established."


def _column_unit(column: str) -> tuple[str, str]:
    return _COLUMN_UNITS.get(column.strip().lower(), ("Unknown", "No verified unit found in local documentation."))


def _finish_column_stats(stats: dict[str, ColumnStats]) -> tuple[dict[str, str], dict[str, int], dict[str, dict[str, Any]]]:
    types = {name: item.dtype for name, item in stats.items()}
    missing = {name: item.missing for name, item in stats.items()}
    quality = {
        name: {
            "dtype": item.dtype,
            "non_missing": item.non_missing,
            "missing": item.missing,
            "minimum": item.minimum,
            "maximum": item.maximum,
            "unit": _column_unit(name)[0],
            "unit_evidence": _column_unit(name)[1],
        }
        for name, item in stats.items()
    }
    return types, missing, quality


def scan_csv(path: Path, dataset_root: Path, guide_lookup: dict[tuple[str, str, int], GuideEntry]) -> tuple[FileRecord, list[dict[str, Any]], dict[str, Any]]:
    import numpy as np

    metadata = _identify_file(path, dataset_root)
    if metadata["modality"] == "cycling_and_RPT" and metadata["file_id"] is not None:
        entry = guide_lookup.get((metadata["group_id"], metadata["cell_id"], metadata["file_id"]))
        if entry:
            metadata["stage"] = entry.stage
            metadata["procedure"] = entry.procedure
            metadata["condition"] = entry.stage
    stats: dict[str, ColumnStats] = {}
    row_count = 0
    step4_rows = 0
    step4_capacity_values: list[float] = []
    step4_current_signs: set[str] = set()
    step_counts: defaultdict[str, int] = defaultdict(int)
    warnings: list[str] = []
    schema_columns: list[str] = []
    try:
        for chunk in pd.read_csv(path, chunksize=100_000, low_memory=False):
            if not schema_columns:
                schema_columns = [str(column) for column in chunk.columns]
                stats = {column: ColumnStats() for column in schema_columns}
            row_count += len(chunk)
            for column in schema_columns:
                stats[column].observe(chunk[column])
            if metadata["modality"] == "cycling_and_RPT" and {"Step", "Amphr"}.issubset(chunk.columns):
                step = pd.to_numeric(chunk["Step"], errors="coerce")
                if "Step" in chunk:
                    for value, count in step.value_counts(dropna=True).items():
                        step_counts[str(int(value) if float(value).is_integer() else value)] += int(count)
                discharge = chunk.loc[step.eq(4)]
                if not discharge.empty:
                    step4_rows += len(discharge)
                    capacities = pd.to_numeric(discharge["Amphr"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                    if not capacities.empty:
                        step4_capacity_values.extend((float(capacities.min()), float(capacities.max())))
                    if "Amps" in discharge:
                        currents = pd.to_numeric(discharge["Amps"], errors="coerce").dropna()
                        step4_current_signs.update("negative" if value < 0 else "positive" if value > 0 else "zero" for value in currents)
    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as error:
        record = _record_for_path(path, dataset_root, metadata)
        record.decode_status = "csv_scan_failed"
        record.error = f"{type(error).__name__}: {error}"
        return record, [], {"error": record.error}
    types, missing, quality = _finish_column_stats(stats)
    warnings.extend(
        f"{column}: {count} missing value(s)"
        for column, count in missing.items()
        if count
    )
    measurement: dict[str, Any] = {"step_counts": dict(step_counts)}
    if step4_rows:
        measurement.update(
            step4_row_count=step4_rows,
            step4_capacity_raw_span=(
                max(step4_capacity_values) - min(step4_capacity_values)
                if step4_capacity_values
                else None
            ),
            step4_current_signs=sorted(step4_current_signs),
            capacity_unit="Unknown; Amphr column unit is not explicitly specified in local Guide.",
        )
    record = _record_for_path(path, dataset_root, metadata)
    record.decode_status = "csv_scanned"
    record.row_count = row_count
    record.columns_json = json.dumps(schema_columns, ensure_ascii=False)
    record.column_types_json = json.dumps(types, ensure_ascii=False, sort_keys=True)
    record.units_json = json.dumps(
        {column: {"unit": _column_unit(column)[0], "evidence": _column_unit(column)[1]} for column in schema_columns},
        ensure_ascii=False,
        sort_keys=True,
    )
    record.missing_counts_json = json.dumps(missing, ensure_ascii=False, sort_keys=True)
    record.measurement_summary_json = json.dumps(measurement, ensure_ascii=False, sort_keys=True)
    record.schema_status = "scanned"
    record.warnings = "; ".join(warnings)
    quality_rows = [
        {
            "relative_path": record.relative_path,
            "group_id": record.group_id,
            "cell_id": record.cell_id,
            "physical_cell_id": record.physical_cell_id,
            "modality": record.modality,
            "column": name,
            **details,
        }
        for name, details in quality.items()
    ]
    return record, quality_rows, measurement


def _record_for_path(path: Path, dataset_root: Path, metadata: dict[str, Any]) -> FileRecord:
    if metadata["modality"] == "unknown":
        relative = str(path.relative_to(dataset_root)).lower()
        if "eis" in relative:
            metadata["modality"] = "EIS_or_impedance_data"
        elif any(f"group_{number}" in relative for number in range(1, 5)):
            metadata["modality"] = "cycling_or_RPT"
        else:
            metadata["modality"] = "unknown"
    return FileRecord(
        relative_path=path.relative_to(dataset_root).as_posix(),
        file_type=path.suffix.lower().lstrip("."),
        size_bytes=path.stat().st_size,
        modified_ns=path.stat().st_mtime_ns,
        format="",
        decode_status="",
        **metadata,
    )


def _join_mat_guide_metadata(record: FileRecord, lookup: dict[tuple[str, str, int], GuideEntry]) -> None:
    if record.modality != "cycling_and_RPT" or record.file_id is None:
        return
    entry = lookup.get((record.group_id, record.cell_id, record.file_id))
    if entry is not None:
        record.procedure = entry.procedure
        record.stage = entry.stage
        record.condition = entry.stage


def _schema_signature(record: FileRecord) -> str:
    return record.columns_json


def audit_dataset(dataset_root: Path) -> tuple[list[FileRecord], list[dict[str, Any]], list[GuideEntry], str]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Part 1 dataset directory does not exist: {dataset_root}")
    guide = dataset_root / "Guide_to_Datafiles.xlsx"
    guide_entries = read_guide_entries(guide)
    lookup = {(entry.group_id, entry.cell_id, entry.file_id): entry for entry in guide_entries}
    files = sorted(
        (path for path in dataset_root.rglob("*") if path.is_file() and path.suffix.lower() in {".csv", ".mat"}),
        key=lambda item: item.relative_to(dataset_root).as_posix().casefold(),
    )
    records: list[FileRecord] = []
    quality_rows: list[dict[str, Any]] = []
    fingerprint = hashlib.sha256()
    total = len(files)
    for index, path in enumerate(files, 1):
        relative = path.relative_to(dataset_root).as_posix()
        stat = path.stat()
        fingerprint.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
        if path.suffix.lower() == ".mat":
            metadata = _identify_file(path, dataset_root)
            record = _record_for_path(path, dataset_root, metadata)
            _join_mat_guide_metadata(record, lookup)
            record.format, record.decode_status, note = _read_mat_header(path)
            record.schema_status = "not_decoded"
            record.warnings = note
            records.append(record)
        else:
            record, qrows, _ = scan_csv(path, dataset_root, lookup)
            records.append(record)
            quality_rows.extend(qrows)
        if index % 25 == 0 or index == total:
            print(f"Audited {index}/{total} Part 1 CSV/MAT files", flush=True)
    if guide.exists():
        guide_stat = guide.stat()
        fingerprint.update(f"Guide_to_Datafiles.xlsx\0{guide_stat.st_size}\0{guide_stat.st_mtime_ns}\n".encode())
    return records, quality_rows, guide_entries, fingerprint.hexdigest()


def validate_cached_manifest(dataset_root: Path, records: list[FileRecord], expected_hash: str) -> None:
    files = sorted(
        (
            path
            for path in dataset_root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".csv", ".mat"}
        ),
        key=lambda item: item.relative_to(dataset_root).as_posix().casefold(),
    )
    if len(files) != len(records):
        raise ValueError("Cached audit does not match the current dataset file count.")

    fingerprint = hashlib.sha256()
    for path, record in zip(files, records):
        relative = path.relative_to(dataset_root).as_posix()
        stat = path.stat()
        if (
            relative != record.relative_path
            or stat.st_size != record.size_bytes
            or stat.st_mtime_ns != record.modified_ns
        ):
            raise ValueError(f"Cached audit is stale for dataset file: {relative}")
        fingerprint.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
    guide = dataset_root / "Guide_to_Datafiles.xlsx"
    if guide.exists():
        guide_stat = guide.stat()
        fingerprint.update(
            f"Guide_to_Datafiles.xlsx\0{guide_stat.st_size}\0{guide_stat.st_mtime_ns}\n".encode()
        )
    if fingerprint.hexdigest() != expected_hash:
        raise ValueError("Cached audit fingerprint does not match the current Part 1 dataset.")


def feature_schema(records: list[FileRecord], dataset_hash: str) -> dict[str, Any]:
    columns: dict[str, dict[str, Any]] = {}
    schemas: defaultdict[str, list[str]] = defaultdict(list)
    for record in records:
        if record.file_type != "csv" or not record.columns_json:
            continue
        schemas[_schema_signature(record)].append(record.relative_path)
        names = json.loads(record.columns_json)
        types = json.loads(record.column_types_json or "{}")
        units = json.loads(record.units_json or "{}")
        for name in names:
            details = units.get(name, {})
            entry = columns.setdefault(
                name,
                {
                    "observed_types": set(),
                    "units_observed": set(),
                    "unit_evidence": set(),
                    "file_count": 0,
                    "modalities": set(),
                },
            )
            entry["observed_types"].add(types.get(name, "unknown"))
            entry["units_observed"].add(details.get("unit", "Unknown"))
            entry["unit_evidence"].add(details.get("evidence", "No verified unit found in local documentation."))
            entry["file_count"] += 1
            entry["modalities"].add(record.modality)
    normalized_columns = {
        name: {
            **{key: sorted(value) if isinstance(value, set) else value for key, value in details.items()},
            "meaning": _column_meaning(name),
        }
        for name, details in sorted(columns.items())
    }
    return {
        "dataset_scope": "Oxford Path Dependent Battery Degradation Dataset Part 1 only",
        "dataset_manifest_hash": dataset_hash,
        "manifest_hash_method": "SHA-256 of sorted relative paths, file sizes, modification timestamps, and Guide timestamp; not a content hash.",
        "columns": normalized_columns,
        "csv_schema_signatures": [
            {"columns_json": key, "file_count": len(paths), "example_files": paths[:5]}
            for key, paths in sorted(schemas.items())
        ],
    }


def _column_meaning(name: str) -> str:
    normalized = name.strip().lower()
    known = {
        "volts": "Measured voltage (measurement semantics inferred from protocol).",
        "voltage": "Measured voltage.",
        "amps": "Measured current (sign convention must be interpreted per protocol).",
        "current": "Measured current (sign convention must be interpreted per protocol).",
        "amphr": "Capacity-like column; absolute unit is not explicitly verified.",
        "capacity": "Capacity-like column; absolute unit is not explicitly verified.",
        "cyc": "File-local cycle/index field; not assumed to be lifetime cycle count.",
        "step": "Test-step index; mapped to discharge/charge/pseudo-OCV only for documented protocol rows.",
        "temp1": "Temperature measurement channel; exact sensor placement is undocumented.",
        "rez_ohm": "Real impedance component, per column name.",
        "n_a": "Unresolved EIS field; its meaning and unit are not documented locally.",
        "soc": "State-of-charge-like field; numeric scale varies by source file.",
    }
    return known.get(normalized, "Meaning and units are not established by local documentation.")
