"""Audit the updated CNN-3M Physics contract without changing frozen data."""

import argparse
import csv
import hashlib
import json
import math
import re
import tempfile
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

import numpy as np

FEATURES = (
    "nominal_cycle_age",
    "voltage_droop_50_soc",
    "discharge_voltage_slope_20_80",
    "discharge_voltage_derivative_mad_10_90",
    "discharge_voltage_segment_contrast_10_30_vs_70_90",
)
UNITS = ("nominal cycles", "V", "V per unit progress", "V per unit progress", "V")
SPLITS = {"train": 99, "validation": 40, "external_test": 74}


def nominal_age(row, mapped):
    """Use the verified Guide ordinal and nominal test cadence, without counting raw Cyc."""
    part = int(row["dataset_part"])
    group = int(row["group"])
    rpt = int(row["rpt_index"])
    if rpt <= 0 or not (
        (part in (1, 2) and 1 <= group <= 4) or (part == 3 and 7 <= group <= 10)
    ):
        raise ValueError("Unsupported nominal 48-cycle schedule")
    stage = re.fullmatch(r"RPT\s*(\d+)", row["guide_stage"].strip(), re.IGNORECASE)
    if stage is None or int(stage[1]) != rpt:
        raise ValueError("Guide stage does not independently identify the RPT ordinal")
    if mapped is not None:
        age = 48 * rpt
        if (mapped["procedure"] != row["source_tpg"]
                or mapped["cycle_age_status"] != "nominal_48_cycles_per_RPT"
                or mapped["chronology_basis"] != "guide_rpt_ordinal"
                or float(mapped["nominal_cycle_age"]) != age):
            raise ValueError("Unified age provenance does not match the frozen RPT")
        return age, "verified Guide ordinal and Oxford 48-cycle schedule"
    if (part == 3 and group == 10 and row["physical_cell_id"] == "G10-Cell2"
            and "identity confirmed by project owner" in row["identity_provenance"]):
        return 48 * rpt, "owner-confirmed frozen G10-Cell2 mapping and Oxford Part 3 schedule"
    raise ValueError("No independent RPT age provenance")


def rest_and_pulse(part):
    return {1: (21, 22), 2: (21, 22), 3: (19, 20)}[int(part)]


def pulse_droop(rest, pulse):
    """Final zero-current sample minus interpolated pulse voltage at StepTime 10 s.

    Samples are (TestTime minutes, StepTime minutes, Amps, Volts).
    """
    if not rest or len(pulse) < 2:
        raise ValueError("Missing adjacent rest or pulse samples")
    if not all(math.isfinite(v) for row in rest + pulse for v in row):
        raise ValueError("Non-finite pulse measurements")
    if any(abs(r[2]) > 0.01 for r in rest) or any(not -1.55 <= p[2] <= -1.45 for p in pulse):
        raise ValueError("Rest or nominal -1.5 A discharge-current check failed")
    if any(b[0] <= a[0] for a, b in zip(rest + pulse, (rest + pulse)[1:])):
        raise ValueError("Non-increasing TestTime at rest-to-pulse transition")
    if (pulse[0][0] - rest[-1][0]) * 60 > 1.5:
        raise ValueError("Rest-to-pulse sampling gap exceeds 1.5 s")
    times = [60 * p[1] for p in pulse]
    if not 0 <= times[0] <= 1.5 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Pulse onset or StepTime monotonicity check failed")
    for i, (left, right) in enumerate(zip(times, times[1:])):
        if left <= 10 <= right:
            if right - left > 1.5:
                raise ValueError("10 s sample is not covered at the expected one-second cadence")
            v0, v1 = pulse[i][3], pulse[i + 1][3]
            drop = rest[-1][3] - (v0 + (v1 - v0) * (10 - left) / (right - left))
            if not 0 <= drop <= 1:
                raise ValueError("Abnormal discharge droop outside 0?1 V")
            return drop, times[-1], left, right
    raise ValueError("Pulse does not bracket 10 s")


def raw_droop(root, row):
    reference = row["step10_segment_reference"]
    match = re.search(r"rows(\d+)-(\d+)", reference)
    if match is None:
        raise ValueError("Missing frozen Step 10 row anchor")
    end = int(match[2])
    rest_step, pulse_step = rest_and_pulse(row["dataset_part"])
    source = (root / row["source_file"]).resolve()
    if not source.is_relative_to(root / "data/raw/oxford"):
        raise ValueError("Source path outside Oxford data")
    rest, pulse = [], []
    rest_first = pulse_first = None
    previous = 10
    with source.open("rb", buffering=8 * 1024**2) as stream:
        fields = stream.readline().decode("utf-8-sig").strip().split(",")
        columns = [fields.index(k) for k in ("Step", "TestTime", "StepTime", "Amps", "Volts")]
        anchor = max(2, end - 4)
        deque(islice(stream, anchor - 2), maxlen=0)
        anchor_row = stream.readline().split(b",")
        if int(anchor_row[columns[0]]) != 10:
            raise ValueError("Frozen Step 10 row anchor does not match raw source")
        for line_num, line in enumerate(stream, start=anchor + 1):
            if line_num - anchor > 200000:
                raise ValueError("Pulse outside bounded Step 10 continuation")
            fields = line.split(b",")
            step = int(fields[columns[0]])
            if step < previous or (step == 10 and previous != 10):
                break
            if pulse and step != pulse_step:
                break
            if step > pulse_step:
                break
            if step in (rest_step, pulse_step):
                samples = tuple(float(fields[i]) for i in columns[1:])
                if step == rest_step:
                    if rest_first is None:
                        rest_first = line_num
                    rest.append(samples)
                else:
                    if pulse_first is None:
                        if previous != rest_step:
                            raise ValueError("Target pulse does not immediately follow target rest")
                        pulse_first = line_num
                    pulse.append(samples)
            previous = step
    drop, duration, left, right = pulse_droop(rest, pulse)
    return drop, f"Step {rest_step} rows {rest_first}-{pulse_first - 1}; Step {pulse_step} rows {pulse_first}-{pulse_first + len(pulse) - 1}", duration, left, right


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def key(row, ordinal="rpt_index"):
    return int(row["dataset_part"]), row["physical_cell_id"], int(row[ordinal])


def audit(root):
    frozen = root / "experiments/final2/training_6_models"
    manifest = json.loads((frozen / "frozen_dataset_manifest_v2.json").read_bytes())
    for name, sha in manifest["dataset_artifact_sha256"].items():
        if hashlib.sha256((frozen / "frozen_dataset" / name).read_bytes()).hexdigest() != sha:
            raise ValueError(f"Frozen hash mismatch: {name}")
    accepted = [r for r in read_csv(frozen / "frozen_dataset/observation_manifest.csv") if r["status"] == "ACCEPTED"]
    if len(accepted) != 213 or len({key(r) for r in accepted}) != 213:
        raise ValueError("Unexpected frozen cohort")
    mappings = defaultdict(list)
    for row in read_csv(root / "data/manifests/unified_observation_manifest.csv"):
        if row["rpt_order"].isdigit() and row["dataset_part"].isdigit():
            mappings[key(row, "rpt_order")].append(row)
    descriptors = {}
    for split, size in SPLITS.items():
        selected = {key(r) for r in accepted if r["final2_partition"] == split}
        with np.load(frozen / "frozen_dataset" / f"{split}.npz", allow_pickle=False) as data:
            keys = list(zip(data["dataset_part"], data["physical_cell_id"], data["rpt_index"]))
            values = data["X_physics_raw"][:, 2:]
            if len(keys) != size or set(keys) != selected or not np.isfinite(values).all():
                raise ValueError(f"Frozen descriptor alignment failed for {split}")
            descriptors.update(zip(keys, values.tolist()))

    def inspect(row):
        identity = key(row)
        result = {f: row[f] for f in ("final2_partition", "dataset_part", "group", "physical_cell_id", "rpt_index", "source_tpg", "source_file", "step4_segment_reference", "step10_segment_reference", "step4_source_sha256_snapshot", "step10_source_sha256_snapshot")}
        result.update(zip(FEATURES[2:], descriptors[identity]))
        result.update({f+"_unit": u for f, u in zip(FEATURES, UNITS)})
        errors = []
        try:
            matches = mappings[identity]
            if len(matches) > 1:
                raise ValueError("Ambiguous Guide age provenance")
            result[FEATURES[0]], result["age_provenance"] = nominal_age(row, next(iter(matches), None))
        except ValueError as error:
            errors.append(f"age: {error}")
        try:
            values = raw_droop(root, row)
            fields = (FEATURES[1], "pulse_source_rows", "pulse_duration_s", "interpolation_left_s", "interpolation_right_s")
            result.update(zip(fields, values))
        except (ValueError, OSError, IndexError) as error:
            errors.append(f"pulse: {error}")
        result.update(status="verified" if not errors else "blocked", missing_reason="; ".join(errors))
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(inspect, accepted))
    by_cell = defaultdict(list)
    for row in results:
        by_cell[row["physical_cell_id"]].append(row)
    for cell, rows in by_cell.items():
        ordered = sorted(rows, key=lambda r: (int(r["dataset_part"]), int(r["rpt_index"])))
        ages = [row[FEATURES[0]] for row in ordered if FEATURES[0] in row]
        if any(right <= left for left, right in zip(ages, ages[1:])):
            raise ValueError(f"Nominal age is not increasing across the recorded parts for {cell}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = audit(Path(__file__).resolve().parents[3])
    output = args.output.resolve()
    if output.exists():
        parser.error("Audit output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=output.parent, delete=False) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(stream.name)
    temporary.rename(output)
    counts = Counter((row["final2_partition"], row["status"]) for row in rows)
    print(dict(counts))
    if any(row["status"] != "verified" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
