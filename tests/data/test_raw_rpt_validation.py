from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from bafuse.data.raw_rpt_validation import (
    MAX_TIME_DELTAS_FOR_MEDIAN,
    PHYSICS_FEATURE_NAMES,
    SegmentReference,
    SegmentRequest,
    _audit_one_candidate,
    _block_validity,
    _reference_disposition,
    _scan_source_file,
    parse_segment_reference,
)


def _append_rows(
    rows: list[dict[str, str]],
    *,
    step: int,
    count: int,
    start_voltage: float,
    end_voltage: float,
    current: float,
    start_capacity: float,
    start_time: float,
    temperature: bool = True,
) -> None:
    for index in range(count):
        fraction = index / (count - 1)
        row = {
            "Step": str(step),
            "TestTime": str(start_time + index * 0.1),
            "StepTime": str(index * 0.1),
            "Amphr": str(start_capacity + index * 0.01),
            "Amps": str(current),
            "Volts": str(start_voltage + fraction * (end_voltage - start_voltage)),
        }
        if temperature:
            row["Temp1"] = str(24.0 + index * 0.01)
        rows.append(row)


def _write_two_step_source(path: Path, *, temperature: bool = True) -> tuple[tuple[int, int], tuple[int, int]]:
    rows: list[dict[str, str]] = []
    _append_rows(
        rows,
        step=4,
        count=140,
        start_voltage=4.2,
        end_voltage=2.5,
        current=-1.5,
        start_capacity=0.0,
        start_time=0.0,
        temperature=temperature,
    )
    for index in range(3):
        boundary = {
            "Step": "7",
            "TestTime": str(14.0 + index * 0.1),
            "StepTime": str(index * 0.1),
            "Amphr": "1.4",
            "Amps": "0",
            "Volts": "2.5",
        }
        if temperature:
            boundary["Temp1"] = "25"
        rows.append(boundary)
    _append_rows(
        rows,
        step=10,
        count=140,
        start_voltage=2.5,
        end_voltage=4.2,
        current=0.125,
        start_capacity=1.4,
        start_time=14.3,
        temperature=temperature,
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return (2, 141), (145, 284)


def _request(
    path: Path,
    step: int,
    start: int,
    end: int,
    source_part: str = "1",
) -> SegmentRequest:
    reference = SegmentReference(
        source_text=f"source.csv#Step{step}:segment1:rows{start}-{end}",
        source_relative="source.csv",
        step=step,
        start_line=start,
        end_line=end,
    )
    return SegmentRequest(
        block_key=f"{path.name}|{step}|{start}|{end}",
        source_path=path,
        source_part=source_part,
        reference=reference,
    )


class SegmentReferenceTests(unittest.TestCase):
    def test_parser_keeps_exact_step_and_contiguous_source_line_range(self) -> None:
        reference = parse_segment_reference(
            "data/raw/oxford/part1/Group/TPG1.csv#Step10:segment7:rows100-199:sub1"
        )
        self.assertEqual(reference.step, 10)
        self.assertEqual((reference.start_line, reference.end_line), (100, 199))
        self.assertEqual(
            reference.source_relative,
            "data/raw/oxford/part1/Group/TPG1.csv",
        )

    def test_parser_rejects_reversed_or_unmapped_ranges(self) -> None:
        with self.assertRaises(ValueError):
            parse_segment_reference("TPG.csv#Step4:segment1:rows20-10")
        with self.assertRaises(ValueError):
            parse_segment_reference("TPG.csv#Step4:segment1")


class StreamingSegmentTests(unittest.TestCase):
    def test_scanner_validates_exact_step4_and_step10_ranges_without_joining_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "two_steps.csv"
            step4_range, step10_range = _write_two_step_source(source)
            step4_request = _request(source, 4, *step4_range)
            step10_request = _request(source, 10, *step10_range)

            results, file_metadata = _scan_source_file(
                source,
                [step4_request, step10_request],
            )

        step4 = results[step4_request.block_key]
        step10 = results[step10_request.block_key]
        self.assertEqual(step4["row_count"], 140)
        self.assertEqual(step4["observed_step_ids"], "4")
        self.assertAlmostEqual(step4["voltage_min_v"], 2.5)
        self.assertEqual(step10["row_count"], 140)
        self.assertEqual(step10["observed_step_ids"], "10")
        self.assertAlmostEqual(step10["voltage_max_v"], 4.2)
        self.assertAlmostEqual(
            step4["testtime_integral_native_current_time"],
            -20.85,
        )
        self.assertAlmostEqual(
            step4["testtime_integral_if_minutes_native_capacity"],
            20.85 / 60,
        )
        self.assertEqual(file_metadata["csv_data_rows_consumed_to_last_required_line"], 283)
        self.assertEqual(_block_validity(step4, 4)[0], "ACCEPTED")
        self.assertEqual(_block_validity(step10, 10)[0], "ACCEPTED")
        self.assertEqual(
            set(step4["physics_features"]),
            set(PHYSICS_FEATURE_NAMES),
        )
        self.assertFalse(any("capacity" in name.lower() for name in step4["physics_features"]))

    def test_wrong_boundary_step_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "two_steps.csv"
            _write_two_step_source(source)
            wrong_request = _request(source, 4, 141, 145)
            results, _ = _scan_source_file(source, [wrong_request])

        status, reasons = _block_validity(results[wrong_request.block_key], 4)
        self.assertEqual(status, "REJECTED")
        self.assertIn("observed_step_ids_do_not_match_expected", reasons)

    def test_inherited_cutoff_tolerance_is_used_without_inventing_a_new_one(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "two_steps.csv"
            step4_range, _ = _write_two_step_source(source)
            request = _request(source, 4, *step4_range)
            results, _ = _scan_source_file(source, [request])

        metrics = results[request.block_key]
        metrics["voltage_min_v"] = 2.500038
        self.assertEqual(_block_validity(metrics, 4)[0], "REVIEW")
        self.assertEqual(
            _block_validity(metrics, 4, step4_cutoff_tolerance_v=0.05)[0],
            "ACCEPTED",
        )
        self.assertEqual(
            _block_validity(metrics, 4, step4_cutoff_tolerance_v=0.00001)[0],
            "REJECTED",
        )

    def test_native_curve_with_two_points_is_not_rejected_by_a_128_row_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "two_points.csv"
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    ["Step", "TestTime", "StepTime", "Amphr", "Amps", "Volts", "Temp1"]
                )
                writer.writerow([4, 0, 0, 0, -1.5, 4.2, 24])
                writer.writerow([4, 1, 1, 1.5, -1.5, 2.5, 24])
            request = _request(source, 4, 2, 3)
            results, _ = _scan_source_file(source, [request])

        self.assertEqual(_block_validity(results[request.block_key], 4)[0], "ACCEPTED")

    def test_step10_short_of_endpoint_is_review_pending_protocol_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "two_steps.csv"
            _, step10_range = _write_two_step_source(source)
            request = _request(source, 10, *step10_range)
            results, _ = _scan_source_file(source, [request])

        metrics = results[request.block_key]
        metrics["voltage_max_v"] = 4.1999
        status, reasons = _block_validity(metrics, 10)
        self.assertEqual(status, "REVIEW")
        self.assertIn("step10_does_not_reach_documented_4_2v_charge_endpoint", reasons)

    def test_missing_temperature_keeps_voltage_curve_but_not_physics_features(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "no_temperature.csv"
            step4_range, _ = _write_two_step_source(source, temperature=False)
            request = _request(source, 4, *step4_range)
            results, _ = _scan_source_file(source, [request])

        result = results[request.block_key]
        self.assertEqual(_block_validity(result, 4)[0], "ACCEPTED")
        self.assertEqual(result["temperature_column_status"], "temperature_column_absent")
        self.assertEqual(result["missing_temperature_rows"], 140)
        self.assertEqual(result["physics_features"], {})
        self.assertEqual(result["physics_status"], "incomplete_required_step4_physics_inputs")

    def test_nonincreasing_time_axis_yields_review_not_silent_integration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "time_reset.csv"
            step4_range, _ = _write_two_step_source(source)
            data = source.read_text(encoding="utf-8").splitlines()
            fields = data[10].split(",")
            fields[1] = data[9].split(",")[1]
            data[10] = ",".join(fields)
            source.write_text("\n".join(data) + "\n", encoding="utf-8")
            request = _request(source, 4, *step4_range)
            results, _ = _scan_source_file(source, [request])

        status, reasons = _block_validity(results[request.block_key], 4)
        self.assertEqual(status, "REVIEW")
        self.assertIn("time_axis_has_reset_or_nonincreasing_interval", reasons)

    def test_streaming_time_diagnostics_have_a_fixed_memory_sample_cap(self) -> None:
        self.assertEqual(MAX_TIME_DELTAS_FOR_MEDIAN, 2048)


class LabelDispositionTests(unittest.TestCase):
    def test_part1_needs_measurement_validity_and_verified_integration_units(self) -> None:
        common = {
            "mapping_verified": True,
            "step4_status": "ACCEPTED",
            "bol_status": "ACCEPTED",
            "label_sources_match": True,
            "candidate_label_matches_source": True,
            "arithmetic_matches": True,
        }
        disposition, reasons = _reference_disposition(
            part="1",
            integration_units_verified=False,
            **common,
        )
        self.assertEqual(disposition, "REVIEW")
        self.assertIn("current_integration_time_and_current_units_unverified", reasons)

    def test_part2_and_part3_candidates_remain_review_until_policy_approval(self) -> None:
        common = {
            "mapping_verified": True,
            "step4_status": "ACCEPTED",
            "bol_status": "ACCEPTED",
            "label_sources_match": True,
            "candidate_label_matches_source": True,
            "arithmetic_matches": True,
            "integration_units_verified": True,
        }
        for part in ("2", "3"):
            with self.subTest(part=part):
                disposition, reasons = _reference_disposition(part=part, **common)
                self.assertEqual(disposition, "REVIEW")
                self.assertIn(
                    "part_label_policy_remains_provisional_pending_user_approval",
                    reasons,
                )

    def test_incomplete_discharge_reference_is_rejected(self) -> None:
        disposition, _ = _reference_disposition(
            part="1",
            mapping_verified=True,
            step4_status="REJECTED",
            bol_status="ACCEPTED",
            label_sources_match=True,
            candidate_label_matches_source=True,
            arithmetic_matches=True,
            integration_units_verified=True,
        )
        self.assertEqual(disposition, "REJECTED")


class CandidateAuditTests(unittest.TestCase):
    def test_candidate_record_audit_applies_configured_step4_tolerance(self) -> None:
        candidate = {
            "_key": ("1", "G1-Cell9", "1"),
            "_observation": {},
            "_label": None,
            "_mapping_status": "verified",
            "_bol_resolution_status": "resolved",
            "part": "1",
            "partition": "train",
        }
        result = _audit_one_candidate(candidate, {}, {}, 0.05)

        self.assertEqual(result["dataset_part"], "1")
        self.assertFalse(result["primary_benchmark_eligible"])
        self.assertFalse(result["training_authorized"])


if __name__ == "__main__":
    unittest.main()
