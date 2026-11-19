from __future__ import annotations

import hashlib

import pytest

from bafuse.labeling.raw_soh_reconstruction import (
    BlockRequest,
    _BlockAccumulator,
    _block_validity,
    _observed_cutoff_bracket,
    _parse_segment,
    _protocol_compatible,
    _write_report,
)


def _accumulator(expected_sha256: str = "") -> _BlockAccumulator:
    request = BlockRequest(
        source_file="data/raw/oxford/part1/example.csv",
        source_part=1,
        step=4,
        start_line=2,
        end_line=4,
        expected_sha256=expected_sha256,
        keep_voltage=True,
        keep_physics=True,
    )
    columns = {
        "step": 0,
        "testtime": 1,
        "steptime": 2,
        "amphr": 3,
        "amps": 4,
        "volts": 5,
        "temperature": 6,
    }
    return _BlockAccumulator(request, columns, width=7)


def test_streaming_integration_uses_prior_current_for_trapezoids():
    accumulator = _accumulator()
    rows = [
        ["4", "0", "0", "0", "-2", "4.1", "24"],
        ["4", "1", "1", "0.0333", "-2", "3.5", "24.5"],
        ["4", "2", "2", "0.0667", "-2", "2.5", "25"],
    ]
    for row in rows:
        accumulator.update(row)

    result = accumulator.finish(next_step=5)

    assert result["testtime_integral_native_current_time"] == pytest.approx(-4.0)
    assert result["steptime_integral_native_current_time"] == pytest.approx(-4.0)
    assert result["capacity_span_native_amphr"] == pytest.approx(0.0667)
    assert result["next_row_transitions_step"] is True
    expected_hash = hashlib.sha256(
        "".join("\x1f".join(row) + "\n" for row in rows).encode("utf-8")
    ).hexdigest()
    assert result["raw_block_sha256"] == expected_hash


def test_repeated_timestamps_remain_a_review_condition():
    accumulator = _accumulator()
    rows = [
        ["4", "0", "0", "0", "-1", "4.1", "24"],
        ["4", "1", "1", "0.02", "-1", "3.5", "24"],
        ["4", "1", "1", "0.03", "-1", "2.5", "24"],
    ]
    for row in rows:
        accumulator.update(row)
    block = accumulator.finish(next_step=5)

    accepted, reasons = _block_validity(
        block,
        expected_step=4,
        expected_current_sign="negative",
        require_endpoint=True,
    )

    assert not accepted
    assert "repeated_timestamp_interval" in reasons
    assert block["capacity_span_native_amphr"] == pytest.approx(0.03)


def test_zero_current_is_allowed_only_for_charge_step10():
    rows = (
        ["10", "0", "0", "0", "1", "3.8"],
        ["10", "1", "1", "0.1", "0", "3.9"],
        ["10", "2", "2", "0.2", "1", "4.0"],
    )
    expected_sha256 = hashlib.sha256(
        "".join("\x1f".join(row) + "\n" for row in rows).encode("utf-8")
    ).hexdigest()
    request = BlockRequest(
        source_file="data/raw/oxford/part3/example.csv",
        source_part=3,
        step=10,
        start_line=2,
        end_line=4,
        expected_sha256=expected_sha256,
        keep_voltage=True,
        keep_physics=False,
        keep_current=True,
    )
    accumulator = _BlockAccumulator(
        request,
        {"step": 0, "testtime": 1, "steptime": 2, "amphr": 3, "amps": 4, "volts": 5},
        width=6,
    )
    for row in rows:
        accumulator.update(row)
    block = accumulator.finish(next_step=11)

    accepted, reasons = _block_validity(
        block,
        expected_step=10,
        expected_current_sign="positive",
        allow_zero_current=True,
    )

    assert accepted
    assert not reasons
    assert block["zero_current_rows"][0]["line"] == 3
    assert block["positive_current_row_count"] == 2
    with pytest.raises(ValueError, match="only for charge Step 10"):
        _block_validity(
            block,
            expected_step=4,
            expected_current_sign="negative",
            allow_zero_current=True,
        )


def test_empirical_cutoff_code_is_not_a_broad_voltage_tolerance():
    rows = [
        ["4", "0", "0", "0", "-1", "4.1", "24"],
        ["4", "1", "1", "0.02", "-1", "2.500038147555", "24"],
        ["4", "2", "2", "0.04", "-1", "2.500038147555", "24"],
    ]
    expected_hash = hashlib.sha256(
        "".join("\x1f".join(row) + "\n" for row in rows).encode("utf-8")
    ).hexdigest()
    accumulator = _accumulator(expected_hash)
    for row in rows:
        accumulator.update(row)
    block = accumulator.finish(next_step=5)

    accepted, reasons = _block_validity(
        block,
        expected_step=4,
        expected_current_sign="negative",
        require_endpoint=True,
        closest_above_cutoff_v=2.500038147555,
    )

    assert accepted
    assert not reasons


def test_cutoff_bracket_comes_from_closest_observed_step4_samples():
    def finish_with_voltages(start_line: int, voltages: list[str]) -> dict:
        accumulator = _BlockAccumulator(
            BlockRequest(
                source_file=f"data/raw/oxford/part1/example{start_line}.csv",
                source_part=1,
                step=4,
                start_line=start_line,
                end_line=start_line + len(voltages) - 1,
                expected_sha256="",
                keep_voltage=True,
                keep_physics=False,
            ),
            {"step": 0, "testtime": 1, "steptime": 2, "amphr": 3, "amps": 4, "volts": 5},
            width=6,
        )
        for index, voltage in enumerate(voltages):
            accumulator.update(
                ["4", str(index), str(index), str(index), "-1", voltage]
            )
        return accumulator.finish(next_step=5)

    raw_blocks = {
        ("one", 1, 4, 1, 3): finish_with_voltages(
            1, ["4.1", "2.499900", "2.500200"]
        ),
        ("two", 1, 4, 4, 6): finish_with_voltages(
            4, ["4.1", "2.499970", "2.500030"]
        ),
    }

    assert _observed_cutoff_bracket(raw_blocks) == (2.499970, 2.500030)


def test_protocol_compatibility_uses_canonical_physical_cell_identity():
    row = {
        "dataset_part": "2",
        "bol_source_part": "1",
        "physical_cell_id": "G4-Cell19",
        "bol_resolution_status": "resolved",
        "reference_protocol_compatibility": "same_cell_step4_and_part_protocol_crosswalk",
        "bol_protocol": "Maccor 4200; Step 4 CC-CV discharge C/2 to 2.5 V",
    }

    assert _protocol_compatible(row)[0]

    row["physical_cell_id"] = "G5-Cell6"
    assert not _protocol_compatible(row)[0]
    row["physical_cell_id"] = "G4-Cell19"
    row["reference_protocol_compatibility"] = "unresolved_or_invalid"
    assert not _protocol_compatible(row)[0]
    row["physical_cell_id"] = "unresolved-cell"
    assert not _protocol_compatible(row)[0]


def test_report_includes_mapped_step4_cutoff_bracket(tmp_path):
    summary = {
        "generated_at_utc": "2026-11-13T00:00:00+00:00",
        "candidate_count": 0,
        "raw_blocks_reread_and_hash_verified": 0,
        "raw_source_files_reread": 0,
        "label_disposition_counts": {},
        "mapped_step4_closest_observed_voltage_below_cutoff_v": 2.4999,
        "mapped_step4_closest_observed_voltage_above_cutoff_v": 2.5001,
        "by_split": {},
        "modality_counts": {
            "step4_discharge_review": 0,
            "step10_charge_review": 0,
            "physics_review": 0,
            "complete_three_modality_eligible": 0,
        },
        "review_reason_counts": {},
        "historical_label_crosscheck_abs_difference_pp": {"n": 0},
        "secondary_bol_sensitivity_qc": {"n": 0},
        "per_cell_coverage": [],
    }

    report = tmp_path / "validation_report.md"
    _write_report(report, summary, [], [], [])

    assert (
        "2.499900000000 V` below and `2.500100000000 V` above"
        in report.read_text(encoding="utf-8")
    )


def test_segment_parser_uses_exact_mapped_step_and_row_range():
    assert _parse_segment(
        "data/raw/oxford/part1/Group 1/TPG1_4_Cell9.csv"
        "#Step4:segment1:rows320489-327336"
    ) == (
        "data/raw/oxford/part1/Group 1/TPG1_4_Cell9.csv",
        4,
        320489,
        327336,
    )

    with pytest.raises(ValueError, match="Unparseable"):
        _parse_segment("TPG1_4_Cell9.csv")
