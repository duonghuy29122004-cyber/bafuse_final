from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from bafuse.features.soh_voltage import (
    EXPLORATORY_PARTITION_SIZES,
    EXPLORATORY_SPLIT_VERSION,
    EXPECTED_PARTITION_SIZES,
    FINAL_PARTITION_SIZES,
    FINAL_SPLIT_VERSION,
    SPLIT_VERSION,
    extract_rpt_charge_voltages,
    extract_step4_voltage,
    load_split,
    load_split_manifest,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _split_rows() -> list[dict[str, str]]:
    rows = []
    n = 0
    for partition, size in EXPECTED_PARTITION_SIZES.items():
        for _ in range(size):
            n += 1
            rows.append(
                {
                    "split_version": SPLIT_VERSION,
                    "partition": partition,
                    "physical_cell_id": f"G{n}-Cell{n}",
                }
            )
    return rows


def _exploratory_split_rows() -> list[dict[str, str]]:
    rows = []
    n = 0
    for partition, size in EXPLORATORY_PARTITION_SIZES.items():
        for _ in range(size):
            n += 1
            rows.append(
                {
                    "split_version": EXPLORATORY_SPLIT_VERSION,
                    "partition": partition,
                    "physical_cell_id": f"G{n}-Cell{n}",
                }
            )
    return rows


def _final_split_rows() -> list[dict[str, str]]:
    rows = []
    n = 0
    for partition, size in FINAL_PARTITION_SIZES.items():
        for _ in range(size):
            n += 1
            rows.append(
                {
                    "split_version": FINAL_SPLIT_VERSION,
                    "partition": partition,
                    "physical_cell_id": f"G{n}-Cell{n}",
                }
            )
    return rows


class SplitTests(unittest.TestCase):
    def test_split_sizes_are_locked_and_cells_unique(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.csv"
            rows = _split_rows()
            _write_csv(path, rows)
            self.assertEqual(len(load_split(path)), 27)
            rows[1]["partition"] = "validation"
            _write_csv(path, rows)
            with self.assertRaises(ValueError):
                load_split(path)
            rows[1]["partition"] = "train"
            rows[1]["physical_cell_id"] = rows[0]["physical_cell_id"]
            _write_csv(path, rows)
            with self.assertRaises(ValueError):
                load_split(path)

    def test_exploratory_split_is_supported_without_replacing_locked_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "exploratory_split.csv"
            _write_csv(path, _exploratory_split_rows())

            manifest = load_split_manifest(path)

            self.assertEqual(manifest.version, EXPLORATORY_SPLIT_VERSION)
            self.assertEqual(manifest.partition_sizes, EXPLORATORY_PARTITION_SIZES)
            with self.assertRaises(ValueError):
                load_split(path)

    def test_final_split_is_supported_without_replacing_locked_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "final_split.csv"
            _write_csv(path, _final_split_rows())

            manifest = load_split_manifest(path)

            self.assertEqual(manifest.version, FINAL_SPLIT_VERSION)
            self.assertEqual(manifest.partition_sizes, FINAL_PARTITION_SIZES)
            self.assertEqual(len(manifest.cell_to_partition), 28)
            with self.assertRaises(ValueError):
                load_split(path)


class ExtractionTests(unittest.TestCase):
    def test_segment_selection_across_chunk_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "smoke.csv"
            steps = [4] * 200 + [3] * 2 + [4] * 180
            volts = list(np.linspace(4.2, 2.5, 200)) + [3.0, 3.0] + list(
                np.linspace(4.1, 2.5, 180)
            )
            pd.DataFrame({"Step": steps, "Volts": volts}).to_csv(path, index=False)
            _, first = extract_step4_voltage(path, 1, chunk_rows=199)
            _, second = extract_step4_voltage(path, 2, chunk_rows=199)
            self.assertEqual((first["step4_rows"], second["step4_rows"]), (200, 180))
            with self.assertRaises(ValueError):
                extract_step4_voltage(path, 3)

    def test_charge_curve_is_bound_to_step4_rpt_and_protocol_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "charge.csv"
            rows = []

            def append(step: int, count: int, amps: float, start: float, end: float) -> None:
                for voltage in np.linspace(start, end, count):
                    rows.append(
                        {
                            "Step": step,
                            "Amps": amps,
                            "Volts": voltage,
                        }
                    )

            append(6, 90, 1.0, 3.8, 4.2)
            append(4, 80, -1.5, 4.2, 2.5)
            append(5, 4, 0.0, 2.5, 3.0)
            append(6, 200, 1.0, 3.2, 4.2)
            append(7, 4, 0.0, 4.2, 4.2)
            append(8, 8, -0.1, 4.2, 2.5)
            append(4, 80, -1.5, 4.2, 2.5)
            append(5, 70, 1.5, 2.5, 3.6)
            append(6, 80, 0.1, 3.6, 4.2)
            append(7, 4, 0.0, 4.2, 4.2)
            append(8, 8, -0.1, 4.2, 2.5)
            pd.DataFrame(rows).to_csv(path, index=False)

            part12_curves, part12_metadata = extract_rpt_charge_voltages(
                path, [1], "2", chunk_rows=37
            )
            all_part12_curves, all_part12_metadata = extract_rpt_charge_voltages(
                path, [1, 2], "2", chunk_rows=37
            )
            part3_curves, part3_metadata = extract_rpt_charge_voltages(
                path, [2], "3", chunk_rows=37
            )

        self.assertEqual(part12_metadata[1]["status"], "ok")
        self.assertEqual(part12_metadata[1]["step4_segment_count"], 2)
        self.assertEqual(all_part12_metadata[1]["step4_segment_count"], 2)
        self.assertEqual(all_part12_curves[1].shape, (128,))
        self.assertEqual(part12_metadata[1]["charge_steps_used"], "6")
        self.assertEqual(part3_metadata[2]["status"], "ok")
        self.assertEqual(part3_metadata[2]["charge_steps_used"], "5,6")
        self.assertEqual(part12_curves[1].shape, (128,))
        self.assertEqual(part3_curves[2].shape, (128,))
        self.assertAlmostEqual(float(part12_curves[1][0]), 3.2)
        self.assertAlmostEqual(float(part3_curves[2][0]), 2.5)

    def test_charge_curve_requires_step8_boundary_and_positive_current(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid_charge.csv"
            rows = []

            def append(step: int, count: int, amps: float) -> None:
                for voltage in np.linspace(3.0, 4.2, count):
                    rows.append({"Step": step, "Amps": amps, "Volts": voltage})

            append(4, 80, -1.0)
            append(5, 4, 0.0)
            append(6, 80, 1.0)
            append(4, 80, -1.0)
            append(5, 4, 0.0)
            append(6, 72, 1.0)
            append(6, 8, -0.5)
            append(8, 8, -0.1)
            pd.DataFrame(rows).to_csv(path, index=False)

            curves, metadata = extract_rpt_charge_voltages(
                path, [1, 2], "1", chunk_rows=31
            )

        self.assertNotIn(1, curves)
        self.assertEqual(metadata[1]["status"], "missing_step8_boundary")
        self.assertNotIn(2, curves)
        self.assertEqual(metadata[2]["status"], "invalid_charge_current_direction")


if __name__ == "__main__":
    unittest.main()
