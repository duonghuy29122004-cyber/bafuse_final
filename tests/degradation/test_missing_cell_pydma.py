from __future__ import annotations

import tempfile
import unittest
from functools import partial
from pathlib import Path

from bafuse.config import load_project_paths

import numpy as np
import torch
from pydma.core.objectives import combined_objective
from pydma.electrodes.electrode import ElectrodeOCP

from bafuse.labeling.pydma_runner import (
    _read_pocv_with_audit,
    _select_curve_candidate,
)
from bafuse.labeling.audit import GuideEntry
from bafuse.labeling.torch_pydma import TorchOCVEvaluator, cuda_differential_evolution
from bafuse.labeling.run_missing_cell_pydma import (
    _unaccepted_fit_targets,
)


class PseudoOcvCurveSegmentationTests(unittest.TestCase):
    def test_duplicate_file_candidate_with_more_valid_directions_wins(self) -> None:
        invalid = {
            "source": "TPG4_1_Cell19_2.csv",
            "valid_direction_count": 0,
            "selected_point_count": 0,
        }
        valid = {
            "source": "TPG4_1_Cell19_1.csv",
            "valid_direction_count": 2,
            "selected_point_count": 180_000,
        }

        self.assertIs(_select_curve_candidate([valid, invalid]), valid)
        self.assertIs(_select_curve_candidate([invalid]), invalid)

    def _write_repeated_discharge(self, path: Path) -> None:
        lines = ["Step,TestTime,Amphr,Amps,Volts"]
        for index, capacity in enumerate(np.linspace(0.0, 3.0, 20)):
            lines.append(f"8,{index},{capacity},-0.1,{4.1 - capacity * 0.5}")
        for index, capacity in enumerate(np.linspace(0.0, 3.0, 20), start=20):
            lines.append(f"8,{index},{capacity},-0.1,{4.1 - capacity * 0.5}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_capacity_reset_selects_one_complete_segment_and_logs_the_other(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "repeated_rpt.csv"
            self._write_repeated_discharge(path)

            curve, audit = _read_pocv_with_audit(path, step_number=8, direction="discharge")

        self.assertIsNotNone(curve)
        self.assertEqual(len(curve[0]), 20)
        self.assertTrue(np.all(np.diff(curve[0]) > 0))
        self.assertEqual(len(audit["segments"]), 2)
        self.assertEqual(audit["selected_segment"], 1)
        self.assertEqual(audit["selected_source_row_start"], 2)
        self.assertEqual(audit["selected_source_row_end"], 21)
        self.assertEqual(
            audit["selected_testtime_audit"]["status"],
            "verified_strictly_increasing",
        )
        self.assertIn("Never concatenate", audit["selection_rule"])

    def test_no_rows_for_direction_is_audited_without_fitting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "charge_only.csv"
            path.write_text(
                "Step,TestTime,Amphr,Amps,Volts\n" + "\n".join(
                    f"10,{index},{index / 10},0.1,{3.0 + index / 100}" for index in range(12)
                ) + "\n",
                encoding="utf-8",
            )

            curve, audit = _read_pocv_with_audit(path, step_number=8, direction="discharge")

        self.assertIsNone(curve)
        self.assertEqual(audit["status"], "no_direction_filtered_rows")
        self.assertEqual(audit["segments"], [])


class UnacceptedFitTargetSelectionTests(unittest.TestCase):
    def test_accepted_duplicates_are_excluded_and_only_not_run_slots_are_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            path.write_text(
                "group_id,cell_id,measurement_id,direction,source_file,fit_success,identifiability_status\n"
                "1,20,RPT 1,charge,Group_1/Group 1/TPG1.4 - Cell20.csv,False,withheld_fit_rejected\n"
                "1,20,RPT 1,charge,Group_1/Group 1/TPG1.4 - Cell20.csv,True,diagnostic_only_not_independently_validated\n"
                "1,20,RPT 1,discharge,Group_1/Group 1/TPG1.4 - Cell20.csv,False,withheld_fit_rejected\n"
                "1,20,RPT 2,charge,Group_1/Group 1/TPG1.6 - Cell20.csv,False,not_run_curve_unavailable\n"
                "1,20,BoL RPT,charge,Group_1/Group 1/TPG1 - Cell20.csv,False,reference_baseline\n",
                encoding="utf-8",
            )

            targets, summary = _unaccepted_fit_targets(
                [path],
                load_project_paths().dataset_parts["1"],
                [GuideEntry("1", "20", "TPG1.6", "RPT 2", 6)],
            )

        self.assertEqual(
            targets,
            {("1", "20"): {(4, "discharge"), (6, "charge")}},
        )
        self.assertEqual(
            summary,
            {
                "unique_fit_slots": 3,
                "already_accepted": 1,
                "not_run_for_curve_recheck": 1,
                "selected_for_optimizer_retry": 1,
                "selected_total_for_fit_or_curve_recheck": 2,
            },
        )


class CudaPyDmaObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        cls.device = torch.device("cuda:0")
        soc = np.linspace(0.0, 1.0, 101)
        cls.anode = ElectrodeOCP(
            soc=soc,
            voltage=0.35 - 0.25 * soc,
            name="synthetic_anode",
            electrode_type="anode",
        )
        cls.cathode = ElectrodeOCP(
            soc=soc,
            voltage=3.0 + 1.2 * soc,
            name="synthetic_cathode",
            electrode_type="cathode",
        )
        cls.q = np.linspace(0.0, 1.0, 500)
        cls.measured = 3.4 + 0.35 * cls.q
        cls.objective = partial(
            combined_objective,
            anode=cls.anode,
            cathode=cls.cathode,
            meas_voltage=cls.measured,
            meas_dva=np.zeros_like(cls.q),
            meas_ica=np.zeros_like(cls.q),
            q=cls.q,
            dva_roi_mask=np.ones_like(cls.q, dtype=bool),
            ica_roi_mask=np.ones_like(cls.q, dtype=bool),
            roi_ocv_min=0.05,
            roi_ocv_max=0.95,
            q0=1.0,
            w_ocv=100.0,
            w_dva=0.0,
            w_ica=0.0,
        )

    def test_cuda_objective_matches_pydma_cpu_objective(self) -> None:
        evaluator = TorchOCVEvaluator(self.objective, self.device)
        candidates = np.zeros((4, 9), dtype=np.float64)
        candidates[:, 0] = [1.1, 1.25, 1.4, 1.6]
        candidates[:, 1] = [-0.3, -0.2, -0.1, -0.05]
        candidates[:, 2] = [1.2, 1.35, 1.5, 1.7]
        candidates[:, 3] = [-0.25, -0.2, -0.1, -0.05]

        gpu_cost, gpu_rmse = evaluator.evaluate(
            torch.as_tensor(candidates, dtype=torch.float64, device=self.device)
        )
        cpu_cost = np.asarray([self.objective(candidate) for candidate in candidates])

        np.testing.assert_allclose(gpu_cost.cpu().numpy(), cpu_cost, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(
            gpu_rmse.cpu().numpy(), np.sqrt(cpu_cost / 100.0), rtol=1e-10, atol=1e-12
        )

    def test_cuda_differential_evolution_returns_finite_best_fit(self) -> None:
        result = cuda_differential_evolution(
            self.objective,
            [(1.0, 2.0), (-1.0, 0.0), (1.0, 2.1), (-1.0, 0.0)] + [(0.0, 0.0)] * 5,
            maxiter=3,
            popsize=5,
            tol=0.0,
            mutation=(0.5, 1.0),
            recombination=0.8,
            seed=211,
            x0=np.asarray([1.4, -0.2, 1.5, -0.2, 0, 0, 0, 0, 0], dtype=np.float64),
            polish=True,
        )

        self.assertEqual(result.x.shape, (9,))
        self.assertTrue(np.isfinite(result.fun))
        self.assertGreater(result.nfev, 0)


if __name__ == "__main__":
    unittest.main()