from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from bafuse.labeling.pydma_policy import (
    PYDMA_MIN_ACCEPTED_RUNS,
    PYDMA_RMSE_THRESHOLD_MV,
    pydma_fit_passes_gate,
)
from bafuse.labeling.pydma_runner import _fit_one


class PyDMAPolicyTests(unittest.TestCase):
    def test_strict_rmse_threshold_and_minimum_accepted_runs(self) -> None:
        self.assertEqual(PYDMA_RMSE_THRESHOLD_MV, 15.0)
        self.assertEqual(PYDMA_MIN_ACCEPTED_RUNS, 3)
        self.assertTrue(
            pydma_fit_passes_gate(
                rmse_mV=14.999,
                accepted_runs=3,
                is_accepted=True,
            )
        )
        for rmse_mV, accepted_runs, is_accepted in (
            (15.0, 3, True),
            (15.001, 3, True),
            (14.0, 2, True),
            (14.0, 3, False),
            (None, 3, True),
        ):
            with self.subTest(
                rmse_mV=rmse_mV,
                accepted_runs=accepted_runs,
                is_accepted=is_accepted,
            ):
                self.assertFalse(
                    pydma_fit_passes_gate(
                        rmse_mV=rmse_mV,
                        accepted_runs=accepted_runs,
                        is_accepted=is_accepted,
                    )
                )

    def test_optimizer_uses_the_same_15_mv_threshold(self) -> None:
        result = SimpleNamespace(
            rmse=0.014,
            n_accepted_runs=3,
            n_total_runs=3,
            is_accepted=True,
        )
        with (
            patch("bafuse.labeling.pydma_runner.DMAConfig") as dma_config,
            patch("bafuse.labeling.pydma_runner.DMAAnalyzer") as dma_analyzer,
        ):
            dma_analyzer.return_value.analyze.return_value = result
            _fit_one(
                "discharge",
                np.array([0.0, 1.0]),
                np.array([4.0, 3.0]),
                object(),
                object(),
                seed=1,
            )

        self.assertEqual(
            dma_config.call_args.kwargs["rmse_threshold"],
            PYDMA_RMSE_THRESHOLD_MV / 1000.0,
        )


if __name__ == "__main__":
    unittest.main()
