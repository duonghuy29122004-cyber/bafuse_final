import json

import numpy as np
import pandas as pd

from bafuse.labeling.visualize_pydma_fits import _read_source_curves


def test_read_source_curves_uses_audited_monotone_segment(tmp_path):
    capacities = np.r_[np.linspace(0.0, 0.9, 10), np.linspace(0.0, 1.0, 11)]
    frame = pd.DataFrame(
        {
            "Step": [8] * len(capacities),
            "Amphr": capacities,
            "Amps": [-0.1] * len(capacities),
            "Volts": 2.5 + capacities * 1.5,
        }
    )
    path = tmp_path / "rpt.csv"
    frame.to_csv(path, index=False)
    selected = {
        "segment_index": 2,
        "sample_count": 11,
        "capacity_min": 0.0,
        "capacity_max": 1.0,
        "voltage_min": 2.5,
        "voltage_max": 4.0,
    }
    record = {
        "fit_id": 1,
        "test_step": 8,
        "direction": "discharge",
        "audit": {
            "step": 8,
            "direction": "discharge",
            "selected_segment": 2,
            "segments": [{}, selected],
        },
    }

    result = _read_source_curves(path, [record])[1]

    assert result[0].size == 11
    np.testing.assert_allclose(result[0], np.linspace(0.0, 1.0, 11))
    np.testing.assert_allclose(result[1], 2.5 + result[0] * 1.5)


def test_read_source_curves_uses_explicit_candidate_source_rows(tmp_path):
    capacities = np.r_[np.linspace(0.0, 1.0, 11), np.linspace(0.0, 0.8, 9)]
    frame = pd.DataFrame(
        {
            "Step": [8] * len(capacities),
            "Amphr": capacities,
            "Amps": [-0.1] * len(capacities),
            "Volts": 2.5 + capacities * 1.5,
        }
    )
    path = tmp_path / "candidate.csv"
    frame.to_csv(path, index=False)
    record = {
        "fit_id": 2,
        "test_step": 8,
        "direction": "discharge",
        "audit": {
            "step": 8,
            "direction": "discharge",
            "candidate_segment_index": 1,
            "source_row_start": 0,
            "source_row_end": 10,
            "curve_sample_count": 11,
        },
    }

    result = _read_source_curves(path, [record])[2]

    assert result[0].size == 11
    np.testing.assert_allclose(result[0], np.linspace(0.0, 1.0, 11))
