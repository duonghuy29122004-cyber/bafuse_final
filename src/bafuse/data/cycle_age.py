from __future__ import annotations

import numpy as np
import pandas as pd


NOMINAL_CYCLES_PER_RPT = 48


def validated_cycle_age(
    frame: pd.DataFrame, allow_non_nominal_age: bool
) -> np.ndarray:
    raw_age = frame["nominal_cycle_age"]
    age = pd.to_numeric(raw_age, errors="coerce")
    if (raw_age.notna() & age.isna()).any():
        raise ValueError("nominal_cycle_age contains a non-numeric value.")
    status = frame["cycle_age_status"].astype(str)
    nominal = status == "nominal_48_cycles_per_RPT"
    if allow_non_nominal_age:
        part = frame["dataset_part"].astype(str)
        group = frame["group_id"].astype(str)
        group5 = (
            (part == "2")
            & (group == "5")
            & (status == "cadence_not_established")
        )
        group6 = (
            (part == "2")
            & (group == "6")
            & (status == "calendar_aging_only")
        )
        if not (nominal | group5 | group6).all():
            raise ValueError("Unsupported cycle-age provenance in the input rows.")
        if age[~nominal].notna().any():
            raise ValueError("Non-nominal age rows must not contain a cycle-age value.")
    elif not nominal.all():
        raise ValueError(
            "Nominal-age proxy is only valid for rows flagged nominal_48_cycles_per_RPT."
        )

    expected_age = NOMINAL_CYCLES_PER_RPT * pd.to_numeric(
        frame["rpt_order"], errors="raise"
    )
    nominal_age = age[nominal].to_numpy(dtype=float)
    expected_nominal_age = expected_age[nominal].to_numpy(dtype=float)
    if not np.all(np.isfinite(nominal_age)) or not np.allclose(
        nominal_age, expected_nominal_age
    ):
        raise ValueError("nominal_cycle_age is not 48 x RPT order for every nominal row.")
    return age.to_numpy(dtype=float)
