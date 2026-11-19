from __future__ import annotations

PYDMA_RMSE_THRESHOLD_MV = 15.0
PYDMA_MIN_ACCEPTED_RUNS = 3


def pydma_fit_passes_gate(
    *,
    rmse_mV: float | None,
    accepted_runs: int | None,
    is_accepted: bool,
) -> bool:
    return bool(
        is_accepted
        and rmse_mV is not None
        and rmse_mV < PYDMA_RMSE_THRESHOLD_MV
        and accepted_runs is not None
        and accepted_runs >= PYDMA_MIN_ACCEPTED_RUNS
    )
