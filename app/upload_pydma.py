from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
HALF_CELL_ROOT = ROOT / "data" / "raw" / "oxford" / "part1" / "Half_Cell" / "Half Cell"
REQUIRED_COLUMNS = (
    "role",
    "part",
    "group",
    "physical_cell_id",
    "checkpoint_id",
    "source_file",
    "direction",
    "Step",
    "Amphr",
    "Amps",
    "Volts",
)
FIT_ROLES = ("bol", "target")


class UploadedCurveError(ValueError):
    pass


def _validate_bundle(text: str) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    try:
        frame = pd.read_csv(io.StringIO(text))
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise UploadedCurveError(f"Could not parse uploaded PyDMA CSV: {error}") from error
    missing = set(REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise UploadedCurveError(
            "PyDMA CSV is missing required columns: " + ", ".join(sorted(missing))
        )
    unexpected = set(frame.columns).difference(REQUIRED_COLUMNS)
    if unexpected:
        raise UploadedCurveError(
            "PyDMA CSV has unsupported columns: " + ", ".join(sorted(unexpected))
        )
    if frame.empty:
        raise UploadedCurveError("PyDMA CSV contains no curve rows.")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise UploadedCurveError("PyDMA CSV contains missing required values.")
    frame["role"] = frame["role"].astype(str).str.strip().str.casefold()
    if set(frame["role"].unique()) != set(FIT_ROLES):
        raise UploadedCurveError("CSV must contain exactly the `bol` and `target` curve roles.")

    identities: dict[str, dict[str, str]] = {}
    numeric_columns = ("Step", "Amphr", "Amps", "Volts")
    for role in FIT_ROLES:
        curve = frame.loc[frame["role"].eq(role)].copy()
        if curve.empty:
            raise UploadedCurveError(f"CSV has no rows for the {role} curve.")
        for column in ("part", "group", "physical_cell_id", "checkpoint_id", "source_file", "direction"):
            values = curve[column].astype(str).str.strip().unique()
            if len(values) != 1 or not values[0]:
                raise UploadedCurveError(
                    f"{role} rows must have one non-empty {column} value."
                )
        identity = {
            column: str(curve.iloc[0][column]).strip()
            for column in (
                "part",
                "group",
                "physical_cell_id",
                "checkpoint_id",
                "source_file",
                "direction",
            )
        }
        identity["direction"] = identity["direction"].casefold()
        if identity["direction"] not in {"charge", "discharge"}:
            raise UploadedCurveError("Direction must be `charge` or `discharge`.")
        for column in numeric_columns:
            curve[column] = pd.to_numeric(curve[column], errors="coerce")
        if not np.isfinite(curve[list(numeric_columns)].to_numpy(dtype=float)).all():
            raise UploadedCurveError(f"{role} curve has non-numeric or non-finite measurements.")
        frame.loc[curve.index, list(numeric_columns)] = curve[list(numeric_columns)]
        identities[role] = identity

    bol = identities["bol"]
    target = identities["target"]
    for key in ("part", "group", "physical_cell_id", "direction"):
        if bol[key] != target[key]:
            raise UploadedCurveError(
                f"BoL and target curves must match on {key}; got {bol[key]!r} and {target[key]!r}."
            )
    if bol["checkpoint_id"].casefold() not in {"bol", "bol-reference", "bol_reference"}:
        raise UploadedCurveError("The baseline checkpoint_id must explicitly identify BoL.")
    if target["checkpoint_id"].casefold() in {"bol", "bol-reference", "bol_reference"}:
        raise UploadedCurveError("The target checkpoint_id must identify a non-BoL checkpoint.")
    try:
        part = int(bol["part"])
        group = int(bol["group"])
    except ValueError as error:
        raise UploadedCurveError("Part and Group metadata must be integer identifiers.") from error
    if part not in {1, 2, 3}:
        raise UploadedCurveError("Part must be 1, 2, or 3.")
    if part == 2 and group in {5, 6}:
        raise UploadedCurveError("Part 2 Groups 5 and 6 are excluded from this PyDMA mode.")
    return frame, identities


def _direction_references(direction: str) -> tuple[Any, Any]:
    from pydma.electrodes.electrode import ElectrodeOCP
    from pydma.preprocessing.loader import load_ocp

    if not HALF_CELL_ROOT.is_dir():
        raise FileNotFoundError(f"Direction-matched half-cell references are missing: {HALF_CELL_ROOT}")
    if direction == "charge":
        return (
            load_ocp(
                HALF_CELL_ROOT / "CH_HC_anode.csv",
                electrode_type="anode",
                soc_col="SoC",
                voltage_col="Volts",
            ),
            load_ocp(
                HALF_CELL_ROOT / "CH_HC_cathode.csv",
                electrode_type="cathode",
                soc_col="SoC",
                voltage_col="Volts",
            ),
        )

    anode_frame = pd.read_csv(
        HALF_CELL_ROOT / "DCH_HC_anode.csv", usecols=["SoC", "Volts"]
    )
    finite = np.isfinite(anode_frame["SoC"].to_numpy(dtype=float)) & np.isfinite(
        anode_frame["Volts"].to_numpy(dtype=float)
    )
    if not finite.any():
        raise ValueError("No finite discharge-anode reference pairs are available.")
    return (
        ElectrodeOCP(
            soc=anode_frame.loc[finite, "SoC"].to_numpy(dtype=float),
            voltage=anode_frame.loc[finite, "Volts"].to_numpy(dtype=float),
            name="DCH_HC_anode_valid_pairs",
            electrode_type="anode",
        ),
        load_ocp(
            HALF_CELL_ROOT / "DHC_HC_cathode.csv",
            electrode_type="cathode",
            soc_col="SoC",
            voltage_col="Volts",
        ),
    )


def run_uploaded_pydma(
    text: str,
    *,
    fit_runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    from bafuse.labeling.pydma_policy import pydma_fit_passes_gate
    from bafuse.labeling.pydma_runner import _fit_one, _read_pocv_with_audit
    from pydma import DMAAnalyzer, DMAConfig

    frame, identities = _validate_bundle(text)
    direction = identities["target"]["direction"]
    step = 10 if direction == "charge" else 8
    references = _direction_references(direction)
    fitter = fit_runner or _fit_one
    fits: dict[str, Any] = {}
    curve_audits: dict[str, dict[str, Any]] = {}
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    with tempfile.TemporaryDirectory(prefix="bafuse-pydma-upload-") as directory:
        root = Path(directory)
        for role in FIT_ROLES:
            path = root / f"{role}.csv"
            frame.loc[frame["role"].eq(role), ["Step", "Amphr", "Amps", "Volts"]].to_csv(
                path, index=False
            )
            curve, audit = _read_pocv_with_audit(path, step, direction)
            curve_audits[role] = audit
            if curve is None:
                raise UploadedCurveError(
                    f"{role} curve did not yield a valid monotone Step {step} {direction} segment: "
                    f"{audit.get('status')}"
                )
            curves[role] = curve
            capacity, voltage = curve
            if capacity.size < 10 or np.any(np.diff(capacity) <= 0):
                raise UploadedCurveError(f"{role} curve failed monotonic capacity validation.")
            fits[role] = fitter(
                direction,
                capacity,
                voltage,
                *references,
                seed=211,
                context=(
                    f"uploaded {role} part={identities[role]['part']} "
                    f"cell={identities[role]['physical_cell_id']} "
                    f"checkpoint={identities[role]['checkpoint_id']}"
                ),
                use_cuda=torch.cuda.is_available(),
            )

    fit_results: dict[str, dict[str, Any]] = {}
    for role, fit in fits.items():
        rmse_mv = float(fit.rmse) * 1000
        accepted = pydma_fit_passes_gate(
            rmse_mV=rmse_mv,
            accepted_runs=int(fit.n_accepted_runs),
            is_accepted=bool(fit.is_accepted),
        )
        fit_results[role] = {
            "checkpoint_id": identities[role]["checkpoint_id"],
            "source_file_user_supplied": identities[role]["source_file"],
            "rmse_mV": rmse_mv,
            "accepted_runs": int(fit.n_accepted_runs),
            "total_runs": int(fit.n_total_runs),
            "is_accepted_by_original_gate": accepted,
            "gate": "RMSE < 15 mV and at least 3 accepted runs",
            "selected_segment_audit": curve_audits[role],
            "selected_sample_count": int(curves[role][0].size),
            "capacity_axis_units": "uploaded Amphr values; units are not independently verified",
        }

    estimates: dict[str, float] | None = None
    estimate_status = "withheld_fit_rejected"
    if all(fit_results[role]["is_accepted_by_original_gate"] for role in FIT_ROLES):
        comparison = DMAAnalyzer(
            config=DMAConfig(direction=direction),
            anode=references[0],
            cathode=references[1],
        ).compare_with_reference(
            reference_params=fits["bol"].fitted_params,
            current_params=fits["target"].fitted_params,
            reference_capacity=float(np.ptp(curves["bol"][0])),
            current_capacity=float(np.ptp(curves["target"][0])),
        )
        estimates = {
            "LLI_percent": float(comparison.lli * 100),
            "Anode_LAM_percent": float(comparison.lam_anode * 100),
            "Cathode_LAM_percent": float(comparison.lam_cathode * 100),
        }
        estimate_status = "user_uploaded_diagnostic_estimates"

    metadata = identities["target"]
    return {
        "part": metadata["part"],
        "group": metadata["group"],
        "physical_cell_id": metadata["physical_cell_id"],
        "direction": direction,
        "step": step,
        "target_checkpoint_id": metadata["checkpoint_id"],
        "mapping_status": "user_provided_unverified_not_guide_audited",
        "bol_reference_status": "user_provided_same_cell_and_direction_not_independently_verified",
        "fit_results": fit_results,
        "estimate_status": estimate_status,
        "degradation_estimates": estimates,
        "negative_estimate_flags": (
            {
                name: value < 0
                for name, value in estimates.items()
            }
            if estimates is not None
            else None
        ),
        "interpretation": (
            "Experimental model-derived diagnostics only. Signed values are not clipped; "
            "the uploaded Guide/procedure/Step identity and capacity units were not independently validated."
        ),
    }
