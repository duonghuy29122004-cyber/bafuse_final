from __future__ import annotations

import argparse
import csv
from pathlib import Path

DEFAULT_TARGET_CELLS = (("1", "20"), ("2", "4"), ("3", "11"), ("4", "19"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def priority_cells(labels_dir: Path) -> list[tuple[str, str]]:
    labels_path = labels_dir / "labels_degradation.csv"
    if not labels_path.is_file():
        return list(DEFAULT_TARGET_CELLS)

    rows = _read_csv(labels_path)
    by_cell: dict[tuple[str, str], dict[str, str | None]] = {}
    for row in rows:
        key = (str(row.get("group_id", "")).strip(), str(row.get("cell_id", "")).strip())
        if not key[0] or not key[1]:
            continue
        cell = by_cell.setdefault(key, {"charge": None, "discharge": None})
        direction = str(row.get("direction", "")).strip().casefold()
        status = str(row.get("identifiability_status", "")).strip()
        if direction == "charge" and cell["charge"] is None:
            cell["charge"] = status
        if direction == "discharge" and cell["discharge"] is None:
            cell["discharge"] = status

    missing_cell = [
        (group, cell)
        for (group, cell), state in by_cell.items()
        if state.get("charge") == "not_run" and state.get("discharge") == "not_run"
    ]
    partial_cell = [
        (group, cell)
        for (group, cell), state in by_cell.items()
        if (state.get("charge") == "not_run" and state.get("discharge") != "not_run")
        or (state.get("discharge") == "not_run" and state.get("charge") != "not_run")
    ]
    ordered = missing_cell + partial_cell
    if not ordered:
        return list(DEFAULT_TARGET_CELLS)

    unique: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in ordered:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def classify_priority_cells(labels_dir: Path) -> dict[str, list[tuple[str, str]]]:
    cells = priority_cells(labels_dir)
    priority_1 = []
    priority_2 = []
    rows = _read_csv(labels_dir / "labels_degradation.csv")
    by_cell: dict[tuple[str, str], dict[str, str | None]] = {}
    for row in rows:
        key = (str(row.get("group_id", "")).strip(), str(row.get("cell_id", "")).strip())
        if not key[0] or not key[1]:
            continue
        state = by_cell.setdefault(key, {"charge": None, "discharge": None})
        direction = str(row.get("direction", "")).strip().casefold()
        status = str(row.get("identifiability_status", "")).strip()
        if direction == "charge" and state["charge"] is None:
            state["charge"] = status
        if direction == "discharge" and state["discharge"] is None:
            state["discharge"] = status
    for key, state in by_cell.items():
        if state.get("charge") == "not_run" and state.get("discharge") == "not_run":
            priority_1.append(key)
        elif (state.get("charge") == "not_run" and state.get("discharge") != "not_run") or (
            state.get("discharge") == "not_run" and state.get("charge") != "not_run"
        ):
            priority_2.append(key)
    return {"priority_1": priority_1, "priority_2": priority_2, "ordered": cells}


def main() -> None:
    parser = argparse.ArgumentParser(description="List the highest-value missing/partial PyDMA cells in priority order.")
    parser.add_argument("--labels", type=Path, required=True, help="Path to labeling_output_part1_pydma_all")
    args = parser.parse_args()

    classified = classify_priority_cells(args.labels)
    print("PRIORITY_1_MISSING_CELL")
    print(classified["priority_1"])
    print("\nPRIORITY_2_MISSING_DIRECTION")
    print(classified["priority_2"])
    print("\nORDERED_FOR_RERUN")
    print(classified["ordered"])


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
