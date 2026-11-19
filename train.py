"""Train one official Final2 architecture from scratch into a new candidate directory."""
import argparse
import csv
from pathlib import Path
import sys

from benchmark import MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--output-dir", type=Path, help="New candidate directory within the repository.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = (root / args.output_dir).resolve() if args.output_dir is not None else None
    if output is not None and (output == root or not output.is_relative_to(root)):
        parser.error("--output-dir must be a new directory within the repository")
    coverage = root / "data/processed/physics/coverage.csv"
    if args.model == "CNN-3M":
        if not coverage.is_file():
            parser.exit(1, "CNN-3M training is blocked by unverified updated Physics features. "
                           "See docs/04_PHYSICS_AND_PYDMA.md.\n")
        with coverage.open(encoding="utf-8-sig", newline="") as stream:
            failures = [row for row in csv.DictReader(stream) if row["status"] != "verified"]
        if failures:
            for row in failures:
                print(f"{row['final2_partition']} {row['physical_cell_id']} "
                      f"RPT{row['rpt_index']}: {row['missing_reason']}", file=sys.stderr)
            parser.exit(1, f"CNN-3M candidate blocked: {len(failures)} invalid Physics observations.\n")
    sys.path.insert(0, str(root / "src"))
    from bafuse.training.updated_cnn import train_candidate

    try:
        candidate = train_candidate(root, coverage, model_name=args.model, output=output)
    except (ValueError, OSError) as error:
        parser.exit(1, f"{args.model} candidate blocked: {error}\n")
    print_results(candidate)


def print_results(candidate):
    selected = candidate["selected"]
    model = selected["model"]
    line = "=" * 61
    print(f"{line}\n             BaFuse Final2 — {model} RESULTS\n{line}")
    print(f"Model: {model}\nDevice: {candidate['device']}\n"
          f"Best Epoch: {selected['best_epoch']}\nCheckpoint: {candidate['checkpoint']}\n")
    print(f"{'Split':<16}{'N':>5}{'MAE':>11}{'RMSE':>11}{'R²':>11}")
    print("-" * 61)
    for name, metrics in (
        ("Train", selected["train"]), ("Validation", selected["validation"]),
        ("External Test", candidate["external"]),
    ):
        print(f"{name:<16}{metrics['N']:>5}{metrics['MAE']:>11.4f}"
              f"{metrics['RMSE']:>11.4f}{metrics['R2']:>11.4f}")
    print(f"{line}\nMAE / RMSE: SoH percentage points\nR²: dimensionless\n"
          f"External SoH labels: provisional\nPromotion: {candidate['promotion_status']}")


if __name__ == "__main__":
    main()
