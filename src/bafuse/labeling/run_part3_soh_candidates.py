from __future__ import annotations

import argparse
import json
from pathlib import Path

from .soh_label_generation import run_part3_soh_candidate_generation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate separate capacity-derived Part 3 SOH candidates."
    )
    parser.add_argument("--corrected-manifest", type=Path, required=True)
    parser.add_argument("--rpt-scan-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run_part3_soh_candidate_generation(
        args.corrected_manifest,
        args.rpt_scan_dir,
        args.dataset_root,
        args.output,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
