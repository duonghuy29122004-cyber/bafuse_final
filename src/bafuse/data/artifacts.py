from __future__ import annotations

from pathlib import Path

from bafuse.config import load_project_paths


_REPOSITORY_ROOT = load_project_paths().root
_EXPERIMENTS_ROOT = _REPOSITORY_ROOT / "experiments"
_ARTIFACTS_ROOT = _REPOSITORY_ROOT / "models" / "legacy_experiments"


def artifact_output_path(output_dir: Path, filename: str) -> Path:
    if not filename or Path(filename).name != filename:
        raise ValueError("Artifact filename must be a non-empty basename.")

    output_dir = output_dir.resolve()
    try:
        relative_run = output_dir.relative_to(_EXPERIMENTS_ROOT)
    except ValueError:
        artifact = output_dir / filename
    else:
        artifact = _ARTIFACTS_ROOT / relative_run / filename
    artifact.parent.mkdir(parents=True, exist_ok=True)
    return artifact


def resolve_run_artifact(run_dir: Path, filename: str) -> Path:
    if not filename or Path(filename).name != filename:
        raise ValueError("Artifact filename must be a non-empty basename.")

    run_dir = run_dir.resolve()
    direct_path = run_dir / filename
    if direct_path.is_file():
        return direct_path

    try:
        relative_run = run_dir.relative_to(_EXPERIMENTS_ROOT)
    except ValueError:
        relative_run = None
    if relative_run is not None:
        artifact_path = _ARTIFACTS_ROOT / relative_run / filename
        if artifact_path.is_file():
            return artifact_path
        raise FileNotFoundError(
            f"Run artifact is missing from both the experiment and artifact paths: "
            f"{direct_path}; {artifact_path}"
        )
    raise FileNotFoundError(f"Run artifact does not exist: {direct_path}")
