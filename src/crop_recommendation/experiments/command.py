"""CLI for the fixed, training-only Random Forest candidate comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import yaml

from crop_recommendation.data.preprocessing import prepare_data_split
from crop_recommendation.data.validation import default_dataset_path, repository_root
from crop_recommendation.experiments.comparison import (
    DatasetLineage,
    compare_candidates,
    write_comparison_report,
)
from crop_recommendation.experiments.tracking import (
    DEFAULT_COMPARISON_REPORT_PATH,
    DEFAULT_CV_EXPERIMENT_NAME,
    default_cv_tracking_config,
    log_comparison,
)
from crop_recommendation.tracking.mlflow_tracking import MlflowTrackingConfig


def _hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset_lineage(dataset_path: str | Path) -> DatasetLineage:
    """Verify the workspace CSV against its native DVC pointer."""

    dataset = Path(dataset_path).expanduser().resolve()
    pointer = repository_root() / "data" / "raw" / "Crop_recommendation.csv.dvc"
    try:
        pointer_data = yaml.safe_load(pointer.read_text(encoding="utf-8"))
        output = pointer_data["outs"][0]
        dvc_md5 = str(output["md5"])
        expected_size = int(output["size"])
    except (OSError, KeyError, IndexError, TypeError, ValueError, yaml.YAMLError) as error:
        raise ValueError(f"Could not read DVC dataset pointer '{pointer}': {error}") from error
    if dataset != default_dataset_path().resolve():
        raise ValueError(
            "Controlled comparison requires the canonical DVC-tracked dataset."
        )
    actual_size = dataset.stat().st_size
    actual_md5 = _hash_file(dataset, "md5")
    if actual_size != expected_size or actual_md5 != dvc_md5:
        raise ValueError("Workspace dataset does not match its DVC pointer identity.")
    return DatasetLineage(
        path=dataset.relative_to(repository_root()).as_posix(),
        sha256=_hash_file(dataset, "sha256"),
        dvc_md5=dvc_md5,
        size=actual_size,
    )


def _build_parser() -> argparse.ArgumentParser:
    defaults = default_cv_tracking_config()
    parser = argparse.ArgumentParser(
        description=(
            "Compare RF-50, RF-100, and RF-200 using training-only "
            "stratified cross-validation and log the evidence to MLflow."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset_path(),
        help="Canonical DVC-tracked CSV path.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=defaults.tracking_uri,
        help="MLflow tracking URI.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=defaults.artifact_root,
        help="MLflow artifact root.",
    )
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_CV_EXPERIMENT_NAME,
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_COMPARISON_REPORT_PATH,
        help="Ignored local path for the stable comparison JSON report.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the complete result as JSON."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    dataset_path = arguments.dataset.expanduser().resolve()
    lineage = load_dataset_lineage(dataset_path)
    outer_split = prepare_data_split(dataset_path)
    X_train = outer_split.X_train
    y_train = outer_split.y_train
    del outer_split

    comparison = compare_candidates(X_train, y_train, lineage)
    tracking_config = MlflowTrackingConfig(
        tracking_uri=arguments.tracking_uri,
        artifact_root=arguments.artifact_root,
        experiment_name=arguments.experiment_name,
        run_name="candidate",
    )
    logged = log_comparison(comparison, tracking_config)
    write_comparison_report(comparison, arguments.report)

    output = logged.to_dict()
    output["stable_report_path"] = str(arguments.report.expanduser().resolve())
    if arguments.json:
        print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
        return 0

    print("Crop Recommendation - Training-only CV Comparison")
    print(f"Experiment: {logged.experiment_name} ({logged.experiment_id})")
    print("Protocol: 5-fold StratifiedKFold on 1,760 training rows only")
    print("Primary metric: mean CV macro F1")
    print("")
    print("Candidate  Mean Macro F1   Std Macro F1   Mean Accuracy")
    for result in comparison.candidates:
        print(
            f"{result.candidate.name:<10} "
            f"{result.mean_metrics['macro_f1']:.12f}  "
            f"{result.standard_deviation_metrics['macro_f1']:.12f}  "
            f"{result.mean_metrics['accuracy']:.12f}"
        )
    print("")
    print(f"Best observed mean: {comparison.best_observed_candidate}")
    print(f"Recommended candidate: {comparison.recommended_candidate}")
    print("Outer 440-row holdout used: no")
    for run in logged.runs:
        print(f"{run.candidate_name} run ID: {run.run_id}")
    print(f"Stable report: {arguments.report.expanduser().resolve()}")
    print("Recommendation is not production approval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
