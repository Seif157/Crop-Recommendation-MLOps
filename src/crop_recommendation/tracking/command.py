"""Command-line workflow for recording the established baseline in MLflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from crop_recommendation.data.preprocessing import prepare_data_split
from crop_recommendation.data.validation import default_dataset_path
from crop_recommendation.tracking.mlflow_tracking import (
    DEFAULT_EXPERIMENT_NAME,
    MlflowTrackingConfig,
    default_artifact_root,
    default_tracking_uri,
    log_baseline_experiment,
)
from crop_recommendation.training.evaluate import evaluate_model
from crop_recommendation.training.train import train_model


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run and record the fixed crop baseline in MLflow."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="CSV path (defaults to data/raw/Crop_recommendation.csv).",
    )
    parser.add_argument(
        "--tracking-uri",
        default=default_tracking_uri(),
        help="MLflow tracking URI (defaults to repository-local SQLite).",
    )
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--run-name",
        default="random-forest-fixed-holdout",
        help="MLflow run display name.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print tracking identities and measured metrics as JSON.",
    )
    arguments = parser.parse_args(argv)

    dataset_path = (
        arguments.dataset.expanduser().resolve()
        if arguments.dataset is not None
        else default_dataset_path()
    )
    split = prepare_data_split(dataset_path)
    training = train_model(split.X_train, split.y_train)
    evaluation = evaluate_model(training.estimator, split.X_test, split.y_test)
    config = MlflowTrackingConfig(
        tracking_uri=arguments.tracking_uri,
        artifact_root=default_artifact_root(),
        experiment_name=arguments.experiment_name,
        run_name=arguments.run_name,
    )
    result = log_baseline_experiment(
        training,
        evaluation,
        split,
        dataset_path,
        config,
    )

    output = {
        **result.to_dict(),
        "metrics": {
            "accuracy": evaluation.accuracy,
            "macro_precision": evaluation.macro_precision,
            "macro_recall": evaluation.macro_recall,
            "macro_f1": evaluation.macro_f1,
            "test_sample_count": evaluation.test_sample_count,
        },
    }
    if arguments.json:
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        print("Crop Recommendation - MLflow Baseline Run")
        print(f"Experiment: {result.experiment_name} ({result.experiment_id})")
        print(f"Run ID: {result.run_id}")
        print(f"Model: {result.model_uri}")
        if result.model_id is not None:
            print(f"Model ID: {result.model_id}")
        print(f"Tracking URI: {result.tracking_uri}")
        print(f"Accuracy: {evaluation.accuracy:.12f}")
        print(f"Macro F1: {evaluation.macro_f1:.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
