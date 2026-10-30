"""Command-line orchestration for the fixed baseline evaluation workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from crop_recommendation.data.preprocessing import prepare_data_split
from crop_recommendation.data.validation import default_dataset_path
from crop_recommendation.training.evaluate import EvaluationResult, evaluate_model
from crop_recommendation.training.train import TrainingMetadata, train_model


@dataclass(frozen=True)
class BaselineEvaluationReport:
    """Reproducibility context plus the measured baseline evaluation."""

    dataset_path: str
    dataset_sha256: str
    split_configuration: dict[str, float | int]
    training_metadata: TrainingMetadata
    evaluation: EvaluationResult

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_path": self.dataset_path,
            "dataset_sha256": self.dataset_sha256,
            "split_configuration": self.split_configuration,
            "training_metadata": self.training_metadata.to_dict(),
            "evaluation": self.evaluation.to_dict(),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_baseline_evaluation(
    dataset_path: str | Path | None = None,
) -> BaselineEvaluationReport:
    """Run the fixed split, train-only fit, and held-out evaluation workflow."""

    path = Path(dataset_path) if dataset_path is not None else default_dataset_path()
    path = path.expanduser().resolve()
    split = prepare_data_split(path)
    training = train_model(split.X_train, split.y_train)
    evaluation = evaluate_model(training.estimator, split.X_test, split.y_test)

    return BaselineEvaluationReport(
        dataset_path=str(path),
        dataset_sha256=_sha256(path),
        split_configuration={
            "test_size": split.config.test_size,
            "random_state": split.config.random_state,
        },
        training_metadata=training.metadata,
        evaluation=evaluation,
    )


def format_baseline_report(report: BaselineEvaluationReport) -> str:
    """Format all measured metrics, class rows, and confusion counts as text."""

    training = report.training_metadata
    evaluation = report.evaluation
    lines = [
        "Crop Recommendation - Baseline Evaluation",
        "",
        f"Dataset SHA-256: {report.dataset_sha256}",
        f"Model: {training.model_type}",
        f"Trees: {training.model_configuration['n_estimators']}",
        f"Random seed: {training.random_seed}",
        f"Split test size: {report.split_configuration['test_size']}",
        f"Training samples: {training.training_rows}",
        f"Test samples: {evaluation.test_sample_count}",
        f"Classes: {evaluation.evaluation_metadata.class_count}",
        f"Scikit-learn: {evaluation.evaluation_metadata.sklearn_version}",
        "",
        f"Accuracy:        {evaluation.accuracy:.12f}",
        f"Macro Precision: {evaluation.macro_precision:.12f}",
        f"Macro Recall:    {evaluation.macro_recall:.12f}",
        f"Macro F1:        {evaluation.macro_f1:.12f}",
        "",
        "Per-class report:",
        "class,precision,recall,f1,support",
    ]
    for label in evaluation.class_labels:
        metrics = evaluation.class_report[label]
        lines.append(
            f"{label},{metrics.precision:.12f},{metrics.recall:.12f},"
            f"{metrics.f1:.12f},{metrics.support}"
        )

    lines.extend(
        [
            "",
            "Confusion matrix (rows=actual, columns=predicted):",
            "labels," + ",".join(evaluation.class_labels),
        ]
    )
    for label, row in zip(
        evaluation.class_labels, evaluation.confusion_matrix, strict=True
    ):
        lines.append(label + "," + ",".join(str(value) for value in row))

    if evaluation.warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {warning}" for warning in evaluation.warnings)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the fixed crop recommendation baseline."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete report as JSON instead of formatted text.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="CSV path (defaults to data/raw/Crop_recommendation.csv).",
    )
    arguments = parser.parse_args(argv)
    report = run_baseline_evaluation(arguments.dataset)
    if arguments.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_baseline_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
