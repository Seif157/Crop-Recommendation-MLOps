"""Release-gate verification for the complete Phase 1 baseline workflow."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from crop_recommendation.data.preprocessing import DataSplit, prepare_data_split
from crop_recommendation.data.validation import ValidationReport, validate_dataset
from crop_recommendation.training.evaluate import EvaluationResult, evaluate_model
from crop_recommendation.training.train import TrainingResult, train_model


@dataclass(frozen=True)
class WorkflowRun:
    """Observable outputs from one independent Phase 1 execution."""

    validation: ValidationReport
    split: DataSplit
    training: TrainingResult
    predictions: tuple[str, ...]
    evaluation: EvaluationResult


def _run_phase_one_workflow() -> WorkflowRun:
    validation = validate_dataset()
    split = prepare_data_split()
    training = train_model(split.X_train, split.y_train)
    predictions = tuple(
        str(value) for value in training.estimator.predict(split.X_test)
    )
    evaluation = evaluate_model(training.estimator, split.X_test, split.y_test)
    return WorkflowRun(validation, split, training, predictions, evaluation)


@pytest.mark.integration
def test_real_phase_one_workflow_is_reproducible_and_self_consistent() -> None:
    """Two independent runs must agree and satisfy cross-component contracts."""

    first = _run_phase_one_workflow()
    second = _run_phase_one_workflow()

    assert first.validation.is_valid
    assert first.validation.total_rows == 2_200
    assert first.validation.to_dict() == second.validation.to_dict()
    assert (
        tuple(first.validation.feature_columns)
        == first.training.metadata.feature_names
    )

    assert first.split.config == second.split.config
    assert_frame_equal(first.split.X_train, second.split.X_train)
    assert_frame_equal(first.split.X_test, second.split.X_test)
    assert_series_equal(first.split.y_train, second.split.y_train)
    assert_series_equal(first.split.y_test, second.split.y_test)
    assert set(first.split.X_train.index).isdisjoint(first.split.X_test.index)
    assert len(first.split.X_train.index) + len(first.split.X_test.index) == (
        first.validation.total_rows
    )
    assert set(first.split.train_class_distribution.values()) == {80}
    assert set(first.split.test_class_distribution.values()) == {20}

    assert first.training.metadata.to_dict() == second.training.metadata.to_dict()
    assert first.training.metadata.training_rows == len(first.split.X_train)
    assert (
        first.training.metadata.class_distribution
        == first.split.train_class_distribution
    )
    assert first.predictions == second.predictions
    assert first.evaluation.to_dict() == second.evaluation.to_dict()
    assert first.evaluation.test_sample_count == first.split.test_rows
    assert (
        first.evaluation.evaluation_metadata.feature_names
        == first.split.feature_names
        == first.training.metadata.feature_names
    )
    assert (
        first.evaluation.evaluation_metadata.class_count
        == first.training.metadata.class_count
        == len(first.evaluation.class_labels)
    )
    assert (
        first.evaluation.evaluation_metadata.sklearn_version
        == first.training.metadata.sklearn_version
    )

    labels = first.evaluation.class_labels
    predictions = list(first.predictions)
    assert first.evaluation.accuracy == pytest.approx(
        accuracy_score(first.split.y_test, predictions)
    )
    assert first.evaluation.macro_precision == pytest.approx(
        precision_score(
            first.split.y_test,
            predictions,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )
    assert first.evaluation.macro_recall == pytest.approx(
        recall_score(
            first.split.y_test,
            predictions,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )
    assert first.evaluation.macro_f1 == pytest.approx(
        f1_score(
            first.split.y_test,
            predictions,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )

    independent_matrix = confusion_matrix(
        first.split.y_test, predictions, labels=labels
    )
    assert first.evaluation.confusion_matrix == tuple(
        tuple(int(value) for value in row) for row in independent_matrix.tolist()
    )
    assert sum(sum(row) for row in first.evaluation.confusion_matrix) == len(
        first.split.y_test
    )
    assert sum(
        metrics.support for metrics in first.evaluation.class_report.values()
    ) == len(first.split.y_test)
