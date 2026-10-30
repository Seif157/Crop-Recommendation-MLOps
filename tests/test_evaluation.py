"""Tests for prediction-only evaluation of fitted crop classifiers."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.base import BaseEstimator, ClassifierMixin

from crop_recommendation.data.preprocessing import prepare_data_split
from crop_recommendation.data.validation import (
    BASELINE_CROP_CLASSES,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from crop_recommendation.training.evaluate import EvaluationError, evaluate_model
from crop_recommendation.training.train import create_training_pipeline, train_model


class FixedPredictionEstimator(ClassifierMixin, BaseEstimator):
    """Fitted test double whose fit method must never be called."""

    def __init__(
        self,
        predictions: tuple[str, ...],
        classes: tuple[str, ...],
        feature_names: tuple[str, ...] = FEATURE_COLUMNS,
    ) -> None:
        self.predictions = predictions
        self.classes = classes
        self.feature_names = feature_names
        self.classes_ = list(classes)
        self.feature_names_in_ = list(feature_names)

    def __sklearn_is_fitted__(self) -> bool:
        return True

    def fit(self, X: object, y: object) -> "FixedPredictionEstimator":
        raise AssertionError("evaluate_model must never call fit")

    def predict(self, X: pd.DataFrame) -> list[str]:
        if len(X) != len(self.predictions):
            raise ValueError("Unexpected probe size")
        return list(self.predictions)


@pytest.fixture
def evaluation_data() -> tuple[pd.DataFrame, pd.Series, tuple[str, ...]]:
    labels = BASELINE_CROP_CLASSES[:4]
    records: list[dict[str, float | int]] = []
    for row_id in range(8):
        records.append(
            {
                "N": row_id,
                "P": row_id + 5,
                "K": row_id + 10,
                "temperature": 20.0 + row_id / 10,
                "humidity": 50.0 + row_id / 10,
                "ph": 6.0 + row_id / 100,
                "rainfall": 100.0 + row_id,
            }
        )
    X_test = pd.DataFrame(records, columns=FEATURE_COLUMNS)
    y_test = pd.Series(
        [
            labels[0],
            labels[0],
            labels[1],
            labels[1],
            labels[2],
            labels[2],
            labels[3],
            labels[3],
        ],
        name=TARGET_COLUMN,
    )
    predictions = (
        labels[0],
        labels[1],
        labels[1],
        labels[1],
        labels[2],
        labels[3],
        labels[3],
        labels[0],
    )
    return X_test, y_test, predictions


def _estimator(predictions: tuple[str, ...]) -> FixedPredictionEstimator:
    return FixedPredictionEstimator(predictions, BASELINE_CROP_CLASSES[:4])


def test_accuracy_is_computed_correctly(
    evaluation_data: tuple[pd.DataFrame, pd.Series, tuple[str, ...]],
) -> None:
    X_test, y_test, predictions = evaluation_data

    result = evaluate_model(_estimator(predictions), X_test, y_test)

    assert result.accuracy == pytest.approx(5 / 8)


def test_macro_precision_is_computed_correctly(
    evaluation_data: tuple[pd.DataFrame, pd.Series, tuple[str, ...]],
) -> None:
    X_test, y_test, predictions = evaluation_data

    result = evaluate_model(_estimator(predictions), X_test, y_test)

    assert result.macro_precision == pytest.approx((1 / 2 + 2 / 3 + 1 + 1 / 2) / 4)


def test_macro_recall_is_computed_correctly(
    evaluation_data: tuple[pd.DataFrame, pd.Series, tuple[str, ...]],
) -> None:
    X_test, y_test, predictions = evaluation_data

    result = evaluate_model(_estimator(predictions), X_test, y_test)

    assert result.macro_recall == pytest.approx((1 / 2 + 1 + 1 / 2 + 1 / 2) / 4)


def test_macro_f1_is_computed_correctly(
    evaluation_data: tuple[pd.DataFrame, pd.Series, tuple[str, ...]],
) -> None:
    X_test, y_test, predictions = evaluation_data

    result = evaluate_model(_estimator(predictions), X_test, y_test)

    assert result.macro_f1 == pytest.approx((1 / 2 + 4 / 5 + 2 / 3 + 1 / 2) / 4)


def test_confusion_matrix_counts(
    evaluation_data: tuple[pd.DataFrame, pd.Series, tuple[str, ...]],
) -> None:
    X_test, y_test, predictions = evaluation_data

    result = evaluate_model(_estimator(predictions), X_test, y_test)

    assert result.confusion_matrix == (
        (1, 1, 0, 0),
        (0, 2, 0, 0),
        (0, 0, 1, 1),
        (1, 0, 0, 1),
    )


def test_class_order_is_stable(
    evaluation_data: tuple[pd.DataFrame, pd.Series, tuple[str, ...]],
) -> None:
    X_test, y_test, predictions = evaluation_data
    estimator = FixedPredictionEstimator(
        predictions, tuple(reversed(BASELINE_CROP_CLASSES[:4]))
    )

    result = evaluate_model(estimator, X_test, y_test)

    assert result.class_labels == BASELINE_CROP_CLASSES[:4]


def test_confusion_matrix_dimensions(
    evaluation_data: tuple[pd.DataFrame, pd.Series, tuple[str, ...]],
) -> None:
    X_test, y_test, predictions = evaluation_data

    result = evaluate_model(_estimator(predictions), X_test, y_test)

    assert len(result.confusion_matrix) == 4
    assert all(len(row) == 4 for row in result.confusion_matrix)


def test_per_class_support(
    evaluation_data: tuple[pd.DataFrame, pd.Series, tuple[str, ...]],
) -> None:
    X_test, y_test, predictions = evaluation_data

    result = evaluate_model(_estimator(predictions), X_test, y_test)

    assert {metrics.support for metrics in result.class_report.values()} == {2}


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), -1.0])
def test_invalid_feature_values_are_rejected(
    evaluation_data: tuple[pd.DataFrame, pd.Series, tuple[str, ...]],
    invalid_value: float,
) -> None:
    X_test, y_test, predictions = evaluation_data
    invalid_features = X_test.copy()
    invalid_features["N"] = invalid_features["N"].astype(float)
    invalid_features.loc[0, "N"] = invalid_value

    with pytest.raises(EvaluationError, match="validation failed"):
        evaluate_model(_estimator(predictions), invalid_features, y_test)


def test_feature_order_mismatch_is_rejected(
    evaluation_data: tuple[pd.DataFrame, pd.Series, tuple[str, ...]],
) -> None:
    X_test, y_test, predictions = evaluation_data
    reordered = X_test[["P", "N", *FEATURE_COLUMNS[2:]]]

    with pytest.raises(EvaluationError, match="feature order is invalid"):
        evaluate_model(_estimator(predictions), reordered, y_test)


def test_missing_feature_column_is_rejected(
    evaluation_data: tuple[pd.DataFrame, pd.Series, tuple[str, ...]],
) -> None:
    X_test, y_test, predictions = evaluation_data

    with pytest.raises(EvaluationError, match="missing required feature columns"):
        evaluate_model(_estimator(predictions), X_test.drop(columns="rainfall"), y_test)


def test_features_must_match_fitted_estimator(
    evaluation_data: tuple[pd.DataFrame, pd.Series, tuple[str, ...]],
) -> None:
    X_test, y_test, predictions = evaluation_data
    fitted_order = ("P", "N", *FEATURE_COLUMNS[2:])
    estimator = FixedPredictionEstimator(predictions, BASELINE_CROP_CLASSES[:4], fitted_order)

    with pytest.raises(EvaluationError, match="do not match the estimator"):
        evaluate_model(estimator, X_test, y_test)


def test_target_column_is_rejected_inside_features(
    evaluation_data: tuple[pd.DataFrame, pd.Series, tuple[str, ...]],
) -> None:
    X_test, y_test, predictions = evaluation_data

    with pytest.raises(EvaluationError, match="Target leakage detected"):
        evaluate_model(_estimator(predictions), X_test.assign(label=y_test), y_test)


def test_unfitted_estimator_is_rejected(
    evaluation_data: tuple[pd.DataFrame, pd.Series, tuple[str, ...]],
) -> None:
    X_test, y_test, _ = evaluation_data

    with pytest.raises(EvaluationError, match="must be fitted"):
        evaluate_model(create_training_pipeline(), X_test, y_test)


@pytest.mark.parametrize("empty_input", ["features", "target"])
def test_empty_evaluation_inputs_are_rejected(
    evaluation_data: tuple[pd.DataFrame, pd.Series, tuple[str, ...]],
    empty_input: str,
) -> None:
    X_test, y_test, predictions = evaluation_data
    if empty_input == "features":
        X_test = X_test.iloc[0:0]
    else:
        y_test = y_test.iloc[0:0]

    with pytest.raises(EvaluationError, match="is empty"):
        evaluate_model(_estimator(predictions), X_test, y_test)


def test_feature_target_length_mismatch_is_rejected(
    evaluation_data: tuple[pd.DataFrame, pd.Series, tuple[str, ...]],
) -> None:
    X_test, y_test, predictions = evaluation_data

    with pytest.raises(EvaluationError, match="same number of rows"):
        evaluate_model(_estimator(predictions), X_test, y_test.iloc[:-1])


def test_feature_target_index_mismatch_is_rejected(
    evaluation_data: tuple[pd.DataFrame, pd.Series, tuple[str, ...]],
) -> None:
    X_test, y_test, predictions = evaluation_data
    misaligned = y_test.copy()
    misaligned.index = range(1, len(misaligned) + 1)

    with pytest.raises(EvaluationError, match="indices must match"):
        evaluate_model(_estimator(predictions), X_test, misaligned)


def test_unknown_target_label_is_rejected(
    evaluation_data: tuple[pd.DataFrame, pd.Series, tuple[str, ...]],
) -> None:
    X_test, y_test, predictions = evaluation_data
    unknown_target = y_test.copy()
    unknown_target.iloc[0] = BASELINE_CROP_CLASSES[4]

    with pytest.raises(EvaluationError, match="unknown to the fitted estimator"):
        evaluate_model(_estimator(predictions), X_test, unknown_target)


def test_unknown_prediction_label_is_rejected(
    evaluation_data: tuple[pd.DataFrame, pd.Series, tuple[str, ...]],
) -> None:
    X_test, y_test, predictions = evaluation_data
    invalid_predictions = (*predictions[:-1], BASELINE_CROP_CLASSES[4])

    with pytest.raises(EvaluationError, match="outside its known classes"):
        evaluate_model(_estimator(invalid_predictions), X_test, y_test)


def test_undefined_metrics_use_zero_and_report_affected_classes(
    evaluation_data: tuple[pd.DataFrame, pd.Series, tuple[str, ...]],
) -> None:
    X_test, y_test, _ = evaluation_data
    predictions = tuple(BASELINE_CROP_CLASSES[0] for _ in range(len(y_test)))

    result = evaluate_model(_estimator(predictions), X_test, y_test)

    assert result.class_report[BASELINE_CROP_CLASSES[1]].precision == 0.0
    assert result.warnings
    assert BASELINE_CROP_CLASSES[1] in " ".join(result.warnings)


def test_original_features_are_unchanged(
    evaluation_data: tuple[pd.DataFrame, pd.Series, tuple[str, ...]],
) -> None:
    X_test, y_test, predictions = evaluation_data
    original = X_test.copy(deep=True)

    evaluate_model(_estimator(predictions), X_test, y_test)

    assert_frame_equal(X_test, original)


def test_original_target_is_unchanged(
    evaluation_data: tuple[pd.DataFrame, pd.Series, tuple[str, ...]],
) -> None:
    X_test, y_test, predictions = evaluation_data
    original = y_test.copy(deep=True)

    evaluate_model(_estimator(predictions), X_test, y_test)

    assert_series_equal(y_test, original)


def test_evaluation_does_not_call_fit(
    evaluation_data: tuple[pd.DataFrame, pd.Series, tuple[str, ...]],
) -> None:
    X_test, y_test, predictions = evaluation_data

    result = evaluate_model(_estimator(predictions), X_test, y_test)

    assert result.test_sample_count == len(y_test)


def test_evaluation_does_not_alter_estimator(
    evaluation_data: tuple[pd.DataFrame, pd.Series, tuple[str, ...]],
) -> None:
    X_test, y_test, predictions = evaluation_data
    estimator = _estimator(predictions)
    classes_before = list(estimator.classes_)
    features_before = list(estimator.feature_names_in_)

    evaluate_model(estimator, X_test, y_test)

    assert estimator.classes_ == classes_before
    assert estimator.feature_names_in_ == features_before


def test_repeated_evaluation_is_identical(
    evaluation_data: tuple[pd.DataFrame, pd.Series, tuple[str, ...]],
) -> None:
    X_test, y_test, predictions = evaluation_data
    estimator = _estimator(predictions)

    first = evaluate_model(estimator, X_test, y_test)
    second = evaluate_model(estimator, X_test, y_test)

    assert first.to_dict() == second.to_dict()


def test_result_is_json_serializable(
    evaluation_data: tuple[pd.DataFrame, pd.Series, tuple[str, ...]],
) -> None:
    X_test, y_test, predictions = evaluation_data
    result = evaluate_model(_estimator(predictions), X_test, y_test)

    serialized = json.dumps(result.to_dict(), allow_nan=False)

    assert '"macro_f1"' in serialized


def test_confusion_counts_sum_to_evaluated_samples(
    evaluation_data: tuple[pd.DataFrame, pd.Series, tuple[str, ...]],
) -> None:
    X_test, y_test, predictions = evaluation_data
    result = evaluate_model(_estimator(predictions), X_test, y_test)

    assert sum(sum(row) for row in result.confusion_matrix) == len(y_test)


def test_evaluation_works_without_original_csv(
    evaluation_data: tuple[pd.DataFrame, pd.Series, tuple[str, ...]],
) -> None:
    X_test, y_test, predictions = evaluation_data

    result = evaluate_model(_estimator(predictions), X_test, y_test)

    assert result.test_sample_count == 8


@pytest.mark.integration
def test_real_held_out_partition_evaluation() -> None:
    split = prepare_data_split()
    X_train_before = split.X_train.copy(deep=True)
    y_train_before = split.y_train.copy(deep=True)
    X_test_before = split.X_test.copy(deep=True)
    y_test_before = split.y_test.copy(deep=True)
    training = train_model(split.X_train, split.y_train)
    classifier = training.estimator.named_steps["classifier"]
    tree_identities = tuple(id(tree) for tree in classifier.estimators_)

    result = evaluate_model(training.estimator, split.X_test, split.y_test)

    assert result.test_sample_count == 440
    assert len(result.class_labels) == 22
    assert len(result.confusion_matrix) == 22
    assert all(len(row) == 22 for row in result.confusion_matrix)
    assert all(
        0.0 <= metric <= 1.0
        for metric in (
            result.accuracy,
            result.macro_precision,
            result.macro_recall,
            result.macro_f1,
        )
    )
    assert sum(sum(row) for row in result.confusion_matrix) == 440
    assert tuple(id(tree) for tree in classifier.estimators_) == tree_identities
    assert_frame_equal(split.X_train, X_train_before)
    assert_series_equal(split.y_train, y_train_before)
    assert_frame_equal(split.X_test, X_test_before)
    assert_series_equal(split.y_test, y_test_before)
