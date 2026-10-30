"""Evaluation of fitted crop classifiers on untouched test partitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
import sklearn
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from crop_recommendation.data.validation import (
    BASELINE_CROP_CLASSES,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_dataframe,
)


class EvaluationError(ValueError):
    """Raised when a fitted estimator or evaluation input violates the contract."""


@dataclass(frozen=True)
class ClassMetrics:
    """Unrounded classification metrics for one target class."""

    class_name: str
    precision: float
    recall: float
    f1: float
    support: int

    def to_dict(self) -> dict[str, str | float | int]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationMetadata:
    """Serializable context needed to interpret an evaluation result."""

    model_type: str
    feature_names: tuple[str, ...]
    class_count: int
    sklearn_version: str
    zero_division_policy: int

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["feature_names"] = list(self.feature_names)
        return result


@dataclass(frozen=True)
class EvaluationResult:
    """Machine-readable metrics from one prediction-only evaluation pass."""

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    test_sample_count: int
    class_labels: tuple[str, ...]
    class_report: dict[str, ClassMetrics]
    confusion_matrix: tuple[tuple[int, ...], ...]
    evaluation_metadata: EvaluationMetadata
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Convert NumPy-backed metric outputs into JSON-compatible Python values."""

        return {
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "test_sample_count": self.test_sample_count,
            "class_labels": list(self.class_labels),
            "class_report": {
                label: metrics.to_dict()
                for label, metrics in self.class_report.items()
            },
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
            "evaluation_metadata": self.evaluation_metadata.to_dict(),
            "warnings": list(self.warnings),
        }


def _validate_feature_schema(estimator: object, X_test: pd.DataFrame) -> None:
    columns = tuple(str(column) for column in X_test.columns)
    if TARGET_COLUMN in columns:
        raise EvaluationError(
            f"Target leakage detected: '{TARGET_COLUMN}' must not be in X_test."
        )
    if X_test.columns.duplicated().any():
        raise EvaluationError("X_test contains duplicate feature columns.")

    missing = [feature for feature in FEATURE_COLUMNS if feature not in columns]
    unexpected = [column for column in columns if column not in FEATURE_COLUMNS]
    if missing:
        raise EvaluationError(
            "X_test is missing required feature columns: " + ", ".join(missing) + "."
        )
    if unexpected:
        raise EvaluationError(
            "X_test contains unexpected feature columns: "
            + ", ".join(sorted(unexpected))
            + "."
        )
    if columns != FEATURE_COLUMNS:
        raise EvaluationError(
            "X_test feature order is invalid. Expected: "
            + ", ".join(FEATURE_COLUMNS)
            + "."
        )

    fitted_feature_names = getattr(estimator, "feature_names_in_", None)
    if fitted_feature_names is None:
        raise EvaluationError(
            "The fitted estimator does not expose feature_names_in_; feature "
            "compatibility cannot be verified."
        )
    trained_columns = tuple(str(column) for column in fitted_feature_names)
    if trained_columns != columns:
        raise EvaluationError(
            "X_test feature names/order do not match the estimator's fitted features."
        )


def _fitted_class_labels(estimator: object) -> tuple[str, ...]:
    fitted_classes = getattr(estimator, "classes_", None)
    if fitted_classes is None:
        raise EvaluationError("The fitted estimator does not expose known classes_.")
    if any(not isinstance(label, str) or not label.strip() for label in fitted_classes):
        raise EvaluationError("The fitted estimator contains invalid class labels.")

    fitted_class_set = set(fitted_classes)
    unexpected = sorted(fitted_class_set.difference(BASELINE_CROP_CLASSES))
    if unexpected:
        raise EvaluationError(
            "The fitted estimator contains classes outside the baseline vocabulary: "
            + ", ".join(unexpected)
            + "."
        )

    labels = tuple(
        label for label in BASELINE_CROP_CLASSES if label in fitted_class_set
    )
    if len(labels) != len(fitted_classes):
        raise EvaluationError("The fitted estimator contains duplicate class labels.")
    return labels


def _validate_evaluation_inputs(
    estimator: object, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple[str, ...]:
    if not isinstance(X_test, pd.DataFrame):
        raise EvaluationError("X_test must be a Pandas DataFrame.")
    if not isinstance(y_test, pd.Series):
        raise EvaluationError("y_test must be a Pandas Series.")
    if X_test.empty:
        raise EvaluationError("X_test is empty; evaluation requires test rows.")
    if y_test.empty:
        raise EvaluationError("y_test is empty; evaluation requires target labels.")
    if len(X_test) != len(y_test):
        raise EvaluationError(
            "X_test and y_test must contain the same number of rows; "
            f"received {len(X_test)} and {len(y_test)}."
        )
    if not X_test.index.equals(y_test.index):
        raise EvaluationError(
            "X_test and y_test indices must match in the same order."
        )
    if not X_test.index.is_unique:
        raise EvaluationError("Evaluation row indices must be unique.")
    if y_test.name != TARGET_COLUMN:
        raise EvaluationError(f"y_test must be named '{TARGET_COLUMN}'.")

    try:
        check_is_fitted(estimator)
    except (NotFittedError, TypeError) as error:
        raise EvaluationError("The estimator must be fitted before evaluation.") from error

    _validate_feature_schema(estimator, X_test)

    validation_frame = X_test.copy(deep=True)
    validation_frame[TARGET_COLUMN] = y_test.copy(deep=True)
    report = validate_dataframe(validation_frame)
    if not report.is_valid:
        details = "; ".join(report.validation_errors)
        raise EvaluationError(f"Evaluation data validation failed: {details}")

    class_labels = _fitted_class_labels(estimator)
    unknown_targets = sorted(set(y_test).difference(class_labels))
    if unknown_targets:
        raise EvaluationError(
            "y_test contains labels unknown to the fitted estimator: "
            + ", ".join(unknown_targets)
            + "."
        )
    return class_labels


def _model_type(estimator: object) -> str:
    if isinstance(estimator, Pipeline) and estimator.steps:
        return estimator.steps[-1][1].__class__.__name__
    return estimator.__class__.__name__


def _undefined_metric_warnings(
    y_test: pd.Series, predictions: list[str], class_labels: tuple[str, ...]
) -> tuple[str, ...]:
    actual_counts = y_test.value_counts()
    predicted_counts = pd.Series(predictions).value_counts()
    without_actual = [label for label in class_labels if actual_counts.get(label, 0) == 0]
    without_predictions = [
        label for label in class_labels if predicted_counts.get(label, 0) == 0
    ]

    warnings: list[str] = []
    if without_actual:
        warnings.append(
            "zero_division=0 applied to recall/F1 for classes without actual "
            "samples: "
            + ", ".join(without_actual)
            + "."
        )
    if without_predictions:
        warnings.append(
            "zero_division=0 applied to precision/F1 for classes without predicted "
            "samples: "
            + ", ".join(without_predictions)
            + "."
        )
    return tuple(warnings)


def evaluate_model(
    estimator: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> EvaluationResult:
    """Measure a fitted estimator without fitting or modifying it.

    Rows in the confusion matrix are actual classes and columns are predictions.
    Undefined per-class divisions use zero, with affected classes listed in
    ``warnings``.
    """

    class_labels = _validate_evaluation_inputs(estimator, X_test, y_test)
    raw_predictions = estimator.predict(X_test.copy(deep=True))  # type: ignore[attr-defined]
    if len(raw_predictions) != len(y_test):
        raise EvaluationError(
            "The estimator returned a prediction count that does not match y_test."
        )
    if any(
        not isinstance(prediction, str) or not prediction.strip()
        for prediction in raw_predictions
    ):
        raise EvaluationError("The estimator returned invalid prediction labels.")

    predictions = [str(prediction) for prediction in raw_predictions]
    unknown_predictions = sorted(set(predictions).difference(class_labels))
    if unknown_predictions:
        raise EvaluationError(
            "The estimator predicted labels outside its known classes: "
            + ", ".join(unknown_predictions)
            + "."
        )

    per_precision, per_recall, per_f1, per_support = (
        precision_recall_fscore_support(
            y_test,
            predictions,
            labels=class_labels,
            zero_division=0,
        )
    )
    class_report = {
        label: ClassMetrics(
            class_name=label,
            precision=float(per_precision[index]),
            recall=float(per_recall[index]),
            f1=float(per_f1[index]),
            support=int(per_support[index]),
        )
        for index, label in enumerate(class_labels)
    }
    matrix = confusion_matrix(y_test, predictions, labels=class_labels)

    return EvaluationResult(
        accuracy=float(accuracy_score(y_test, predictions)),
        macro_precision=float(
            precision_score(
                y_test,
                predictions,
                labels=class_labels,
                average="macro",
                zero_division=0,
            )
        ),
        macro_recall=float(
            recall_score(
                y_test,
                predictions,
                labels=class_labels,
                average="macro",
                zero_division=0,
            )
        ),
        macro_f1=float(
            f1_score(
                y_test,
                predictions,
                labels=class_labels,
                average="macro",
                zero_division=0,
            )
        ),
        test_sample_count=len(y_test),
        class_labels=class_labels,
        class_report=class_report,
        confusion_matrix=tuple(
            tuple(int(value) for value in row) for row in matrix.tolist()
        ),
        evaluation_metadata=EvaluationMetadata(
            model_type=_model_type(estimator),
            feature_names=FEATURE_COLUMNS,
            class_count=len(class_labels),
            sklearn_version=sklearn.__version__,
            zero_division_policy=0,
        ),
        warnings=_undefined_metric_warnings(y_test, predictions, class_labels),
    )
