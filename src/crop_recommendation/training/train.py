"""Deterministic Random Forest training using training-partition inputs only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral

import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from crop_recommendation.data.validation import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_dataframe,
)


class TrainingError(ValueError):
    """Raised when model configuration or training inputs are invalid."""


@dataclass(frozen=True)
class TrainingConfig:
    """Explicit configuration for the initial Random Forest baseline."""

    n_estimators: int = 100
    random_state: int = 42

    def __post_init__(self) -> None:
        if isinstance(self.n_estimators, bool) or not isinstance(
            self.n_estimators, Integral
        ):
            raise TypeError("n_estimators must be an integer.")
        if int(self.n_estimators) <= 0:
            raise ValueError("n_estimators must be greater than 0.")

        if isinstance(self.random_state, bool) or not isinstance(
            self.random_state, Integral
        ):
            raise TypeError("random_state must be an integer.")
        if not 0 <= int(self.random_state) <= 2**32 - 1:
            raise ValueError("random_state must be between 0 and 2**32 - 1.")

        object.__setattr__(self, "n_estimators", int(self.n_estimators))
        object.__setattr__(self, "random_state", int(self.random_state))

    def to_dict(self) -> dict[str, int]:
        """Return the serializable estimator configuration."""

        return asdict(self)


@dataclass(frozen=True)
class TrainingMetadata:
    """Serializable facts describing one fitted training run."""

    model_type: str
    model_configuration: dict[str, int]
    training_rows: int
    feature_names: tuple[str, ...]
    feature_count: int
    class_count: int
    class_distribution: dict[str, int]
    random_seed: int
    sklearn_version: str

    def to_dict(self) -> dict[str, object]:
        """Return metadata suitable for logs or future experiment tracking."""

        return asdict(self)


@dataclass(frozen=True)
class TrainingResult:
    """A fitted estimator kept separate from its serializable metadata."""

    estimator: Pipeline
    metadata: TrainingMetadata


def create_training_pipeline(config: TrainingConfig | None = None) -> Pipeline:
    """Create a fresh classifier-only pipeline for the Random Forest baseline.

    No transformation is needed for the current seven finite numeric features.
    Future fitted preprocessing can be inserted before ``classifier`` and will
    then be fitted only through this training pipeline.
    """

    effective_config = config if config is not None else TrainingConfig()
    classifier = RandomForestClassifier(
        n_estimators=effective_config.n_estimators,
        random_state=effective_config.random_state,
    )
    return Pipeline([("classifier", classifier)])


def _validate_feature_schema(X_train: pd.DataFrame) -> None:
    columns = tuple(str(column) for column in X_train.columns)
    if TARGET_COLUMN in columns:
        raise TrainingError(
            f"Target leakage detected: '{TARGET_COLUMN}' must not be in X_train."
        )
    if X_train.columns.duplicated().any():
        duplicates = sorted(
            {str(column) for column in X_train.columns[X_train.columns.duplicated()]}
        )
        raise TrainingError(
            "X_train contains duplicate feature columns: "
            + ", ".join(duplicates)
            + "."
        )

    missing = [feature for feature in FEATURE_COLUMNS if feature not in columns]
    unexpected = [column for column in columns if column not in FEATURE_COLUMNS]
    if missing:
        raise TrainingError(
            "X_train is missing required feature columns: " + ", ".join(missing) + "."
        )
    if unexpected:
        raise TrainingError(
            "X_train contains unexpected feature columns: "
            + ", ".join(sorted(unexpected))
            + "."
        )
    if columns != FEATURE_COLUMNS:
        raise TrainingError(
            "X_train feature order is invalid. Expected: "
            + ", ".join(FEATURE_COLUMNS)
            + "."
        )


def _validate_training_inputs(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    if not isinstance(X_train, pd.DataFrame):
        raise TrainingError("X_train must be a Pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TrainingError("y_train must be a Pandas Series.")
    if X_train.empty:
        raise TrainingError("X_train is empty; at least one training row is required.")
    if y_train.empty:
        raise TrainingError("y_train is empty; at least one target label is required.")
    if len(X_train) != len(y_train):
        raise TrainingError(
            "X_train and y_train must contain the same number of rows; "
            f"received {len(X_train)} and {len(y_train)}."
        )
    if not X_train.index.equals(y_train.index):
        raise TrainingError(
            "X_train and y_train indices must match in the same order to prevent "
            "feature/label misalignment."
        )
    if not X_train.index.is_unique:
        raise TrainingError("Training row indices must be unique.")
    if y_train.name != TARGET_COLUMN:
        raise TrainingError(f"y_train must be named '{TARGET_COLUMN}'.")

    _validate_feature_schema(X_train)

    validation_frame = X_train.copy(deep=True)
    validation_frame[TARGET_COLUMN] = y_train.copy(deep=True)
    report = validate_dataframe(validation_frame)
    if not report.is_valid:
        details = "; ".join(report.validation_errors)
        raise TrainingError(f"Training data validation failed: {details}")


def _class_distribution(y_train: pd.Series) -> dict[str, int]:
    return {
        str(label): int(count)
        for label, count in y_train.value_counts().sort_index().items()
    }


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Validate and fit a new Random Forest using training inputs only.

    The function has no access to a test partition and does not compute any
    evaluation metric. Input objects are copied before fitting and are not mutated.
    """

    effective_config = config if config is not None else TrainingConfig()
    _validate_training_inputs(X_train, y_train)

    estimator = create_training_pipeline(effective_config)
    try:
        estimator.fit(X_train.copy(deep=True), y_train.copy(deep=True))
    except ValueError as error:
        raise TrainingError(f"Random Forest fitting failed: {error}") from error

    metadata = TrainingMetadata(
        model_type="RandomForestClassifier",
        model_configuration=effective_config.to_dict(),
        training_rows=len(X_train),
        feature_names=FEATURE_COLUMNS,
        feature_count=len(FEATURE_COLUMNS),
        class_count=int(y_train.nunique()),
        class_distribution=_class_distribution(y_train),
        random_seed=effective_config.random_state,
        sklearn_version=sklearn.__version__,
    )
    return TrainingResult(estimator=estimator, metadata=metadata)
