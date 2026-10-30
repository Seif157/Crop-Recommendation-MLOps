"""Validated feature extraction and reproducible train/test splitting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from crop_recommendation.data.validation import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    default_dataset_path,
    validate_dataframe,
    validate_dataset,
)


class DataPreparationError(ValueError):
    """Raised when validated data cannot be prepared or split safely."""


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for a deterministic stratified holdout split."""

    test_size: float = 0.2
    random_state: int = 42

    def __post_init__(self) -> None:
        if isinstance(self.test_size, bool) or not isinstance(self.test_size, Real):
            raise TypeError("test_size must be a numeric proportion between 0 and 1.")

        normalized_test_size = float(self.test_size)
        if not 0 < normalized_test_size < 1:
            raise ValueError("test_size must be greater than 0 and less than 1.")

        if isinstance(self.random_state, bool) or not isinstance(
            self.random_state, Integral
        ):
            raise TypeError("random_state must be an integer.")

        normalized_random_state = int(self.random_state)
        if not 0 <= normalized_random_state <= 2**32 - 1:
            raise ValueError("random_state must be between 0 and 2**32 - 1.")

        object.__setattr__(self, "test_size", normalized_test_size)
        object.__setattr__(self, "random_state", normalized_random_state)


@dataclass(frozen=True)
class DataSplit:
    """Train/test partitions and the metadata needed to audit their creation.

    Row indices retain source-row identity. No scaling, encoding, feature
    engineering, or other fitted transformation is performed in this step.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: tuple[str, ...]
    config: SplitConfig
    source_rows: int
    train_rows: int
    test_rows: int
    train_class_distribution: dict[str, int]
    test_class_distribution: dict[str, int]


def _validation_failure_message(errors: list[str]) -> str:
    details = "; ".join(errors) if errors else "unknown validation failure"
    return f"Dataset validation failed; preparation was stopped: {details}"


def load_validated_dataframe(dataset_path: str | Path | None = None) -> pd.DataFrame:
    """Load the CSV only after the existing validation contract succeeds.

    The default path comes from the validation module and is independent of the
    process's current working directory. The source CSV is read but never changed.
    """

    path = Path(dataset_path) if dataset_path is not None else default_dataset_path()
    report = validate_dataset(path)
    if not report.is_valid:
        raise DataPreparationError(_validation_failure_message(report.validation_errors))

    return pd.read_csv(Path(report.dataset_path))


def _require_valid_dataframe(dataframe: pd.DataFrame) -> None:
    report = validate_dataframe(dataframe)
    if not report.is_valid:
        raise DataPreparationError(_validation_failure_message(report.validation_errors))
    if not dataframe.index.is_unique:
        raise DataPreparationError(
            "DataFrame index values must be unique so source rows can be tracked "
            "across train and test partitions."
        )


def _extract_validated_features_and_target(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    features = dataframe.loc[:, list(FEATURE_COLUMNS)].copy(deep=True)
    target = dataframe.loc[:, TARGET_COLUMN].copy(deep=True)
    target.name = TARGET_COLUMN
    return features, target


def extract_features_and_target(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Validate and separate the ordered feature matrix ``X`` and target ``y``."""

    _require_valid_dataframe(dataframe)
    return _extract_validated_features_and_target(dataframe)


def _class_distribution(target: pd.Series) -> dict[str, int]:
    return {
        str(label): int(count)
        for label, count in target.value_counts().sort_index().items()
    }


def _validate_stratification_feasibility(
    target: pd.Series, config: SplitConfig
) -> None:
    class_counts = target.value_counts()
    class_count = int(class_counts.size)
    sample_count = int(target.size)

    classes_with_too_few_samples = class_counts[class_counts < 2]
    if not classes_with_too_few_samples.empty:
        details = ", ".join(
            f"{label}={int(count)}"
            for label, count in classes_with_too_few_samples.sort_index().items()
        )
        raise DataPreparationError(
            "Stratified splitting requires at least 2 samples per class; "
            f"insufficient classes: {details}."
        )

    test_rows = math.ceil(sample_count * config.test_size)
    train_rows = sample_count - test_rows
    if test_rows < class_count:
        raise DataPreparationError(
            "The configured test_size produces "
            f"{test_rows} test rows for {class_count} classes; at least one test "
            "row per class is required for stratification."
        )
    if train_rows < class_count:
        raise DataPreparationError(
            "The configured test_size leaves "
            f"{train_rows} training rows for {class_count} classes; at least one "
            "training row per class is required for stratification."
        )


def _verify_partition_integrity(
    source_index: pd.Index, X_train: pd.DataFrame, X_test: pd.DataFrame
) -> None:
    train_index = set(X_train.index)
    test_index = set(X_test.index)
    source_identity = set(source_index)

    if train_index.intersection(test_index):
        raise RuntimeError("Train and test partitions share source-row identities.")
    if train_index.union(test_index) != source_identity:
        raise RuntimeError("The train/test split did not assign every source row once.")
    if len(X_train) + len(X_test) != len(source_index):
        raise RuntimeError("The train/test split changed the total number of rows.")


def _split_validated_dataframe(
    dataframe: pd.DataFrame, config: SplitConfig
) -> DataSplit:
    X, y = _extract_validated_features_and_target(dataframe)
    _validate_stratification_feasibility(y, config)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=y,
        )
    except ValueError as error:
        raise DataPreparationError(f"Stratified train/test split failed: {error}") from error

    _verify_partition_integrity(dataframe.index, X_train, X_test)
    if TARGET_COLUMN in X_train.columns or TARGET_COLUMN in X_test.columns:
        raise RuntimeError("Target leakage detected: label is present in model features.")

    return DataSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=FEATURE_COLUMNS,
        config=config,
        source_rows=len(dataframe),
        train_rows=len(X_train),
        test_rows=len(X_test),
        train_class_distribution=_class_distribution(y_train),
        test_class_distribution=_class_distribution(y_test),
    )


def split_dataframe(
    dataframe: pd.DataFrame, config: SplitConfig | None = None
) -> DataSplit:
    """Validate and split an in-memory dataset without mutating it."""

    effective_config = config if config is not None else SplitConfig()
    _require_valid_dataframe(dataframe)
    return _split_validated_dataframe(dataframe, effective_config)


def prepare_data_split(
    dataset_path: str | Path | None = None,
    config: SplitConfig | None = None,
) -> DataSplit:
    """Load, validate, and stratify a CSV into reproducible train/test data."""

    effective_config = config if config is not None else SplitConfig()
    dataframe = load_validated_dataframe(dataset_path)
    _require_valid_dataframe(dataframe)
    return _split_validated_dataframe(dataframe, effective_config)
