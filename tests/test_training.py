"""Tests for deterministic Random Forest training on training-only inputs."""

from __future__ import annotations

import inspect

import pandas as pd
import pytest
import sklearn
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from crop_recommendation.data.preprocessing import prepare_data_split
from crop_recommendation.data.validation import (
    BASELINE_CROP_CLASSES,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from crop_recommendation.training.train import (
    TrainingConfig,
    TrainingError,
    create_training_pipeline,
    train_model,
)


@pytest.fixture
def training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return 48 unique training rows across four allowed crop classes."""

    records: list[dict[str, object]] = []
    for class_index, crop in enumerate(BASELINE_CROP_CLASSES[:4]):
        for sample_index in range(12):
            row_id = class_index * 12 + sample_index
            records.append(
                {
                    "N": row_id,
                    "P": row_id + 5,
                    "K": row_id + 10,
                    "temperature": 20.0 + row_id / 100,
                    "humidity": 50.0 + row_id / 100,
                    "ph": 6.0 + row_id / 1_000,
                    "rainfall": 100.0 + row_id,
                    TARGET_COLUMN: crop,
                }
            )
    dataframe = pd.DataFrame(records)
    return dataframe.loc[:, list(FEATURE_COLUMNS)], dataframe[TARGET_COLUMN]


def test_valid_random_forest_training(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X_train, y_train = training_data

    result = train_model(X_train, y_train)

    assert isinstance(result.estimator, Pipeline)
    assert isinstance(
        result.estimator.named_steps["classifier"], RandomForestClassifier
    )


def test_default_number_of_trees() -> None:
    pipeline = create_training_pipeline()

    assert pipeline.named_steps["classifier"].n_estimators == 100


def test_default_random_seed() -> None:
    pipeline = create_training_pipeline()

    assert pipeline.named_steps["classifier"].random_state == 42


def test_custom_model_configuration() -> None:
    config = TrainingConfig(n_estimators=25, random_state=7)

    pipeline = create_training_pipeline(config)

    classifier = pipeline.named_steps["classifier"]
    assert classifier.n_estimators == 25
    assert classifier.random_state == 7


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("n_estimators", 0, ValueError),
        ("n_estimators", -1, ValueError),
        ("n_estimators", True, TypeError),
        ("n_estimators", 1.5, TypeError),
        ("random_state", -1, ValueError),
        ("random_state", True, TypeError),
    ],
)
def test_invalid_estimator_configuration(
    field: str, value: object, error_type: type[Exception]
) -> None:
    arguments = {field: value}

    with pytest.raises(error_type):
        TrainingConfig(**arguments)  # type: ignore[arg-type]


def test_empty_training_features(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X_train, y_train = training_data

    with pytest.raises(TrainingError, match="X_train is empty"):
        train_model(X_train.iloc[0:0], y_train.iloc[0:0])


def test_empty_training_labels(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X_train, y_train = training_data

    with pytest.raises(TrainingError, match="y_train is empty"):
        train_model(X_train, y_train.iloc[0:0])


def test_mismatched_feature_and_target_rows(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X_train, y_train = training_data

    with pytest.raises(TrainingError, match="same number of rows"):
        train_model(X_train, y_train.iloc[:-1])


def test_misaligned_feature_and_target_indices(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X_train, y_train = training_data
    misaligned_target = y_train.copy()
    misaligned_target.index = range(1, len(misaligned_target) + 1)

    with pytest.raises(TrainingError, match="indices must match"):
        train_model(X_train, misaligned_target)


def test_missing_feature_column(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X_train, y_train = training_data

    with pytest.raises(TrainingError, match="missing required feature columns"):
        train_model(X_train.drop(columns="rainfall"), y_train)


def test_incorrect_feature_order(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X_train, y_train = training_data
    reordered = X_train[["P", "N", *FEATURE_COLUMNS[2:]]]

    with pytest.raises(TrainingError, match="feature order is invalid"):
        train_model(reordered, y_train)


def test_target_column_is_rejected_inside_features(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X_train, y_train = training_data
    leaking_features = X_train.assign(label=y_train)

    with pytest.raises(TrainingError, match="Target leakage detected"):
        train_model(leaking_features, y_train)


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), -1.0])
def test_missing_or_invalid_feature_values_are_rejected(
    training_data: tuple[pd.DataFrame, pd.Series], invalid_value: float
) -> None:
    X_train, y_train = training_data
    invalid_features = X_train.copy()
    invalid_features["N"] = invalid_features["N"].astype(float)
    invalid_features.loc[0, "N"] = invalid_value

    with pytest.raises(TrainingError, match="validation failed"):
        train_model(invalid_features, y_train)


def test_non_numeric_feature_value_is_rejected(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X_train, y_train = training_data
    invalid_features = X_train.copy()
    invalid_features["temperature"] = invalid_features["temperature"].astype(object)
    invalid_features.loc[0, "temperature"] = "invalid"

    with pytest.raises(TrainingError, match="validation failed"):
        train_model(invalid_features, y_train)


def test_model_is_fitted_successfully(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X_train, y_train = training_data

    result = train_model(X_train, y_train)

    check_is_fitted(result.estimator.named_steps["classifier"])


def test_fitted_model_can_predict_valid_features(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X_train, y_train = training_data
    result = train_model(X_train, y_train)

    predictions = result.estimator.predict(X_train.iloc[:5])

    assert len(predictions) == 5
    assert set(predictions).issubset(set(y_train))


def test_training_result_metadata(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X_train, y_train = training_data

    result = train_model(X_train, y_train)

    assert result.metadata.model_type == "RandomForestClassifier"
    assert result.metadata.model_configuration == {
        "n_estimators": 100,
        "random_state": 42,
    }
    assert result.metadata.training_rows == 48
    assert result.metadata.feature_names == FEATURE_COLUMNS
    assert result.metadata.feature_count == 7
    assert result.metadata.class_count == 4
    assert set(result.metadata.class_distribution.values()) == {12}
    assert result.metadata.random_seed == 42
    assert result.metadata.sklearn_version == sklearn.__version__


def test_feature_dataframe_is_not_mutated(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X_train, y_train = training_data
    original = X_train.copy(deep=True)

    train_model(X_train, y_train)

    assert_frame_equal(X_train, original)


def test_target_series_is_not_mutated(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X_train, y_train = training_data
    original = y_train.copy(deep=True)

    train_model(X_train, y_train)

    assert_series_equal(y_train, original)


def test_identical_input_and_seed_produce_identical_predictions(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X_train, y_train = training_data
    first = train_model(X_train, y_train, TrainingConfig(random_state=42))
    second = train_model(X_train, y_train, TrainingConfig(random_state=42))
    probe = X_train.iloc[[0, 13, 25, 37]]

    assert first.estimator.predict(probe).tolist() == second.estimator.predict(
        probe
    ).tolist()
    first_classes = first.estimator.named_steps["classifier"].classes_.tolist()
    second_classes = second.estimator.named_steps["classifier"].classes_.tolist()
    assert first_classes == second_classes


def test_training_calls_create_independent_estimators(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X_train, y_train = training_data

    first = train_model(X_train, y_train)
    second = train_model(X_train, y_train)

    assert first.estimator is not second.estimator
    assert first.estimator.named_steps["classifier"] is not second.estimator.named_steps[
        "classifier"
    ]


def test_core_training_function_does_not_require_test_data() -> None:
    parameters = tuple(inspect.signature(train_model).parameters)

    assert parameters == ("X_train", "y_train", "config")
    assert "X_test" not in parameters
    assert "y_test" not in parameters


@pytest.mark.integration
def test_real_training_partition_fits_without_evaluation() -> None:
    split = prepare_data_split()

    result = train_model(split.X_train, split.y_train)

    check_is_fitted(result.estimator.named_steps["classifier"])
    assert result.metadata.training_rows == 1_760
    assert result.metadata.feature_count == 7
    assert result.metadata.class_count == 22
