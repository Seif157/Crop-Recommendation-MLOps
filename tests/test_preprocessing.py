"""Tests for validated feature extraction and reproducible data splitting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from crop_recommendation.data.preprocessing import (
    DataPreparationError,
    SplitConfig,
    extract_features_and_target,
    prepare_data_split,
    split_dataframe,
)
from crop_recommendation.data.validation import (
    BASELINE_CROP_CLASSES,
    EXPECTED_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


@pytest.fixture
def synthetic_dataframe() -> pd.DataFrame:
    """Return 40 unique rows across four valid crop classes."""

    records: list[dict[str, object]] = []
    for class_index, crop in enumerate(BASELINE_CROP_CLASSES[:4]):
        for sample_index in range(10):
            row_id = class_index * 10 + sample_index
            records.append(
                {
                    "N": row_id,
                    "P": row_id + 5,
                    "K": row_id + 10,
                    "temperature": 20.0 + row_id / 100,
                    "humidity": 50.0 + row_id / 100,
                    "ph": 6.0 + row_id / 1_000,
                    "rainfall": 100.0 + row_id,
                    "label": crop,
                }
            )
    return pd.DataFrame(records, columns=EXPECTED_COLUMNS)


def test_successful_split_of_valid_data(synthetic_dataframe: pd.DataFrame) -> None:
    result = split_dataframe(synthetic_dataframe)

    assert result.source_rows == 40
    assert result.train_rows == 32
    assert result.test_rows == 8


def test_feature_columns_have_correct_number_and_order(
    synthetic_dataframe: pd.DataFrame,
) -> None:
    result = split_dataframe(synthetic_dataframe)

    assert result.feature_names == FEATURE_COLUMNS
    assert tuple(result.X_train.columns) == FEATURE_COLUMNS
    assert tuple(result.X_test.columns) == FEATURE_COLUMNS
    assert result.X_train.shape[1] == 7


def test_target_is_separated_from_features(synthetic_dataframe: pd.DataFrame) -> None:
    X, y = extract_features_and_target(synthetic_dataframe)

    assert TARGET_COLUMN not in X.columns
    assert y.name == TARGET_COLUMN
    assert_series_equal(y, synthetic_dataframe[TARGET_COLUMN])


def test_default_split_uses_eighty_twenty_proportions(
    synthetic_dataframe: pd.DataFrame,
) -> None:
    result = split_dataframe(synthetic_dataframe)

    assert result.train_rows / result.source_rows == pytest.approx(0.8)
    assert result.test_rows / result.source_rows == pytest.approx(0.2)
    assert result.config == SplitConfig(test_size=0.2, random_state=42)


def test_stratification_preserves_class_distribution(
    synthetic_dataframe: pd.DataFrame,
) -> None:
    result = split_dataframe(synthetic_dataframe)

    assert set(result.train_class_distribution.values()) == {8}
    assert set(result.test_class_distribution.values()) == {2}


def test_same_seed_produces_identical_partitions(
    synthetic_dataframe: pd.DataFrame,
) -> None:
    first = split_dataframe(synthetic_dataframe, SplitConfig(random_state=42))
    second = split_dataframe(synthetic_dataframe, SplitConfig(random_state=42))

    assert_frame_equal(first.X_train, second.X_train)
    assert_frame_equal(first.X_test, second.X_test)
    assert_series_equal(first.y_train, second.y_train)
    assert_series_equal(first.y_test, second.y_test)


def test_different_seed_changes_partition(synthetic_dataframe: pd.DataFrame) -> None:
    first = split_dataframe(synthetic_dataframe, SplitConfig(random_state=42))
    second = split_dataframe(synthetic_dataframe, SplitConfig(random_state=7))

    assert set(first.X_test.index) != set(second.X_test.index)


def test_partitions_share_no_source_row_identities(
    synthetic_dataframe: pd.DataFrame,
) -> None:
    result = split_dataframe(synthetic_dataframe)

    assert set(result.X_train.index).isdisjoint(result.X_test.index)
    assert set(result.y_train.index).isdisjoint(result.y_test.index)


def test_split_conserves_every_input_row(synthetic_dataframe: pd.DataFrame) -> None:
    result = split_dataframe(synthetic_dataframe)

    partition_indices = set(result.X_train.index) | set(result.X_test.index)
    assert partition_indices == set(synthetic_dataframe.index)
    assert result.train_rows + result.test_rows == len(synthetic_dataframe)


@pytest.mark.parametrize(
    "test_size",
    [0, 1, -0.1, 1.1, True, "0.2"],
    ids=["zero", "one", "negative", "above-one", "boolean", "string"],
)
def test_invalid_test_sizes_are_rejected(test_size: object) -> None:
    with pytest.raises((TypeError, ValueError), match="test_size"):
        SplitConfig(test_size=test_size)  # type: ignore[arg-type]


def test_class_with_one_sample_cannot_be_stratified(
    synthetic_dataframe: pd.DataFrame,
) -> None:
    dataframe = synthetic_dataframe[
        (synthetic_dataframe[TARGET_COLUMN] != BASELINE_CROP_CLASSES[0])
        | (synthetic_dataframe.index == 0)
    ]

    with pytest.raises(DataPreparationError, match="at least 2 samples per class"):
        split_dataframe(dataframe)


def test_test_partition_must_represent_every_class(
    synthetic_dataframe: pd.DataFrame,
) -> None:
    with pytest.raises(DataPreparationError, match="at least one test row per class"):
        split_dataframe(synthetic_dataframe, SplitConfig(test_size=0.05))


def test_training_partition_must_represent_every_class(
    synthetic_dataframe: pd.DataFrame,
) -> None:
    with pytest.raises(DataPreparationError, match="at least one training row per class"):
        split_dataframe(synthetic_dataframe, SplitConfig(test_size=0.95))


def test_invalid_input_dataset_is_rejected(
    synthetic_dataframe: pd.DataFrame,
) -> None:
    dataframe = synthetic_dataframe.copy()
    dataframe.loc[0, "N"] = -1

    with pytest.raises(DataPreparationError, match="validation failed"):
        split_dataframe(dataframe)


def test_non_unique_source_indices_are_rejected(
    synthetic_dataframe: pd.DataFrame,
) -> None:
    dataframe = synthetic_dataframe.copy()
    dataframe.index = [index // 2 for index in range(len(dataframe))]

    with pytest.raises(DataPreparationError, match="index values must be unique"):
        split_dataframe(dataframe)


def test_original_dataframe_is_not_mutated(
    synthetic_dataframe: pd.DataFrame,
) -> None:
    original = synthetic_dataframe.copy(deep=True)

    split_dataframe(synthetic_dataframe)

    assert_frame_equal(synthetic_dataframe, original)


@pytest.mark.integration
def test_real_dataset_split_has_expected_sizes_and_distributions() -> None:
    result = prepare_data_split()

    assert result.source_rows == 2_200
    assert result.train_rows == 1_760
    assert result.test_rows == 440
    assert result.train_class_distribution == {
        crop: 80 for crop in BASELINE_CROP_CLASSES
    }
    assert result.test_class_distribution == {
        crop: 20 for crop in BASELINE_CROP_CLASSES
    }


@pytest.mark.integration
def test_preparation_is_independent_of_current_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = prepare_data_split()

    assert result.source_rows == 2_200
    assert result.config.random_state == 42
