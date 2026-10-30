"""Tests for the crop dataset contract and validation report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from crop_recommendation.data.validation import (
    BASELINE_CROP_CLASSES,
    EXPECTED_COLUMNS,
    default_dataset_path,
    main,
    validate_dataframe,
    validate_dataset,
)


@pytest.fixture
def valid_dataframe() -> pd.DataFrame:
    """Return a small, balanced frame containing the baseline vocabulary."""

    records = []
    for index, crop in enumerate(BASELINE_CROP_CLASSES):
        records.append(
            {
                "N": index,
                "P": index + 5,
                "K": index + 10,
                "temperature": 20.0 + index / 10,
                "humidity": 50.0 + index / 10,
                "ph": 6.5,
                "rainfall": 100.0 + index,
                "label": crop,
            }
        )
    return pd.DataFrame(records, columns=EXPECTED_COLUMNS)


def assert_error_contains(report: object, expected: str) -> None:
    errors = getattr(report, "validation_errors")
    assert any(expected in error for error in errors), errors


def test_valid_dataset(valid_dataframe: pd.DataFrame) -> None:
    report = validate_dataframe(valid_dataframe)

    assert report.is_valid
    assert report.validation_status == "valid"
    assert report.validation_errors == []
    assert report.validation_warnings == []


def test_missing_required_column(valid_dataframe: pd.DataFrame) -> None:
    report = validate_dataframe(valid_dataframe.drop(columns="rainfall"))

    assert not report.is_valid
    assert_error_contains(report, "Missing required columns: rainfall")


def test_unexpected_column(valid_dataframe: pd.DataFrame) -> None:
    dataframe = valid_dataframe.assign(unknown_feature=1)

    report = validate_dataframe(dataframe)

    assert not report.is_valid
    assert_error_contains(report, "Unexpected columns detected: unknown_feature")


def test_duplicate_column_name(valid_dataframe: pd.DataFrame) -> None:
    dataframe = valid_dataframe.copy()
    dataframe.columns = [*EXPECTED_COLUMNS[:-1], "N"]

    report = validate_dataframe(dataframe)

    assert not report.is_valid
    assert_error_contains(report, "Duplicate column names detected: N")


def test_incorrect_column_order(valid_dataframe: pd.DataFrame) -> None:
    dataframe = valid_dataframe[["P", "N", *EXPECTED_COLUMNS[2:]]]

    report = validate_dataframe(dataframe)

    assert not report.is_valid
    assert_error_contains(report, "Column order is invalid")


def test_non_numeric_feature_value(valid_dataframe: pd.DataFrame) -> None:
    dataframe = valid_dataframe.copy()
    dataframe["temperature"] = dataframe["temperature"].astype(object)
    dataframe.loc[0, "temperature"] = "not-a-number"

    report = validate_dataframe(dataframe)

    assert not report.is_valid
    assert_error_contains(report, "must have a numeric dtype")
    assert_error_contains(report, "contains 1 non-numeric value")


def test_missing_feature_value(valid_dataframe: pd.DataFrame) -> None:
    dataframe = valid_dataframe.copy()
    dataframe.loc[0, "N"] = float("nan")

    report = validate_dataframe(dataframe)

    assert not report.is_valid
    assert report.missing_values["N"] == 1
    assert_error_contains(report, "Feature 'N' contains 1 missing value")


def test_infinite_numerical_value(valid_dataframe: pd.DataFrame) -> None:
    dataframe = valid_dataframe.copy()
    dataframe.loc[0, "temperature"] = float("inf")

    report = validate_dataframe(dataframe)

    assert not report.is_valid
    assert report.infinite_values["temperature"] == 1
    assert_error_contains(report, "contains 1 infinite value")


def test_duplicate_record(valid_dataframe: pd.DataFrame) -> None:
    dataframe = pd.concat([valid_dataframe, valid_dataframe.iloc[[0]]], ignore_index=True)

    report = validate_dataframe(dataframe)

    assert not report.is_valid
    assert report.duplicate_rows == 1
    assert_error_contains(report, "contains 1 duplicate row")


@pytest.mark.parametrize("feature", ["N", "P", "K"])
def test_negative_nutrient_measurement(
    valid_dataframe: pd.DataFrame, feature: str
) -> None:
    dataframe = valid_dataframe.copy()
    dataframe.loc[0, feature] = -1

    report = validate_dataframe(dataframe)

    assert not report.is_valid
    assert_error_contains(report, f"Feature '{feature}' contains 1 negative value")


@pytest.mark.parametrize("humidity", [100.1, -0.1], ids=["above-100", "below-0"])
def test_invalid_humidity(valid_dataframe: pd.DataFrame, humidity: float) -> None:
    dataframe = valid_dataframe.copy()
    dataframe.loc[0, "humidity"] = humidity

    report = validate_dataframe(dataframe)

    assert not report.is_valid
    assert_error_contains(report, "Feature 'humidity' contains 1 value(s) outside [0, 100]")


@pytest.mark.parametrize("ph_value", [14.1, -0.1], ids=["above-14", "below-0"])
def test_invalid_ph(valid_dataframe: pd.DataFrame, ph_value: float) -> None:
    dataframe = valid_dataframe.copy()
    dataframe.loc[0, "ph"] = ph_value

    report = validate_dataframe(dataframe)

    assert not report.is_valid
    assert_error_contains(report, "Feature 'ph' contains 1 value(s) outside [0, 14]")


def test_negative_rainfall(valid_dataframe: pd.DataFrame) -> None:
    dataframe = valid_dataframe.copy()
    dataframe.loc[0, "rainfall"] = -0.1

    report = validate_dataframe(dataframe)

    assert not report.is_valid
    assert_error_contains(report, "Feature 'rainfall' contains 1 negative value")


def test_empty_dataset() -> None:
    dataframe = pd.DataFrame(columns=EXPECTED_COLUMNS)

    report = validate_dataframe(dataframe)

    assert not report.is_valid
    assert report.total_rows == 0
    assert_error_contains(report, "Dataset is empty")


def test_empty_crop_label(valid_dataframe: pd.DataFrame) -> None:
    dataframe = valid_dataframe.copy()
    dataframe.loc[0, "label"] = "   "

    report = validate_dataframe(dataframe)

    assert not report.is_valid
    assert_error_contains(report, "Target labels contain 1 empty string value")


def test_missing_crop_label(valid_dataframe: pd.DataFrame) -> None:
    dataframe = valid_dataframe.copy()
    dataframe.loc[0, "label"] = None

    report = validate_dataframe(dataframe)

    assert not report.is_valid
    assert_error_contains(report, "Target labels contain 1 missing value")


def test_unexpected_crop_label(valid_dataframe: pd.DataFrame) -> None:
    dataframe = valid_dataframe.copy()
    dataframe.loc[0, "label"] = "wheat"

    report = validate_dataframe(dataframe)

    assert not report.is_valid
    assert_error_contains(report, "Unexpected crop labels detected: wheat")


def test_correct_target_class_distribution(valid_dataframe: pd.DataFrame) -> None:
    report = validate_dataframe(valid_dataframe)

    assert report.class_count == 22
    assert report.class_distribution == {
        crop: 1 for crop in BASELINE_CROP_CLASSES
    }


def test_class_imbalance_is_a_warning(valid_dataframe: pd.DataFrame) -> None:
    extra_row = valid_dataframe.iloc[[0]].copy()
    extra_row.loc[:, "N"] = 99
    dataframe = pd.concat([valid_dataframe, extra_row], ignore_index=True)

    report = validate_dataframe(dataframe)

    assert report.is_valid
    assert any("imbalanced" in warning for warning in report.validation_warnings)


def test_cli_returns_nonzero_for_invalid_data(
    tmp_path: Path, valid_dataframe: pd.DataFrame, capsys: pytest.CaptureFixture[str]
) -> None:
    invalid_path = tmp_path / "invalid.csv"
    valid_dataframe.drop(columns="rainfall").to_csv(invalid_path, index=False)

    exit_code = main([str(invalid_path)])

    assert exit_code == 1
    assert '"validation_status": "invalid"' in capsys.readouterr().out


@pytest.mark.integration
def test_real_dataset_is_valid() -> None:
    report = validate_dataset(default_dataset_path())

    assert report.is_valid, report.validation_errors
    assert report.total_rows == 2_200
    assert report.total_columns == 8
    assert report.class_count == 22
    assert set(report.class_distribution.values()) == {100}


@pytest.mark.integration
def test_validation_is_independent_of_current_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    report = validate_dataset()

    assert report.is_valid, report.validation_errors
    assert Path(report.dataset_path) == default_dataset_path()
