"""Validate crop recommendation data without modifying the source records."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

FEATURE_COLUMNS: tuple[str, ...] = (
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
)
TARGET_COLUMN = "label"
EXPECTED_COLUMNS: tuple[str, ...] = (*FEATURE_COLUMNS, TARGET_COLUMN)

# Verified from data/raw/Crop_recommendation.csv. The contract controls allowed
# labels; it does not require every dataset to contain 100 rows per class.
BASELINE_CROP_CLASSES: tuple[str, ...] = (
    "apple",
    "banana",
    "blackgram",
    "chickpea",
    "coconut",
    "coffee",
    "cotton",
    "grapes",
    "jute",
    "kidneybeans",
    "lentil",
    "maize",
    "mango",
    "mothbeans",
    "mungbean",
    "muskmelon",
    "orange",
    "papaya",
    "pigeonpeas",
    "pomegranate",
    "rice",
    "watermelon",
)


@dataclass(frozen=True)
class ValidationReport:
    """Structured outcome of validating one crop recommendation dataset."""

    dataset_path: str
    total_rows: int
    total_columns: int
    feature_columns: list[str]
    target_column: str
    missing_values: dict[str, int]
    infinite_values: dict[str, int]
    duplicate_rows: int
    invalid_rows: int
    class_count: int
    class_distribution: dict[str, int]
    numeric_summary: dict[str, dict[str, float | None]]
    validation_status: str
    validation_errors: list[str]
    validation_warnings: list[str]

    @property
    def is_valid(self) -> bool:
        """Return whether the dataset satisfies every required contract rule."""

        return self.validation_status == "valid"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the report."""

        return asdict(self)


def repository_root() -> Path:
    """Resolve the repository root from this module, never from the caller's CWD."""

    return Path(__file__).resolve().parents[3]


def default_dataset_path() -> Path:
    """Return the repository's canonical raw crop dataset path."""

    return repository_root() / "data" / "raw" / "Crop_recommendation.csv"


def _duplicate_names(columns: Sequence[str]) -> list[str]:
    return sorted(name for name, count in Counter(columns).items() if count > 1)


def _missing_value_counts(dataframe: pd.DataFrame) -> dict[str, int]:
    """Count missing values while preserving duplicate column visibility."""

    occurrences: Counter[str] = Counter()
    counts: dict[str, int] = {}
    for position, column in enumerate(dataframe.columns):
        name = str(column)
        occurrences[name] += 1
        key = name if occurrences[name] == 1 else f"{name}#{occurrences[name]}"
        counts[key] = int(dataframe.iloc[:, position].isna().sum())
    return counts


def _finite_mask(series: pd.Series) -> pd.Series:
    """Return a boolean mask for finite values in an already numeric Series."""

    return series.map(lambda value: bool(pd.notna(value) and math.isfinite(value)))


def _summary_for(series: pd.Series) -> dict[str, float | None]:
    finite_values = series[_finite_mask(series)].astype(float)
    if finite_values.empty:
        return {"min": None, "max": None, "mean": None, "std": None}

    standard_deviation = finite_values.std()
    return {
        "min": float(finite_values.min()),
        "max": float(finite_values.max()),
        "mean": float(finite_values.mean()),
        "std": None if pd.isna(standard_deviation) else float(standard_deviation),
    }


def _unreadable_report(dataset_path: Path, message: str) -> ValidationReport:
    return ValidationReport(
        dataset_path=str(dataset_path),
        total_rows=0,
        total_columns=0,
        feature_columns=list(FEATURE_COLUMNS),
        target_column=TARGET_COLUMN,
        missing_values={},
        infinite_values={column: 0 for column in FEATURE_COLUMNS},
        duplicate_rows=0,
        invalid_rows=0,
        class_count=0,
        class_distribution={},
        numeric_summary={},
        validation_status="invalid",
        validation_errors=[message],
        validation_warnings=[],
    )


def validate_dataframe(
    dataframe: pd.DataFrame,
    *,
    dataset_path: str | Path = "<in-memory>",
    source_columns: Sequence[str] | None = None,
) -> ValidationReport:
    """Validate a DataFrame and return every detected error and warning.

    The function never renames, reorders, drops, fills, or coerces source data.
    ``source_columns`` preserves an original CSV header because Pandas may mangle
    duplicate names while parsing.
    """

    reported_columns = source_columns if source_columns is not None else dataframe.columns
    columns = [str(column) for column in reported_columns]
    errors: list[str] = []
    warnings: list[str] = []
    invalid_row_mask = pd.Series(False, index=dataframe.index, dtype=bool)

    duplicate_columns = _duplicate_names(columns)
    if duplicate_columns:
        errors.append(
            "Duplicate column names detected: " + ", ".join(duplicate_columns) + "."
        )

    missing_columns = [column for column in EXPECTED_COLUMNS if column not in columns]
    unexpected_columns = [column for column in columns if column not in EXPECTED_COLUMNS]
    if missing_columns:
        errors.append("Missing required columns: " + ", ".join(missing_columns) + ".")
    if unexpected_columns:
        errors.append(
            "Unexpected columns detected: "
            + ", ".join(sorted(set(unexpected_columns)))
            + "."
        )
    if not missing_columns and not unexpected_columns and not duplicate_columns:
        if tuple(columns) != EXPECTED_COLUMNS:
            errors.append(
                "Column order is invalid. Expected: "
                + ", ".join(EXPECTED_COLUMNS)
                + "."
            )

    if dataframe.empty:
        errors.append("Dataset is empty.")

    missing_values = _missing_value_counts(dataframe)
    infinite_values = {column: 0 for column in FEATURE_COLUMNS}
    numeric_summary: dict[str, dict[str, float | None]] = {}
    dataframe_column_counts = Counter(str(column) for column in dataframe.columns)

    for feature in FEATURE_COLUMNS:
        if dataframe_column_counts[feature] != 1:
            continue

        series = dataframe[feature]
        missing_mask = series.isna()
        missing_count = int(missing_mask.sum())
        if missing_count:
            errors.append(f"Feature '{feature}' contains {missing_count} missing value(s).")
            invalid_row_mask |= missing_mask

        if not is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype):
            non_missing = series[~missing_mask]
            parsed = pd.to_numeric(non_missing, errors="coerce")
            invalid_numeric_count = int(parsed.isna().sum())
            errors.append(
                f"Feature '{feature}' must have a numeric dtype; found {series.dtype}."
            )
            if invalid_numeric_count:
                errors.append(
                    f"Feature '{feature}' contains {invalid_numeric_count} "
                    "non-numeric value(s)."
                )
                invalid_indices = parsed[parsed.isna()].index
                invalid_row_mask.loc[invalid_indices] = True
            else:
                invalid_row_mask |= ~missing_mask
            continue

        finite_mask = _finite_mask(series)
        infinite_mask = series.notna() & ~finite_mask
        infinite_count = int(infinite_mask.sum())
        infinite_values[feature] = infinite_count
        if infinite_count:
            errors.append(
                f"Feature '{feature}' contains {infinite_count} infinite value(s)."
            )
            invalid_row_mask |= infinite_mask

        numeric_summary[feature] = _summary_for(series)
        comparable_mask = finite_mask & ~missing_mask

        if feature in {"N", "P", "K", "rainfall"}:
            invalid_range_mask = comparable_mask & (series < 0)
            if invalid_range_mask.any():
                count = int(invalid_range_mask.sum())
                errors.append(
                    f"Feature '{feature}' contains {count} negative value(s)."
                )
                invalid_row_mask |= invalid_range_mask
        elif feature == "humidity":
            invalid_range_mask = comparable_mask & ~series.between(0, 100)
            if invalid_range_mask.any():
                count = int(invalid_range_mask.sum())
                errors.append(
                    f"Feature 'humidity' contains {count} value(s) outside [0, 100]."
                )
                invalid_row_mask |= invalid_range_mask
        elif feature == "ph":
            invalid_range_mask = comparable_mask & ~series.between(0, 14)
            if invalid_range_mask.any():
                count = int(invalid_range_mask.sum())
                errors.append(f"Feature 'ph' contains {count} value(s) outside [0, 14].")
                invalid_row_mask |= invalid_range_mask

    class_distribution: dict[str, int] = {}
    if dataframe_column_counts[TARGET_COLUMN] == 1:
        labels = dataframe[TARGET_COLUMN]
        missing_label_mask = labels.isna()
        missing_label_count = int(missing_label_mask.sum())
        if missing_label_count:
            errors.append(f"Target labels contain {missing_label_count} missing value(s).")
            invalid_row_mask |= missing_label_mask

        non_string_mask = labels.notna() & ~labels.map(lambda value: isinstance(value, str))
        if non_string_mask.any():
            count = int(non_string_mask.sum())
            errors.append(f"Target labels contain {count} non-string value(s).")
            invalid_row_mask |= non_string_mask

        empty_label_mask = labels.map(
            lambda value: isinstance(value, str) and not value.strip()
        )
        if empty_label_mask.any():
            count = int(empty_label_mask.sum())
            errors.append(f"Target labels contain {count} empty string value(s).")
            invalid_row_mask |= empty_label_mask

        valid_label_mask = labels.map(
            lambda value: isinstance(value, str) and bool(value.strip())
        )
        valid_labels = labels[valid_label_mask]
        class_distribution = {
            str(label): int(count)
            for label, count in valid_labels.value_counts().sort_index().items()
        }

        unexpected_classes = sorted(
            set(class_distribution).difference(BASELINE_CROP_CLASSES)
        )
        if unexpected_classes:
            errors.append(
                "Unexpected crop labels detected: "
                + ", ".join(unexpected_classes)
                + "."
            )
            invalid_row_mask |= labels.isin(unexpected_classes)

        missing_classes = sorted(
            set(BASELINE_CROP_CLASSES).difference(class_distribution)
        )
        if missing_classes and not dataframe.empty:
            warnings.append(
                "Baseline crop classes absent from this dataset: "
                + ", ".join(missing_classes)
                + "."
            )

        if len(set(class_distribution.values())) > 1:
            warnings.append("Target classes are imbalanced; review the class distribution.")

    duplicate_rows = int(dataframe.duplicated().sum())
    if duplicate_rows:
        errors.append(f"Dataset contains {duplicate_rows} duplicate row(s).")

    return ValidationReport(
        dataset_path=str(dataset_path),
        total_rows=int(len(dataframe)),
        total_columns=len(columns),
        feature_columns=list(FEATURE_COLUMNS),
        target_column=TARGET_COLUMN,
        missing_values=missing_values,
        infinite_values=infinite_values,
        duplicate_rows=duplicate_rows,
        invalid_rows=int(invalid_row_mask.sum()),
        class_count=len(class_distribution),
        class_distribution=class_distribution,
        numeric_summary=numeric_summary,
        validation_status="valid" if not errors else "invalid",
        validation_errors=errors,
        validation_warnings=warnings,
    )


def validate_dataset(dataset_path: str | Path | None = None) -> ValidationReport:
    """Load and validate a CSV dataset without changing its contents."""

    path = Path(dataset_path) if dataset_path is not None else default_dataset_path()
    path = path.expanduser().resolve()
    if not path.is_file():
        return _unreadable_report(path, f"Dataset file does not exist: {path}")

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            source_columns = next(csv.reader(csv_file), [])
        dataframe = pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        return _unreadable_report(path, f"Unable to read dataset: {error}")

    return validate_dataframe(
        dataframe,
        dataset_path=path,
        source_columns=source_columns,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run dataset validation and return a shell-friendly status code."""

    parser = argparse.ArgumentParser(
        description="Validate the crop recommendation CSV data contract."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=None,
        help="CSV path (defaults to data/raw/Crop_recommendation.csv).",
    )
    arguments = parser.parse_args(argv)
    report = validate_dataset(arguments.dataset)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
