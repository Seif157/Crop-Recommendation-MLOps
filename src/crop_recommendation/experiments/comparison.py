"""Deterministic, training-only comparison of Random Forest candidates."""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from crop_recommendation.data.validation import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_dataframe,
)
from crop_recommendation.training.train import (
    TrainingConfig,
    create_training_pipeline,
)

METRIC_NAMES: tuple[str, ...] = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
)
PRIMARY_METRIC = "macro_f1"
SELECTION_TOLERANCE = 1e-12
STANDARD_DEVIATION_CONVENTION = "population"
EXPERIMENT_PROTOCOL = "training_only_stratified_cv"

SCORERS = {
    "accuracy": make_scorer(accuracy_score),
    "macro_precision": make_scorer(
        precision_score, average="macro", zero_division=0
    ),
    "macro_recall": make_scorer(
        recall_score, average="macro", zero_division=0
    ),
    "macro_f1": make_scorer(f1_score, average="macro", zero_division=0),
}


class ExperimentConfigurationError(ValueError):
    """Raised when a comparison configuration violates the fixed protocol."""


class ComparisonError(ValueError):
    """Raised when training-only comparison inputs or results are invalid."""


@dataclass(frozen=True)
class CandidateConfig:
    """One explicitly named Random Forest candidate."""

    name: str
    n_estimators: int
    random_state: int = 42

    def __post_init__(self) -> None:
        normalized_name = self.name.strip()
        if not normalized_name:
            raise ExperimentConfigurationError("Candidate name must not be empty.")
        try:
            validated = TrainingConfig(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            )
        except (TypeError, ValueError) as error:
            raise ExperimentConfigurationError(str(error)) from error
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "n_estimators", validated.n_estimators)
        object.__setattr__(self, "random_state", validated.random_state)

    def training_config(self) -> TrainingConfig:
        return TrainingConfig(self.n_estimators, self.random_state)

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


def default_candidates() -> tuple[CandidateConfig, ...]:
    """Return the predetermined candidate list, independent of observed scores."""

    return (
        CandidateConfig("RF-50", 50, 42),
        CandidateConfig("RF-100", 100, 42),
        CandidateConfig("RF-200", 200, 42),
    )


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated candidate, fold, scoring, and selection configuration."""

    candidates: tuple[CandidateConfig, ...] = default_candidates()
    cv_folds: int = 5
    cv_random_state: int = 42
    cv_shuffle: bool = True
    primary_metric: str = PRIMARY_METRIC
    tie_tolerance: float = SELECTION_TOLERANCE

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        if not candidates:
            raise ExperimentConfigurationError("At least one candidate is required.")
        if not all(isinstance(candidate, CandidateConfig) for candidate in candidates):
            raise ExperimentConfigurationError(
                "Every candidate must be a CandidateConfig instance."
            )
        names = [candidate.name for candidate in candidates]
        if len(names) != len(set(names)):
            raise ExperimentConfigurationError(
                "Candidate identifiers must be unique."
            )
        if isinstance(self.cv_folds, bool) or not isinstance(self.cv_folds, Integral):
            raise ExperimentConfigurationError("cv_folds must be an integer.")
        if int(self.cv_folds) < 2:
            raise ExperimentConfigurationError("cv_folds must be at least 2.")
        if isinstance(self.cv_random_state, bool) or not isinstance(
            self.cv_random_state, Integral
        ):
            raise ExperimentConfigurationError(
                "cv_random_state must be an integer."
            )
        if not 0 <= int(self.cv_random_state) <= 2**32 - 1:
            raise ExperimentConfigurationError(
                "cv_random_state must be between 0 and 2**32 - 1."
            )
        if self.cv_shuffle is not True:
            raise ExperimentConfigurationError(
                "The controlled protocol requires shuffled folds."
            )
        if self.primary_metric != PRIMARY_METRIC:
            raise ExperimentConfigurationError(
                f"The controlled protocol requires primary metric '{PRIMARY_METRIC}'."
            )
        if isinstance(self.tie_tolerance, bool) or not isinstance(
            self.tie_tolerance, Real
        ):
            raise ExperimentConfigurationError("tie_tolerance must be numeric.")
        if float(self.tie_tolerance) < 0:
            raise ExperimentConfigurationError(
                "tie_tolerance must be greater than or equal to zero."
            )
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "cv_folds", int(self.cv_folds))
        object.__setattr__(self, "cv_random_state", int(self.cv_random_state))
        object.__setattr__(self, "tie_tolerance", float(self.tie_tolerance))

    def to_dict(self) -> dict[str, object]:
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "cross_validation": {
                "folds": self.cv_folds,
                "shuffle": self.cv_shuffle,
                "random_state": self.cv_random_state,
            },
            "scoring": list(METRIC_NAMES),
            "primary_metric": self.primary_metric,
            "tie_tolerance": self.tie_tolerance,
            "standard_deviation": STANDARD_DEVIATION_CONVENTION,
        }


@dataclass(frozen=True)
class DatasetLineage:
    """Content identities connecting MLflow evidence to the DVC dataset."""

    path: str
    sha256: str
    dvc_md5: str
    size: int

    def __post_init__(self) -> None:
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ComparisonError("Dataset SHA-256 identity is invalid.")
        if len(self.dvc_md5) != 32 or any(
            character not in "0123456789abcdef" for character in self.dvc_md5
        ):
            raise ComparisonError("DVC MD5 identity is invalid.")
        if self.size <= 0:
            raise ComparisonError("Dataset size must be greater than zero.")

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


@dataclass(frozen=True)
class FoldAssignment:
    """Positional train/validation identities for one internal CV fold."""

    fold: int
    training_positions: tuple[int, ...]
    validation_positions: tuple[int, ...]
    training_index_sha256: str
    validation_index_sha256: str
    training_class_distribution: dict[str, int]
    validation_class_distribution: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "training_rows": len(self.training_positions),
            "validation_rows": len(self.validation_positions),
            "training_index_sha256": self.training_index_sha256,
            "validation_index_sha256": self.validation_index_sha256,
            "training_class_distribution": self.training_class_distribution,
            "validation_class_distribution": self.validation_class_distribution,
        }


@dataclass(frozen=True)
class FoldMetrics:
    """Unrounded measured scores and identities for one candidate fold."""

    fold: int
    training_rows: int
    validation_rows: int
    training_index_sha256: str
    validation_index_sha256: str
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateResult:
    """Fold-level and aggregate evidence for one candidate."""

    candidate: CandidateConfig
    folds: tuple[FoldMetrics, ...]
    mean_metrics: dict[str, float]
    standard_deviation_metrics: dict[str, float]
    fitted_model_count: int
    python_version: str
    scikit_learn_version: str

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.to_dict(),
            "folds": [fold.to_dict() for fold in self.folds],
            "mean_metrics": self.mean_metrics,
            "standard_deviation_metrics": self.standard_deviation_metrics,
            "fitted_model_count": self.fitted_model_count,
            "python_version": self.python_version,
            "scikit_learn_version": self.scikit_learn_version,
        }


@dataclass(frozen=True)
class ComparisonResult:
    """Stable comparison evidence without runtime-specific MLflow identities."""

    configuration: ExperimentConfig
    dataset: DatasetLineage
    training_partition: dict[str, object]
    candidates: tuple[CandidateResult, ...]
    ranking: tuple[str, ...]
    best_observed_candidate: str
    recommended_candidate: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": EXPERIMENT_PROTOCOL,
            "configuration": self.configuration.to_dict(),
            "dataset": self.dataset.to_dict(),
            "training_partition": self.training_partition,
            "candidate_results": [candidate.to_dict() for candidate in self.candidates],
            "ranking": list(self.ranking),
            "best_observed_candidate": self.best_observed_candidate,
            "recommended_candidate": self.recommended_candidate,
            "warnings": list(self.warnings),
        }


def _index_fingerprint(index: pd.Index) -> str:
    values = [str(value) for value in index.tolist()]
    serialized = json.dumps(values, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def training_partition_fingerprint(X_train: pd.DataFrame, y_train: pd.Series) -> str:
    """Fingerprint ordered feature values, labels, columns, and source indices."""

    combined = X_train.copy(deep=True)
    combined[TARGET_COLUMN] = y_train.copy(deep=True)
    hashed = pd.util.hash_pandas_object(combined, index=True).to_numpy(dtype="uint64")
    digest = hashlib.sha256()
    digest.update(json.dumps(list(combined.columns)).encode("utf-8"))
    digest.update(hashed.tobytes())
    return digest.hexdigest()


def _distribution(target: pd.Series) -> dict[str, int]:
    return {
        str(label): int(count)
        for label, count in target.value_counts().sort_index().items()
    }


def _validate_comparison_inputs(
    X_train: pd.DataFrame, y_train: pd.Series, config: ExperimentConfig
) -> None:
    if not isinstance(X_train, pd.DataFrame):
        raise ComparisonError("X_train must be a Pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise ComparisonError("y_train must be a Pandas Series.")
    if X_train.empty or y_train.empty:
        raise ComparisonError("Training inputs must not be empty.")
    if not X_train.index.equals(y_train.index):
        raise ComparisonError("X_train and y_train indices must match exactly.")
    if not X_train.index.is_unique:
        raise ComparisonError("Training row indices must be unique.")
    if tuple(str(column) for column in X_train.columns) != FEATURE_COLUMNS:
        raise ComparisonError(
            "X_train must contain exactly seven ordered crop features."
        )
    if y_train.name != TARGET_COLUMN:
        raise ComparisonError(f"y_train must be named '{TARGET_COLUMN}'.")
    validation_frame = X_train.copy(deep=True)
    validation_frame[TARGET_COLUMN] = y_train.copy(deep=True)
    report = validate_dataframe(validation_frame)
    if not report.is_valid:
        raise ComparisonError(
            "Training data validation failed: "
            + "; ".join(report.validation_errors)
        )
    insufficient = y_train.value_counts()[lambda values: values < config.cv_folds]
    if not insufficient.empty:
        details = ", ".join(
            f"{label}={int(count)}" for label, count in insufficient.items()
        )
        raise ComparisonError(
            "Every class needs at least one row in every validation fold; "
            f"insufficient classes: {details}."
        )


def build_cv_folds(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ExperimentConfig | None = None,
) -> tuple[FoldAssignment, ...]:
    """Build and verify deterministic internal folds from training data only."""

    effective_config = config if config is not None else ExperimentConfig()
    _validate_comparison_inputs(X_train, y_train, effective_config)
    splitter = StratifiedKFold(
        n_splits=effective_config.cv_folds,
        shuffle=effective_config.cv_shuffle,
        random_state=effective_config.cv_random_state,
    )
    assignments: list[FoldAssignment] = []
    validation_coverage: list[int] = []
    for fold_number, (training_positions, validation_positions) in enumerate(
        splitter.split(X_train, y_train), start=1
    ):
        training_set = set(int(value) for value in training_positions)
        validation_set = set(int(value) for value in validation_positions)
        if training_set.intersection(validation_set):
            raise ComparisonError("A CV fold has overlapping train/validation rows.")
        if training_set.union(validation_set) != set(range(len(X_train))):
            raise ComparisonError("A CV fold does not cover the training partition.")
        validation_coverage.extend(validation_set)
        train_index = X_train.iloc[training_positions].index
        validation_index = X_train.iloc[validation_positions].index
        assignments.append(
            FoldAssignment(
                fold=fold_number,
                training_positions=tuple(int(value) for value in training_positions),
                validation_positions=tuple(
                    int(value) for value in validation_positions
                ),
                training_index_sha256=_index_fingerprint(train_index),
                validation_index_sha256=_index_fingerprint(validation_index),
                training_class_distribution=_distribution(
                    y_train.iloc[training_positions]
                ),
                validation_class_distribution=_distribution(
                    y_train.iloc[validation_positions]
                ),
            )
        )
    if sorted(validation_coverage) != list(range(len(X_train))):
        raise ComparisonError(
            "Validation folds must cover every training row exactly once."
        )
    return tuple(assignments)


def _candidate_result(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidate: CandidateConfig,
    assignments: Sequence[FoldAssignment],
) -> CandidateResult:
    cv_indices = [
        (
            np.asarray(fold.training_positions, dtype=int),
            np.asarray(fold.validation_positions, dtype=int),
        )
        for fold in assignments
    ]
    estimator = create_training_pipeline(candidate.training_config())
    results = cross_validate(
        estimator,
        X_train.copy(deep=True),
        y_train.copy(deep=True),
        scoring=SCORERS,
        cv=cv_indices,
        n_jobs=None,
        return_estimator=True,
        error_score="raise",
    )
    fitted_estimators = tuple(results["estimator"])
    if len(fitted_estimators) != len(assignments):
        raise ComparisonError("Cross-validation returned an unexpected model count.")
    if len({id(estimator) for estimator in fitted_estimators}) != len(assignments):
        raise ComparisonError("Cross-validation reused a fitted estimator across folds.")
    for fitted in fitted_estimators:
        if tuple(str(value) for value in fitted.feature_names_in_) != FEATURE_COLUMNS:
            raise ComparisonError("A fitted fold model has an incompatible schema.")

    fold_metrics: list[FoldMetrics] = []
    values_by_metric: dict[str, list[float]] = {
        metric: [] for metric in METRIC_NAMES
    }
    for position, assignment in enumerate(assignments):
        metrics = {
            metric: float(results[f"test_{metric}"][position])
            for metric in METRIC_NAMES
        }
        for metric, value in metrics.items():
            values_by_metric[metric].append(value)
        fold_metrics.append(
            FoldMetrics(
                fold=assignment.fold,
                training_rows=len(assignment.training_positions),
                validation_rows=len(assignment.validation_positions),
                training_index_sha256=assignment.training_index_sha256,
                validation_index_sha256=assignment.validation_index_sha256,
                metrics=metrics,
            )
        )
    means = {
        metric: float(statistics.fmean(values))
        for metric, values in values_by_metric.items()
    }
    standard_deviations = {
        metric: float(statistics.pstdev(values))
        for metric, values in values_by_metric.items()
    }
    return CandidateResult(
        candidate=candidate,
        folds=tuple(fold_metrics),
        mean_metrics=means,
        standard_deviation_metrics=standard_deviations,
        fitted_model_count=len(fitted_estimators),
        python_version=platform.python_version(),
        scikit_learn_version=sklearn.__version__,
    )


def rank_candidates(
    candidates: Sequence[CandidateResult], tolerance: float = SELECTION_TOLERANCE
) -> tuple[str, ...]:
    """Rank by macro F1, treating effective ties as a cost comparison."""

    remaining = list(candidates)
    ranking: list[str] = []
    while remaining:
        highest = max(result.mean_metrics[PRIMARY_METRIC] for result in remaining)
        tied = [
            result
            for result in remaining
            if highest - result.mean_metrics[PRIMARY_METRIC] <= tolerance
        ]
        tied.sort(key=lambda result: (result.candidate.n_estimators, result.candidate.name))
        ranking.extend(result.candidate.name for result in tied)
        tied_names = {result.candidate.name for result in tied}
        remaining = [
            result for result in remaining if result.candidate.name not in tied_names
        ]
    return tuple(ranking)


def compare_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    dataset: DatasetLineage,
    config: ExperimentConfig | None = None,
) -> ComparisonResult:
    """Compare candidates using internal CV with no outer-test API surface."""

    effective_config = config if config is not None else ExperimentConfig()
    _validate_comparison_inputs(X_train, y_train, effective_config)
    X_before = X_train.copy(deep=True)
    y_before = y_train.copy(deep=True)
    assignments = build_cv_folds(X_train, y_train, effective_config)
    candidate_results = tuple(
        _candidate_result(X_train, y_train, candidate, assignments)
        for candidate in effective_config.candidates
    )
    if not X_train.equals(X_before) or not y_train.equals(y_before):
        raise ComparisonError("Candidate comparison mutated its training inputs.")
    ranking = rank_candidates(candidate_results, effective_config.tie_tolerance)
    best_observed = sorted(
        candidate_results,
        key=lambda result: (
            -result.mean_metrics[PRIMARY_METRIC],
            result.candidate.name,
        ),
    )[0].candidate.name
    partition = {
        "sha256": training_partition_fingerprint(X_train, y_train),
        "rows": len(X_train),
        "feature_names": list(FEATURE_COLUMNS),
        "class_count": int(y_train.nunique()),
        "class_distribution": _distribution(y_train),
        "outer_holdout_used": False,
        "folds": [assignment.to_dict() for assignment in assignments],
    }
    return ComparisonResult(
        configuration=effective_config,
        dataset=dataset,
        training_partition=partition,
        candidates=candidate_results,
        ranking=ranking,
        best_observed_candidate=best_observed,
        recommended_candidate=ranking[0],
        warnings=(
            "Observed CV differences do not establish statistical superiority.",
            "The outer 440-row holdout was not used for fitting, ranking, or tie-breaking.",
            "Recommendation is not production approval or model promotion.",
        ),
    )


def write_comparison_report(result: ComparisonResult, path: str | Path) -> None:
    """Write stable comparison evidence without MLflow run identifiers."""

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
