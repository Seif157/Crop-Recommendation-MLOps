"""Tests for controlled, training-only candidate comparison and tracking."""

from __future__ import annotations

import inspect
import json
import statistics
from pathlib import Path

import pandas as pd
import pytest
import yaml
from mlflow.tracking import MlflowClient
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.metrics import f1_score

from crop_recommendation.data.preprocessing import prepare_data_split
from crop_recommendation.data.validation import (
    BASELINE_CROP_CLASSES,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    default_dataset_path,
    repository_root,
)
from crop_recommendation.experiments.command import load_dataset_lineage, main
from crop_recommendation.experiments.comparison import (
    CandidateConfig,
    CandidateResult,
    ComparisonError,
    DatasetLineage,
    ExperimentConfig,
    ExperimentConfigurationError,
    build_cv_folds,
    compare_candidates,
    rank_candidates,
)
from crop_recommendation.experiments.tracking import (
    ComparisonTrackingError,
    LoggedComparison,
    log_comparison,
)
from crop_recommendation.tracking.mlflow_tracking import (
    MlflowTrackingConfig,
    sqlite_uri,
)
from crop_recommendation.training.train import create_training_pipeline


@pytest.fixture(scope="module")
def synthetic_training_data() -> tuple[pd.DataFrame, pd.Series]:
    records: list[dict[str, object]] = []
    for class_index, crop in enumerate(BASELINE_CROP_CLASSES[:4]):
        for sample_index in range(15):
            row_id = class_index * 15 + sample_index
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
    frame = pd.DataFrame(records)
    return frame.loc[:, list(FEATURE_COLUMNS)], frame[TARGET_COLUMN]


@pytest.fixture(scope="module")
def small_config() -> ExperimentConfig:
    return ExperimentConfig(
        candidates=(
            CandidateConfig("Tiny-2", 2, 42),
            CandidateConfig("Tiny-3", 3, 42),
            CandidateConfig("Tiny-4", 4, 42),
        ),
        cv_folds=3,
    )


@pytest.fixture(scope="module")
def synthetic_lineage() -> DatasetLineage:
    return DatasetLineage(
        path="synthetic.csv",
        sha256="a" * 64,
        dvc_md5="b" * 32,
        size=1234,
    )


@pytest.fixture(scope="module")
def synthetic_comparison(
    synthetic_training_data: tuple[pd.DataFrame, pd.Series],
    synthetic_lineage: DatasetLineage,
    small_config: ExperimentConfig,
):
    X_train, y_train = synthetic_training_data
    return compare_candidates(X_train, y_train, synthetic_lineage, small_config)


@pytest.fixture(scope="module")
def logged_synthetic_comparison(
    tmp_path_factory: pytest.TempPathFactory,
    synthetic_comparison,
) -> LoggedComparison:
    directory = tmp_path_factory.mktemp("cv-mlflow")
    config = MlflowTrackingConfig(
        tracking_uri=sqlite_uri(directory / "mlflow.db"),
        artifact_root=directory / "artifacts",
        experiment_name="isolated-cv-test",
        run_name="candidate",
    )
    return log_comparison(synthetic_comparison, config)


def test_default_candidate_configuration_is_fixed() -> None:
    config = ExperimentConfig()

    assert [(candidate.name, candidate.n_estimators) for candidate in config.candidates] == [
        ("RF-50", 50),
        ("RF-100", 100),
        ("RF-200", 200),
    ]
    assert config.cv_folds == 5
    assert config.cv_random_state == 42
    assert config.primary_metric == "macro_f1"


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_tree_count(value: object) -> None:
    with pytest.raises(ExperimentConfigurationError, match="n_estimators"):
        CandidateConfig("invalid", value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [1, 0, True, 2.5])
def test_invalid_cv_fold_count(value: object) -> None:
    with pytest.raises(ExperimentConfigurationError, match="cv_folds"):
        ExperimentConfig(cv_folds=value)  # type: ignore[arg-type]


def test_duplicate_candidate_identifiers_fail() -> None:
    with pytest.raises(ExperimentConfigurationError, match="unique"):
        ExperimentConfig(
            candidates=(CandidateConfig("same", 2), CandidateConfig("same", 3))
        )


def test_fold_assignment_is_deterministic(
    synthetic_training_data: tuple[pd.DataFrame, pd.Series],
    small_config: ExperimentConfig,
) -> None:
    X_train, y_train = synthetic_training_data

    first = build_cv_folds(X_train, y_train, small_config)
    second = build_cv_folds(X_train, y_train, small_config)

    assert first == second


def test_folds_are_stratified_and_sized(
    synthetic_training_data: tuple[pd.DataFrame, pd.Series],
    small_config: ExperimentConfig,
) -> None:
    folds = build_cv_folds(*synthetic_training_data, small_config)

    for fold in folds:
        assert len(fold.training_positions) == 40
        assert len(fold.validation_positions) == 20
        assert set(fold.training_class_distribution.values()) == {10}
        assert set(fold.validation_class_distribution.values()) == {5}


def test_fold_partitions_are_disjoint_and_complete(
    synthetic_training_data: tuple[pd.DataFrame, pd.Series],
    small_config: ExperimentConfig,
) -> None:
    X_train, y_train = synthetic_training_data
    folds = build_cv_folds(X_train, y_train, small_config)
    validation_positions: list[int] = []

    for fold in folds:
        training = set(fold.training_positions)
        validation = set(fold.validation_positions)
        assert training.isdisjoint(validation)
        assert training | validation == set(range(len(X_train)))
        validation_positions.extend(validation)

    assert sorted(validation_positions) == list(range(len(X_train)))


def test_feature_schema_must_be_exactly_seven_ordered_columns(
    synthetic_training_data: tuple[pd.DataFrame, pd.Series],
    synthetic_lineage: DatasetLineage,
    small_config: ExperimentConfig,
) -> None:
    X_train, y_train = synthetic_training_data

    with pytest.raises(ComparisonError, match="seven ordered"):
        compare_candidates(
            X_train.loc[:, list(reversed(FEATURE_COLUMNS))],
            y_train,
            synthetic_lineage,
            small_config,
        )


def test_comparison_api_and_cli_have_no_outer_test_access() -> None:
    parameters = inspect.signature(compare_candidates).parameters
    command_source = inspect.getsource(main)

    assert "X_test" not in parameters
    assert "y_test" not in parameters
    assert ".X_test" not in command_source
    assert ".y_test" not in command_source


def test_cross_validation_fits_one_fresh_model_per_fold(
    synthetic_comparison,
    small_config: ExperimentConfig,
) -> None:
    assert all(
        candidate.fitted_model_count == small_config.cv_folds
        for candidate in synthetic_comparison.candidates
    )


def test_stored_macro_f1_matches_independent_fold_calculation(
    synthetic_training_data: tuple[pd.DataFrame, pd.Series],
    small_config: ExperimentConfig,
    synthetic_comparison,
) -> None:
    X_train, y_train = synthetic_training_data
    assignment = build_cv_folds(X_train, y_train, small_config)[0]
    candidate = small_config.candidates[0]
    estimator = create_training_pipeline(candidate.training_config())
    estimator.fit(
        X_train.iloc[list(assignment.training_positions)],
        y_train.iloc[list(assignment.training_positions)],
    )
    predictions = estimator.predict(
        X_train.iloc[list(assignment.validation_positions)]
    )
    expected = f1_score(
        y_train.iloc[list(assignment.validation_positions)],
        predictions,
        average="macro",
        zero_division=0,
    )

    assert synthetic_comparison.candidates[0].folds[0].metrics[
        "macro_f1"
    ] == pytest.approx(expected)


def test_metric_mean_and_population_standard_deviation(synthetic_comparison) -> None:
    candidate = synthetic_comparison.candidates[0]
    fold_values = [fold.metrics["macro_f1"] for fold in candidate.folds]

    assert candidate.mean_metrics["macro_f1"] == pytest.approx(
        statistics.fmean(fold_values)
    )
    assert candidate.standard_deviation_metrics["macro_f1"] == pytest.approx(
        statistics.pstdev(fold_values)
    )


def _ranking_result(name: str, estimators: int, macro_f1: float) -> CandidateResult:
    return CandidateResult(
        candidate=CandidateConfig(name, estimators),
        folds=(),
        mean_metrics={
            "accuracy": macro_f1,
            "macro_precision": macro_f1,
            "macro_recall": macro_f1,
            "macro_f1": macro_f1,
        },
        standard_deviation_metrics={
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
        },
        fitted_model_count=5,
        python_version="test",
        scikit_learn_version="test",
    )


def test_candidate_ranking_is_stable() -> None:
    candidates = (
        _ranking_result("middle", 100, 0.8),
        _ranking_result("best", 200, 0.9),
        _ranking_result("last", 50, 0.7),
    )

    assert rank_candidates(candidates) == ("best", "middle", "last")
    assert rank_candidates(tuple(reversed(candidates))) == (
        "best",
        "middle",
        "last",
    )


def test_effective_tie_prefers_fewer_estimators() -> None:
    candidates = (
        _ranking_result("larger", 200, 0.9000000000005),
        _ranking_result("smaller", 50, 0.9),
    )

    assert rank_candidates(candidates, tolerance=1e-12)[0] == "smaller"


def test_comparison_does_not_mutate_inputs(
    synthetic_training_data: tuple[pd.DataFrame, pd.Series],
    synthetic_lineage: DatasetLineage,
    small_config: ExperimentConfig,
) -> None:
    X_train, y_train = synthetic_training_data
    X_before = X_train.copy(deep=True)
    y_before = y_train.copy(deep=True)

    compare_candidates(X_train, y_train, synthetic_lineage, small_config)

    assert_frame_equal(X_train, X_before)
    assert_series_equal(y_train, y_before)


def test_comparison_is_json_serializable_and_repeatable(
    synthetic_training_data: tuple[pd.DataFrame, pd.Series],
    synthetic_lineage: DatasetLineage,
    small_config: ExperimentConfig,
    synthetic_comparison,
) -> None:
    repeated = compare_candidates(
        *synthetic_training_data, synthetic_lineage, small_config
    )

    assert repeated.to_dict() == synthetic_comparison.to_dict()
    assert json.loads(json.dumps(repeated.to_dict())) == repeated.to_dict()


def test_mlflow_logs_candidate_parameters_and_metrics(
    logged_synthetic_comparison: LoggedComparison,
) -> None:
    client = MlflowClient(tracking_uri=logged_synthetic_comparison.tracking_uri)
    runs = client.search_runs([logged_synthetic_comparison.experiment_id])

    assert len(runs) == 3
    for run in runs:
        assert run.data.params["candidate_name"] in {"Tiny-2", "Tiny-3", "Tiny-4"}
        assert run.data.params["cv_fold_count"] == "3"
        assert "mean_cv_macro_f1" in run.data.metrics
        assert "std_cv_macro_f1" in run.data.metrics
        assert "fold_3_macro_f1" in run.data.metrics


def test_mlflow_logs_dvc_lineage_holdout_tag_and_reports(
    logged_synthetic_comparison: LoggedComparison,
) -> None:
    client = MlflowClient(tracking_uri=logged_synthetic_comparison.tracking_uri)
    for record in logged_synthetic_comparison.runs:
        run = client.get_run(record.run_id)
        artifacts = client.list_artifacts(record.run_id, "reports")
        assert run.data.tags["dataset_sha256"] == "a" * 64
        assert run.data.tags["dvc_dataset_md5"] == "b" * 32
        assert run.data.tags["outer_holdout_used"] == "false"
        assert run.data.tags["production_approval"] == "not_evaluated"
        assert {artifact.path for artifact in artifacts} == {
            "reports/candidate.json",
            "reports/comparison_summary.json",
            "reports/cv_configuration.json",
            "reports/fold_metrics.json",
        }


def test_mlflow_storage_failure_is_explicit(
    tmp_path: Path, synthetic_comparison
) -> None:
    artifact_file = tmp_path / "not-a-directory"
    artifact_file.write_text("occupied", encoding="utf-8")
    config = MlflowTrackingConfig(
        tracking_uri=sqlite_uri(tmp_path / "mlflow.db"),
        artifact_root=artifact_file,
        experiment_name="failing-cv",
        run_name="candidate",
    )

    with pytest.raises(ComparisonTrackingError, match="logging failed"):
        log_comparison(synthetic_comparison, config)


def test_dvc_dataset_identity_and_baseline_configuration_are_preserved() -> None:
    lineage = load_dataset_lineage(default_dataset_path())
    params = yaml.safe_load(
        (repository_root() / "params.yaml").read_text(encoding="utf-8")
    )

    assert lineage.sha256 == (
        "54a5a6e5408668e668667efc50de2fc867c1b875e0431b4f54dd331b0a109a4e"
    )
    assert lineage.dvc_md5 == "5e743dd4cf49f84d23a904cd51c9b5b8"
    assert params == {
        "split": {"test_size": 0.2, "random_state": 42},
        "model": {"n_estimators": 100, "random_state": 42},
    }


@pytest.mark.integration
def test_real_training_partition_candidate_comparison() -> None:
    split = prepare_data_split()
    lineage = load_dataset_lineage(default_dataset_path())

    result = compare_candidates(split.X_train, split.y_train, lineage)

    assert result.training_partition["rows"] == 1_760
    assert result.training_partition["outer_holdout_used"] is False
    assert len(result.candidates) == 3
    assert all(len(candidate.folds) == 5 for candidate in result.candidates)
    assert all(
        fold.training_rows == 1_408 and fold.validation_rows == 352
        for candidate in result.candidates
        for fold in candidate.folds
    )
    assert all(
        set(fold["training_class_distribution"].values()) == {64}
        and set(fold["validation_class_distribution"].values()) == {16}
        for fold in result.training_partition["folds"]
    )
    assert result.recommended_candidate in {"RF-50", "RF-100", "RF-200"}
    assert len(split.X_test) == 440
