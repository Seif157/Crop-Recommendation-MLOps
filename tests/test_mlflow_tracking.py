"""Tests for explicit MLflow logging of the established baseline."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import mlflow
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.pipeline import Pipeline

from crop_recommendation.data.preprocessing import (
    DataSplit,
    SplitConfig,
    prepare_data_split,
    split_dataframe,
)
from crop_recommendation.data.validation import (
    BASELINE_CROP_CLASSES,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    default_dataset_path,
)
from crop_recommendation.tracking.mlflow_tracking import (
    EVALUATION_PROTOCOL,
    MODEL_ARTIFACT_NAME,
    REPORT_ARTIFACT_PATH,
    MlflowTrackingConfig,
    TrackingError,
    TrackingRunResult,
    default_artifact_root,
    default_backend_path,
    log_baseline_experiment,
    sqlite_uri,
)
from crop_recommendation.training.evaluate import EvaluationResult, evaluate_model
from crop_recommendation.training.train import TrainingResult, train_model

pytestmark = pytest.mark.filterwarnings(
    "ignore:The ``noload`` loader strategy is deprecated.*:"
    "sqlalchemy.exc.SADeprecationWarning"
)


@dataclass(frozen=True)
class RecordedRun:
    config: MlflowTrackingConfig
    result: TrackingRunResult
    split: DataSplit
    training: TrainingResult
    evaluation: EvaluationResult
    dataset_path: Path
    tree_ids: tuple[int, ...]
    X_train_before: pd.DataFrame
    y_train_before: pd.Series
    evaluation_before: dict[str, object]


def _synthetic_dataframe() -> pd.DataFrame:
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
                    TARGET_COLUMN: crop,
                }
            )
    return pd.DataFrame(records)


def _tracking_config(directory: Path) -> MlflowTrackingConfig:
    return MlflowTrackingConfig(
        tracking_uri=sqlite_uri(directory / "mlflow.db"),
        artifact_root=directory / "artifacts",
        experiment_name="test-crop-baseline",
        run_name="test-fixed-holdout",
        input_example_rows=3,
    )


@contextmanager
def _active_tracking_uri(uri: str) -> Iterator[None]:
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    try:
        yield
    finally:
        mlflow.set_tracking_uri(previous)


@pytest.fixture(scope="module")
def recorded_run(tmp_path_factory: pytest.TempPathFactory) -> RecordedRun:
    directory = tmp_path_factory.mktemp("mlflow-tracking")
    dataframe = _synthetic_dataframe()
    dataset_path = directory / "crop.csv"
    dataframe.to_csv(dataset_path, index=False)
    split = split_dataframe(dataframe)
    training = train_model(split.X_train, split.y_train)
    evaluation = evaluate_model(training.estimator, split.X_test, split.y_test)
    classifier = training.estimator.named_steps["classifier"]
    tree_ids = tuple(id(tree) for tree in classifier.estimators_)
    X_train_before = split.X_train.copy(deep=True)
    y_train_before = split.y_train.copy(deep=True)
    evaluation_before = evaluation.to_dict()
    config = _tracking_config(directory)

    result = log_baseline_experiment(
        training,
        evaluation,
        split,
        dataset_path,
        config,
    )
    return RecordedRun(
        config=config,
        result=result,
        split=split,
        training=training,
        evaluation=evaluation,
        dataset_path=dataset_path,
        tree_ids=tree_ids,
        X_train_before=X_train_before,
        y_train_before=y_train_before,
        evaluation_before=evaluation_before,
    )


def _stored_run(recorded: RecordedRun):
    client = MlflowClient(tracking_uri=recorded.config.tracking_uri)
    return client.get_run(recorded.result.run_id)


def test_default_storage_paths_do_not_depend_on_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_backend = default_backend_path()
    expected_artifacts = default_artifact_root()

    monkeypatch.chdir(tmp_path)

    assert default_backend_path() == expected_backend
    assert default_artifact_root() == expected_artifacts
    assert expected_backend.is_absolute()
    assert expected_artifacts.is_absolute()


def test_experiment_and_run_are_created(recorded_run: RecordedRun) -> None:
    client = MlflowClient(tracking_uri=recorded_run.config.tracking_uri)
    experiment = client.get_experiment(recorded_run.result.experiment_id)
    run = _stored_run(recorded_run)

    assert experiment.name == recorded_run.config.experiment_name
    assert experiment.artifact_location == recorded_run.config.artifact_root_uri
    assert run.info.status == "FINISHED"
    assert run.info.experiment_id == recorded_run.result.experiment_id
    assert run.data.tags["tracking_run_id"] == recorded_run.result.run_id


def test_real_parameters_are_logged(recorded_run: RecordedRun) -> None:
    parameters = _stored_run(recorded_run).data.params

    assert parameters["model_type"] == "RandomForestClassifier"
    assert parameters["n_estimators"] == "100"
    assert parameters["random_seed"] == "42"
    assert parameters["split_test_size"] == "0.2"
    assert parameters["split_random_state"] == "42"
    assert parameters["estimator__criterion"] == "gini"
    assert parameters["estimator__max_features"] == "sqrt"


def test_real_metrics_are_logged(recorded_run: RecordedRun) -> None:
    metrics = _stored_run(recorded_run).data.metrics
    evaluation = recorded_run.evaluation

    assert metrics["accuracy"] == pytest.approx(evaluation.accuracy)
    assert metrics["macro_precision"] == pytest.approx(evaluation.macro_precision)
    assert metrics["macro_recall"] == pytest.approx(evaluation.macro_recall)
    assert metrics["macro_f1"] == pytest.approx(evaluation.macro_f1)
    assert metrics["test_sample_count"] == evaluation.test_sample_count


def test_dataset_and_software_metadata_are_logged(
    recorded_run: RecordedRun,
) -> None:
    tags = _stored_run(recorded_run).data.tags

    assert tags["evaluation_protocol"] == EVALUATION_PROTOCOL
    assert tags["dataset_sha256"] == recorded_run.result.dataset_sha256
    assert tags["dataset_rows"] == "40"
    assert tags["dataset_columns"] == "8"
    assert json.loads(tags["feature_names"]) == list(FEATURE_COLUMNS)
    assert tags["crop_class_count"] == "4"
    assert tags["training_sample_count"] == "32"
    assert tags["test_sample_count"] == "8"
    assert tags["python_version"]
    assert tags["scikit_learn_version"]
    assert tags["git_revision"]


def test_evaluation_report_artifact_is_complete(
    recorded_run: RecordedRun, tmp_path: Path
) -> None:
    client = MlflowClient(tracking_uri=recorded_run.config.tracking_uri)
    downloaded = client.download_artifacts(
        recorded_run.result.run_id,
        REPORT_ARTIFACT_PATH,
        dst_path=str(tmp_path),
    )
    report = json.loads(Path(downloaded).read_text(encoding="utf-8"))

    assert report["dataset"]["sha256"] == recorded_run.result.dataset_sha256
    assert report["protocol"]["name"] == EVALUATION_PROTOCOL
    expected_training = json.loads(
        json.dumps(recorded_run.training.metadata.to_dict())
    )
    assert report["training"] == expected_training
    assert report["evaluation"] == recorded_run.evaluation.to_dict()
    assert report["tracking"]["run_id"] == recorded_run.result.run_id
    assert len(report["evaluation"]["class_report"]) == 4
    assert len(report["evaluation"]["confusion_matrix"]) == 4


def test_model_signature_and_round_trip(recorded_run: RecordedRun) -> None:
    input_example = recorded_run.split.X_train.iloc[:3]
    expected = recorded_run.training.estimator.predict(input_example).tolist()

    with _active_tracking_uri(recorded_run.config.tracking_uri):
        model_info = mlflow.models.get_model_info(recorded_run.result.model_uri)
        loaded = mlflow.sklearn.load_model(recorded_run.result.model_uri)
    signature = model_info.signature.to_dict()
    input_schema = json.loads(signature["inputs"])

    assert recorded_run.result.model_name == MODEL_ARTIFACT_NAME
    assert recorded_run.result.model_id
    assert [column["name"] for column in input_schema] == list(FEATURE_COLUMNS)
    assert loaded.predict(input_example).tolist() == expected


def test_tracking_does_not_mutate_inputs(recorded_run: RecordedRun) -> None:
    classifier = recorded_run.training.estimator.named_steps["classifier"]

    assert tuple(id(tree) for tree in classifier.estimators_) == recorded_run.tree_ids
    assert_frame_equal(recorded_run.split.X_train, recorded_run.X_train_before)
    assert_series_equal(recorded_run.split.y_train, recorded_run.y_train_before)
    assert recorded_run.evaluation.to_dict() == recorded_run.evaluation_before


def test_tracking_never_fits_and_reuses_the_experiment(
    recorded_run: RecordedRun,
) -> None:
    with patch.object(
        Pipeline,
        "fit",
        side_effect=AssertionError("tracking must never fit"),
    ):
        second = log_baseline_experiment(
            recorded_run.training,
            recorded_run.evaluation,
            recorded_run.split,
            recorded_run.dataset_path,
            recorded_run.config,
        )

    assert second.experiment_id == recorded_run.result.experiment_id
    assert second.run_id != recorded_run.result.run_id


def test_unavailable_backend_has_a_clear_error(
    recorded_run: RecordedRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(*args: object, **kwargs: object) -> None:
        raise MlflowException("tracking backend unavailable")

    monkeypatch.setattr(MlflowClient, "get_experiment_by_name", unavailable)

    with pytest.raises(TrackingError, match="MLflow tracking failed.*unavailable"):
        log_baseline_experiment(
            recorded_run.training,
            recorded_run.evaluation,
            recorded_run.split,
            recorded_run.dataset_path,
            recorded_run.config,
        )


def test_nonbaseline_split_is_rejected(recorded_run: RecordedRun) -> None:
    changed_split = replace(
        recorded_run.split,
        config=SplitConfig(test_size=0.2, random_state=7),
    )

    with pytest.raises(TrackingError, match="established fixed split"):
        log_baseline_experiment(
            recorded_run.training,
            recorded_run.evaluation,
            changed_split,
            recorded_run.dataset_path,
            recorded_run.config,
        )


def test_dataset_must_match_logged_partitions(
    recorded_run: RecordedRun, tmp_path: Path
) -> None:
    changed_dataset = _synthetic_dataframe()
    changed_dataset.loc[0, "N"] = 99
    changed_path = tmp_path / "changed.csv"
    changed_dataset.to_csv(changed_path, index=False)

    with pytest.raises(TrackingError, match="does not match"):
        log_baseline_experiment(
            recorded_run.training,
            recorded_run.evaluation,
            recorded_run.split,
            changed_path,
            recorded_run.config,
        )


@pytest.mark.integration
def test_real_baseline_run_is_recorded_and_loadable(tmp_path: Path) -> None:
    split = prepare_data_split()
    training = train_model(split.X_train, split.y_train)
    evaluation = evaluate_model(training.estimator, split.X_test, split.y_test)
    config = _tracking_config(tmp_path)

    result = log_baseline_experiment(
        training,
        evaluation,
        split,
        default_dataset_path(),
        config,
    )

    client = MlflowClient(tracking_uri=config.tracking_uri)
    stored = client.get_run(result.run_id)
    assert stored.data.metrics["accuracy"] == pytest.approx(evaluation.accuracy)
    assert stored.data.tags["dataset_rows"] == "2200"
    assert stored.data.tags["crop_class_count"] == "22"

    example = split.X_train.iloc[: config.input_example_rows]
    with _active_tracking_uri(config.tracking_uri):
        loaded = mlflow.sklearn.load_model(result.model_uri)
    assert loaded.predict(example).tolist() == training.estimator.predict(
        example
    ).tolist()
