"""Contracts for DVC data tracking and deterministic pipeline orchestration."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import mlflow
import pandas as pd
import pytest
import yaml
from sklearn.pipeline import Pipeline

import crop_recommendation.pipeline.dvc_pipeline as pipeline_module
from crop_recommendation.data.preprocessing import prepare_data_split
from crop_recommendation.data.validation import (
    BASELINE_CROP_CLASSES,
    EXPECTED_COLUMNS,
    FEATURE_COLUMNS,
    default_dataset_path,
    repository_root,
)
from crop_recommendation.pipeline.dvc_pipeline import (
    PipelineArtifactError,
    PipelineConfigurationError,
    PipelineStageError,
    load_pipeline_config,
    load_trusted_pipeline,
    run_evaluation_stage,
    run_training_stage,
    run_validation_stage,
)
from crop_recommendation.tracking import file_sha256
from crop_recommendation.training.evaluate_command import run_baseline_evaluation

EXPECTED_DATASET_SHA256 = (
    "54a5a6e5408668e668667efc50de2fc867c1b875e0431b4f54dd331b0a109a4e"
)
EXPECTED_DATASET_DVC_MD5 = "5e743dd4cf49f84d23a904cd51c9b5b8"


def _file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_params(
    path: Path,
    *,
    test_size: float = 0.2,
    split_seed: int = 42,
    estimators: int = 9,
    model_seed: int = 42,
) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "split": {
                    "test_size": test_size,
                    "random_state": split_seed,
                },
                "model": {
                    "n_estimators": estimators,
                    "random_state": model_seed,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_small_dataset(path: Path) -> None:
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
                    "label": crop,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records, columns=EXPECTED_COLUMNS).to_csv(path, index=False)


@dataclass(frozen=True)
class PipelineArtifacts:
    dataset: Path
    params: Path
    validation: Path
    model: Path
    training: Path
    evaluation: Path
    metrics: Path


@pytest.fixture(scope="module")
def pipeline_artifacts(tmp_path_factory: pytest.TempPathFactory) -> Iterator[PipelineArtifacts]:
    directory = tmp_path_factory.mktemp("dvc-pipeline")
    artifacts = PipelineArtifacts(
        dataset=directory / "crop.csv",
        params=directory / "params.yaml",
        validation=directory / "validation.json",
        model=directory / "model.skops",
        training=directory / "training.json",
        evaluation=directory / "evaluation.json",
        metrics=directory / "metrics.json",
    )
    _write_small_dataset(artifacts.dataset)
    _write_params(artifacts.params)
    run_validation_stage(artifacts.dataset, artifacts.validation)
    run_training_stage(
        artifacts.dataset,
        artifacts.params,
        artifacts.validation,
        artifacts.model,
        artifacts.training,
    )
    run_evaluation_stage(
        artifacts.dataset,
        artifacts.params,
        artifacts.validation,
        artifacts.model,
        artifacts.training,
        artifacts.evaluation,
        artifacts.metrics,
    )
    yield artifacts


def test_dvc_pointer_tracks_canonical_dataset() -> None:
    pointer = repository_root() / "data" / "raw" / "Crop_recommendation.csv.dvc"
    contents = yaml.safe_load(pointer.read_text(encoding="utf-8"))
    output = contents["outs"][0]

    assert default_dataset_path() == pointer.with_suffix("")
    assert output["path"] == "Crop_recommendation.csv"
    assert output["md5"] == EXPECTED_DATASET_DVC_MD5
    assert output["size"] == default_dataset_path().stat().st_size


def test_dataset_hash_is_preserved_and_matches_mlflow_identity() -> None:
    dataset = default_dataset_path()

    assert _file_digest(dataset) == EXPECTED_DATASET_SHA256
    assert _file_digest(dataset, "md5") == EXPECTED_DATASET_DVC_MD5
    assert file_sha256(dataset) == EXPECTED_DATASET_SHA256


def test_default_pipeline_configuration() -> None:
    config = load_pipeline_config(repository_root() / "params.yaml")

    assert config.split.test_size == 0.2
    assert config.split.random_state == 42
    assert config.model.n_estimators == 100
    assert config.model.random_state == 42


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("split: {}\nmodel: {}\n", "missing required key"),
        (
            "split:\n  test_size: 2\n  random_state: 42\n"
            "model:\n  n_estimators: 100\n  random_state: 42\n",
            "Invalid pipeline parameter",
        ),
        (
            "split:\n  test_size: 0.2\n  random_state: 42\n"
            "model:\n  n_estimators: 100\n  random_state: 42\n  extra: true\n",
            "unexpected key",
        ),
    ],
)
def test_invalid_pipeline_configuration(
    tmp_path: Path, contents: str, message: str
) -> None:
    path = tmp_path / "params.yaml"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(PipelineConfigurationError, match=message):
        load_pipeline_config(path)


def test_missing_pipeline_configuration(tmp_path: Path) -> None:
    with pytest.raises(PipelineConfigurationError, match="does not exist"):
        load_pipeline_config(tmp_path / "missing.yaml")


def test_validation_stage_is_successful_and_deterministic(
    pipeline_artifacts: PipelineArtifacts,
) -> None:
    before = pipeline_artifacts.validation.read_bytes()
    report = run_validation_stage(
        pipeline_artifacts.dataset, pipeline_artifacts.validation
    )

    assert report["stage"] == "validate"
    assert report["validation"]["validation_status"] == "valid"
    assert pipeline_artifacts.validation.read_bytes() == before


def test_invalid_dataset_fails_without_output(tmp_path: Path) -> None:
    dataset = tmp_path / "invalid.csv"
    output = tmp_path / "validation.json"
    _write_small_dataset(dataset)
    pd.read_csv(dataset).drop(columns="rainfall").to_csv(dataset, index=False)

    with pytest.raises(PipelineStageError, match="Dataset validation failed"):
        run_validation_stage(dataset, output)

    assert not output.exists()


def test_training_receives_only_train_partition_and_propagates_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "crop.csv"
    params = tmp_path / "params.yaml"
    validation = tmp_path / "validation.json"
    model = tmp_path / "model.skops"
    training_report = tmp_path / "training.json"
    _write_small_dataset(dataset)
    _write_params(params, split_seed=17, estimators=7, model_seed=19)
    run_validation_stage(dataset, validation)
    expected_split = prepare_data_split(
        dataset, load_pipeline_config(params).split
    )
    original_train_model = pipeline_module.train_model
    observed = {"called": False}

    def guarded_train_model(X_train: pd.DataFrame, y_train: pd.Series, config: object):
        observed["called"] = True
        assert X_train.equals(expected_split.X_train)
        assert y_train.equals(expected_split.y_train)
        assert set(X_train.index).isdisjoint(expected_split.X_test.index)
        return original_train_model(X_train, y_train, config)

    monkeypatch.setattr(pipeline_module, "train_model", guarded_train_model)
    report = run_training_stage(
        dataset, params, validation, model, training_report
    )

    assert observed["called"]
    assert report["configuration"]["split"] == {
        "test_size": 0.2,
        "random_state": 17,
    }
    assert report["training"]["model_configuration"] == {
        "n_estimators": 7,
        "random_state": 19,
    }


def test_trusted_model_loads_with_expected_feature_schema(
    pipeline_artifacts: PipelineArtifacts,
) -> None:
    model = load_trusted_pipeline(pipeline_artifacts.model)

    assert isinstance(model, Pipeline)
    assert tuple(model.feature_names_in_) == FEATURE_COLUMNS


def test_training_metadata_is_deterministic(
    pipeline_artifacts: PipelineArtifacts,
) -> None:
    before = pipeline_artifacts.training.read_bytes()

    report = run_training_stage(
        pipeline_artifacts.dataset,
        pipeline_artifacts.params,
        pipeline_artifacts.validation,
        pipeline_artifacts.model,
        pipeline_artifacts.training,
    )

    assert pipeline_artifacts.training.read_bytes() == before
    assert report["partition"]["train_index_sha256"]
    assert report["partition"]["test_index_sha256"]


def test_model_loader_rejects_incompatible_feature_schema(
    pipeline_artifacts: PipelineArtifacts,
) -> None:
    with pytest.raises(PipelineArtifactError, match="feature schema"):
        load_trusted_pipeline(
            pipeline_artifacts.model, tuple(reversed(FEATURE_COLUMNS))
        )


def test_evaluation_never_refits_model(
    pipeline_artifacts: PipelineArtifacts, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_if_fitted(*args: object, **kwargs: object) -> None:
        raise AssertionError("evaluation called fit")

    monkeypatch.setattr(Pipeline, "fit", fail_if_fitted)
    report, metrics = run_evaluation_stage(
        pipeline_artifacts.dataset,
        pipeline_artifacts.params,
        pipeline_artifacts.validation,
        pipeline_artifacts.model,
        pipeline_artifacts.training,
        pipeline_artifacts.evaluation,
        pipeline_artifacts.metrics,
    )

    assert metrics["accuracy"] == report["evaluation"]["accuracy"]


def test_evaluation_metrics_and_reports_are_deterministic(
    pipeline_artifacts: PipelineArtifacts,
) -> None:
    evaluation_before = pipeline_artifacts.evaluation.read_bytes()
    metrics_before = pipeline_artifacts.metrics.read_bytes()
    report, metrics = run_evaluation_stage(
        pipeline_artifacts.dataset,
        pipeline_artifacts.params,
        pipeline_artifacts.validation,
        pipeline_artifacts.model,
        pipeline_artifacts.training,
        pipeline_artifacts.evaluation,
        pipeline_artifacts.metrics,
    )

    assert pipeline_artifacts.evaluation.read_bytes() == evaluation_before
    assert pipeline_artifacts.metrics.read_bytes() == metrics_before
    assert json.loads(metrics_before) == metrics
    assert report["prediction_sha256"]
    assert sum(sum(row) for row in report["evaluation"]["confusion_matrix"]) == 10


def test_missing_model_artifact_fails(
    pipeline_artifacts: PipelineArtifacts, tmp_path: Path
) -> None:
    with pytest.raises(PipelineArtifactError, match="does not exist"):
        run_evaluation_stage(
            pipeline_artifacts.dataset,
            pipeline_artifacts.params,
            pipeline_artifacts.validation,
            tmp_path / "missing.skops",
            pipeline_artifacts.training,
            tmp_path / "evaluation.json",
            tmp_path / "metrics.json",
        )


def test_pipeline_does_not_mutate_source_or_create_mlflow_run(
    pipeline_artifacts: PipelineArtifacts,
) -> None:
    before = _file_digest(pipeline_artifacts.dataset)
    assert mlflow.active_run() is None

    run_evaluation_stage(
        pipeline_artifacts.dataset,
        pipeline_artifacts.params,
        pipeline_artifacts.validation,
        pipeline_artifacts.model,
        pipeline_artifacts.training,
        pipeline_artifacts.evaluation,
        pipeline_artifacts.metrics,
    )

    assert _file_digest(pipeline_artifacts.dataset) == before
    assert mlflow.active_run() is None


@pytest.mark.integration
def test_real_dvc_pipeline_is_current_and_matches_phase_one() -> None:
    root = repository_root()
    first = subprocess.run(
        [sys.executable, "-m", "dvc", "repro"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    second = subprocess.run(
        [sys.executable, "-m", "dvc", "repro"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    status = subprocess.run(
        [sys.executable, "-m", "dvc", "status"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    expected_outputs = [
        root / "dvc.lock",
        root / "artifacts" / "dvc" / "validation.json",
        root / "artifacts" / "dvc" / "model.skops",
        root / "artifacts" / "dvc" / "training.json",
        root / "artifacts" / "dvc" / "evaluation.json",
        root / "artifacts" / "dvc" / "metrics.json",
    ]
    assert all(path.is_file() for path in expected_outputs)
    lock = yaml.safe_load((root / "dvc.lock").read_text(encoding="utf-8"))
    assert set(lock["stages"]) == {"validate", "train", "evaluate"}
    assert "Data and pipelines are up to date" in second.stdout
    assert "Data and pipelines are up to date" in status.stdout

    dvc_report = json.loads(
        (root / "artifacts" / "dvc" / "evaluation.json").read_text(
            encoding="utf-8"
        )
    )
    baseline = run_baseline_evaluation()
    assert dvc_report["evaluation"] == baseline.evaluation.to_dict()
    assert dvc_report["dataset"]["sha256"] == baseline.dataset_sha256
    assert len(dvc_report["predictions"]) == baseline.evaluation.test_sample_count
    assert first.returncode == 0
